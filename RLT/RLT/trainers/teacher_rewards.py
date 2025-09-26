import os
import abc
import gc
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Sequence
from .teacher_base import (
    find_sublist_start_end, extract_and_left_align_from_mask, TeacherReward,
    find_valid_subsequence, find_first_last_one_idxs, log_tensor_info,
    is_tensor, TeacherTrainer,
)
import re
import random


def combine_items(items):
    """
    合并不同类型的项目列表
    
    Args:
        items: 要合并的项目列表
        
    Returns:
        合并后的结果，根据输入类型返回相应的合并结果
    """
    if isinstance(items[0], torch.Tensor):
        # 如果是张量，沿着第0维拼接
        return torch.cat(items, dim=0)

    elif isinstance(items[0], float):
        # 如果是浮点数，直接返回列表
        return items

    elif isinstance(items[0], list):
        # 如果是列表，直接返回
        return items

    elif isinstance(items[0], dict):
        # 如果是字典，递归合并每个键的值
        combined = {}
        for key in items[0]:
            # 提取所有字典中该键的值
            values = [item[key] for item in items]
            # 递归合并这些值
            combined[key] = combine_items(values)
        return combined
    else:
        # 其他类型直接返回
        return items


def combine_list_elements(list_of_lists):
    """
    合并列表的列表中的对应元素
    
    Args:
        list_of_lists: 包含多个列表的列表
        
    Returns:
        合并后的列表，每个位置包含原列表对应位置的合并结果
    """
    n = len(list_of_lists[0])  # 获取内部列表的长度
    result = []
    for i in range(n):
        # 提取所有列表在位置i的元素
        items = [lst[i] for lst in list_of_lists]
        # 合并这些元素
        result.append(combine_items(items))
    return result


def to_torch_tensor(data, device='cpu', dtype=None):
    """
    将各种数据类型转换为PyTorch张量
    
    Args:
        data: 要转换的数据
        device: 目标设备
        dtype: 目标数据类型
        
    Returns:
        转换后的PyTorch张量
    """
    if isinstance(data, torch.Tensor):
        # 如果已经是张量，转移到指定设备和类型
        return data.to(device, dtype=dtype) if dtype else data.to(device)

    if isinstance(data, np.ndarray):
        # 如果是numpy数组，先转换为张量再转移
        tensor = torch.from_numpy(data)
        return tensor.to(device, dtype=dtype) if dtype else tensor.to(device)

    if isinstance(data, (list, tuple)):
        # 如果是列表或元组，创建张量并转移
        tensor = torch.tensor(
            data, dtype=dtype) if dtype else torch.tensor(data)
        return tensor.to(device)

    raise TypeError  # 不支持的数据类型


class TeacherDummyLengthReward(TeacherReward):
    """
    基于长度的虚拟教师奖励类
    根据完成文本的长度计算奖励分数
    """

    def __init__(
        self,
        student_model=None,
        teacher_model=None,
        tokenizer=None,
        negative=False,
    ):
        """
        初始化长度奖励计算器
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            tokenizer: 分词器
            negative: 是否使用负奖励（长度越长奖励越低）
        """  
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.negative = negative
        self.__name__ = 'TeacherDummyLengthReward'

    def link_with_trainer(
            self, trainer, student_model, teacher_model, tokenizer,):
        """
        与训练器建立链接
        
        Args:
            trainer: 训练器实例
            student_model: 学生模型
            teacher_model: 教师模型
            tokenizer: 分词器
        """
        TeacherReward.link_with_trainer(
            self=self,
            trainer=trainer,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
        )

    def __call__(
        self,
        prompts,
        completions,
        student_system_prompts,
        start_think_teacher_tags,
        end_think_teacher_tags,
        start_think_student_tags,
        end_think_student_tags,
        start_solution_tags,
        end_solution_tags,
        think_prefixes,
        think_solution_delimiters,
        questions,
        solutions,
        **kwargs,
    ):
        """
        计算基于长度的奖励
        
        Args:
            completions: 完成文本列表
            其他参数: 各种标签和提示信息
            
        Returns:
            奖励分数列表
        """
        rewards = []
        for completion in completions:
            # 对完成文本进行编码
            encoding = self.tokenizer(completion)
            # 奖励等于token数量
            reward = len(encoding)
            if self.negative:
                # 如果设置为负奖励，取负值
                reward = -1*reward
            rewards.append(reward)
        return rewards


class TeacherKLBasedReward(TeacherReward):
    """
    基于KL散度的教师奖励类
    通过比较教师和学生模型的输出分布来计算奖励
    """

    def __init__(
        self,
        # 模型相关参数
        student_model: Any = None,
        teacher_model: Any = None,
        tokenizer: Any = None,

        # 奖励系数
        answer_log_prob_coeff: float | list = 1.0,  # 答案对数概率系数
        kl_penalty_reward_coeff: float | list = 1.0,  # KL惩罚奖励系数
        
        # 归一化函数
        normalize_log_prob_fn: Optional[Callable | str] = 'exp',  # 对数概率归一化函数
        clip_log_prob: Optional[float] = None,  # 对数概率裁剪阈值
        normalize_kl_fn: Optional[Callable | str] = 'exp_norm',  # KL散度归一化函数
        clip_kl: Optional[float] = None,  # KL散度裁剪阈值

        # 约简函数
        reduction_log_prob_fn: Callable | str | list = 'mean',  # 对数概率约简函数
        reduction_kl_fn: Callable | str | list = 'mean',  # KL散度约简函数

        # KL估计相关
        use_schulman_kl_estimation: bool = False,  # 是否使用Schulman的KL估计方法

        positive_kl_estimation: bool = False,  # 是否使用正KL估计
        not_matched_penalty: float = -1.0,  # 不匹配时的惩罚值

        # 去偏相关
        unbias_teacher_log_probs: Optional[bool] = None,  # 是否去偏教师对数概率

        unbias_student_log_probs_temp: Optional[float] = None,  # 学生对数概率去偏温度

        # 熵相关
        include_teacher_think_entropy: Optional[bool] = None,  # 是否包含教师思考熵

        # 生成相关
        correct_generation_coeff: float = 0.0,  # 正确生成系数
        correct_generation_rollouts: int = 8,  # 正确生成轮数
        generation_kwargs: dict = {},  # 生成参数
        generation_check_stategy: str = 'ground_truth',  # 生成检查策略
        formatting_sub_rewards: list = [],  # 格式化子奖励

        # 其他
        evaluate_refined_solution: bool = False,  # 是否评估精炼解决方案
    ):
        """
        初始化基于KL散度的奖励计算器
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        if isinstance(student_model, str):
            raise NotImplementedError

        # 初始化奖励处理和日志记录
        self.initialize_reward_processing_and_logging(
            answer_log_prob_coeff=answer_log_prob_coeff,
            kl_penalty_reward_coeff=kl_penalty_reward_coeff,
            normalize_log_prob_fn=normalize_log_prob_fn,
            normalize_kl_fn=normalize_kl_fn,
            reduction_log_prob_fn=reduction_log_prob_fn,
            reduction_kl_fn=reduction_kl_fn,
            clip_log_prob=clip_log_prob,
            clip_kl=clip_kl,
        )

        self.use_schulman_kl_estimation = use_schulman_kl_estimation
        self.not_matched_penalty = not_matched_penalty

        self.unbias_teacher_log_probs = unbias_teacher_log_probs
        self.unbias_student_log_probs_temp = unbias_student_log_probs_temp
        if self.unbias_student_log_probs_temp is None:
            self.unbias_student_log_probs_temp = 1
        else:
            assert self.unbias_student_log_probs_temp > 0
        
        self.include_teacher_think_entropy = include_teacher_think_entropy
        if self.include_teacher_think_entropy is None:
            # 默认包含教师思考熵
            self.include_teacher_think_entropy = True

        self.correct_generation_coeff = correct_generation_coeff
        self.correct_generation_rollouts = correct_generation_rollouts
        self.generation_kwargs = generation_kwargs
        self.generation_check_stategy = generation_check_stategy
        if self.correct_generation_coeff != 0.0:
            raise NotImplementedError

        self.formatting_sub_rewards = formatting_sub_rewards
        self.evaluate_refined_solution = evaluate_refined_solution
        if self.formatting_sub_rewards or self.evaluate_refined_solution:
            raise NotImplementedError

        self.__name__ = 'TeacherKLBasedReward'

    def used_device(self, ):
        """
        获取使用的设备
        
        Returns:
            设备信息
        """
        teacher_device = self.teacher_model.device
        return teacher_device
        if str(teacher_device) == 'meta':
            return 'cuda'
        else:
            return teacher_device

    def initialize_reward_processing_and_logging(
            self,
            answer_log_prob_coeff,
            kl_penalty_reward_coeff,
            normalize_log_prob_fn,
            normalize_kl_fn,
            reduction_log_prob_fn,
            reduction_kl_fn,
            clip_log_prob,
            clip_kl,
    ):
        """
        初始化奖励处理和日志记录相关参数
        
        Args:
            answer_log_prob_coeff: 答案对数概率系数
            kl_penalty_reward_coeff: KL惩罚奖励系数
            normalize_log_prob_fn: 对数概率归一化函数
            normalize_kl_fn: KL散度归一化函数
            reduction_log_prob_fn: 对数概率约简函数
            reduction_kl_fn: KL散度约简函数
            clip_log_prob: 对数概率裁剪阈值
            clip_kl: KL散度裁剪阈值
        """
        # 处理答案对数概率系数
        self.answer_log_prob_coeff = answer_log_prob_coeff
        if isinstance(self.answer_log_prob_coeff, Sequence):
            self.use_answer_log_prob_coeff = True
            self.answer_log_prob_coeff = torch.tensor(
                self.answer_log_prob_coeff)
            # 将系数移动到合适的设备
            if self.teacher_model is not None:
                self.answer_log_prob_coeff = self.answer_log_prob_coeff.to(
                    self.teacher_model.device)
            elif self.student_model is not None:
                self.answer_log_prob_coeff = self.answer_log_prob_coeff.to(
                    self.student_model.device)
            assert (self.answer_log_prob_coeff.shape[-1] ==
                    len(reduction_log_prob_fn))
        else:
            self.use_answer_log_prob_coeff = answer_log_prob_coeff > 0
        
        # 处理KL惩罚奖励系数
        self.kl_penalty_reward_coeff = kl_penalty_reward_coeff
        if isinstance(self.kl_penalty_reward_coeff, Sequence):
            self.use_kl_penalty_reward_coeff = True
            self.kl_penalty_reward_coeff = torch.tensor(
                self.kl_penalty_reward_coeff)
            # 将系数移动到合适的设备
            if self.teacher_model is not None:
                self.kl_penalty_reward_coeff = self.kl_penalty_reward_coeff.to(
                    self.teacher_model.device)
            elif self.student_model is not None:
                self.kl_penalty_reward_coeff = self.kl_penalty_reward_coeff.to(
                    self.student_model.device)
            assert (self.kl_penalty_reward_coeff.shape[-1] ==
                    len(reduction_kl_fn))
        else:
            self.use_kl_penalty_reward_coeff = kl_penalty_reward_coeff > 0

        # 初始化奖励处理函数
        self.initialize_reward_processing_fns(
            normalize_log_prob_fn=normalize_log_prob_fn,
            normalize_kl_fn=normalize_kl_fn,
            reduction_log_prob_fn=reduction_log_prob_fn,
            reduction_kl_fn=reduction_kl_fn,
            clip_log_prob=clip_log_prob,
            clip_kl=clip_kl,
        )

    def initialize_reward_processing_fns(
            self,
            normalize_log_prob_fn,
            normalize_kl_fn,
            reduction_log_prob_fn,
            reduction_kl_fn,
            clip_log_prob,
            clip_kl,
    ):
        """
        初始化奖励处理函数
        
        Args:
            normalize_log_prob_fn: 对数概率归一化函数
            normalize_kl_fn: KL散度归一化函数
            reduction_log_prob_fn: 对数概率约简函数
            reduction_kl_fn: KL散度约简函数
            clip_log_prob: 对数概率裁剪阈值
            clip_kl: KL散度裁剪阈值
        """
        if clip_log_prob is not None:
            # 对数概率裁剪值需要取负数
            clip_log_prob = -1*clip_log_prob

        # 创建对数概率归一化函数
        self.normalize_log_prob_fn = self._make_normalize_fn(
            normalize_log_prob_fn,
            temp=1,
            clip_min=clip_log_prob,
        )

        # 创建KL散度归一化函数
        self.normalize_kl_fn = self._make_normalize_fn(
            normalize_kl_fn,
            temp=1,
            clip_max=clip_kl,
        )

        # 创建对数概率约简函数
        self.reduction_log_prob_fn, self.log_lp_names = self._make_reduction_fn(
            reduction_log_prob_fn, function_log_name='answer_log_prob')

        # 创建KL散度约简函数
        self.reduction_kl_fn, self.log_kl_names = self._make_reduction_fn(
            reduction_kl_fn, function_log_name='reasoning_kl')

        # 初始化日志记录的约简函数
        self.initialize_reductions_to_log()

    def initialize_reductions_to_log(self,):
        """
        初始化用于日志记录的约简函数
        """
        # 定义要记录的约简类型
        reductions_to_log = ['mean', 'sum', 'min', 'max', 'median',
                             'first_quartile', 'last_quartile']
        
        # 创建KL散度日志约简函数
        self.log_reduction_kl_fn, self.log_reduction_kl_log_names = (
            self._make_reduction_fn(reductions_to_log,
                                    'unprocessed_thought_kl/'))
        
        # 创建对数概率日志约简函数
        self.log_reduction_prob_fn, self.log_reduction_prob_log_names = (
            self._make_reduction_fn(reductions_to_log,
                                    'unprocessed_answer_log_prob/'))

    def get_components_dictionaries_to_log(
            self, kl, log_probs, teacher_mask, student_solution_masks):
        """
        获取用于日志记录的组件字典
        
        Args:
            kl: KL散度张量
            log_probs: 对数概率张量
            teacher_mask: 教师掩码
            student_solution_masks: 学生解决方案掩码
            
        Returns:
            包含日志信息的字典
        """
        full_dict = {}
        
        if kl is not None:
            # 计算KL散度的约简值
            reduced_kl = self.log_reduction_kl_fn(x=kl, mask=teacher_mask)
            kl_scores_to_log = {n: reduced_kl[..., i] for i, n in
                                enumerate(self.log_reduction_kl_log_names)}
            full_dict.update(kl_scores_to_log)
        
        if log_probs is not None:
            # 计算对数概率的约简值
            reduced_log_probs = self.log_reduction_prob_fn(
                x=log_probs, mask=student_solution_masks)
            log_prob_scores_to_log = {n: reduced_log_probs[..., i] for i, n in
                                      enumerate(self.log_reduction_prob_log_names)}
            full_dict.update(log_prob_scores_to_log)
        
        return full_dict

    def _print_debugging_logs(self, to_print: str):
        """
        打印调试日志
        
        Args:
            to_print: 要打印的字符串
        """
        self.trainer._print_debugging_logs(to_print=to_print)

    def get_student_chats_and_relevant_num_tokens(
            self,
            completions,
            student_system_prompts,
            questions,
            solutions,
            start_think_teacher_tags,
            end_think_teacher_tags,
            start_think_student_tags,
            end_think_student_tags,
            start_solution_tags,
            end_solution_tags,
            think_prefixes,
            think_solution_delimiters,
    ):
        """
        获取学生对话和相关token数量
        
        Args:
            completions: 完成文本列表
            student_system_prompts: 学生系统提示列表
            questions: 问题列表
            solutions: 解决方案列表
            各种标签参数: 用于标记思考和解决方案的开始/结束标签
            
        Returns:
            包含学生对话、匹配奖励和各种索引信息的元组
        """
        match_reward = []  # 匹配奖励列表
        chats = []  # 学生对话列表
        teacher_completion_list = []  # 教师完成文本列表
        start_end_teacher_thought_idxs_list = []  # 教师思考索引列表
        start_end_student_thought_idxs_list = []  # 学生思考索引列表
        start_end_student_solution_idxs_list = []  # 学生解决方案索引列表
        
        # 创建迭代器
        chat_iterator = zip(
            completions,
            student_system_prompts,
            questions,
            solutions,
            start_think_teacher_tags,
            end_think_teacher_tags,
            start_think_student_tags,
            end_think_student_tags,
            start_solution_tags,
            end_solution_tags,
            think_prefixes,
            think_solution_delimiters,
        )

        for batch in chat_iterator:
            # 解包批次数据
            completion, student_system_prompt, question, solution = batch[:4]

            # 转义标签以用于正则表达式
            (start_think_teacher_tag, end_think_teacher_tag,
             start_think_student_tag, end_think_student_tag,
             start_solution_tag, end_solution_tag,) = [
                 re.escape(tag) for tag in batch[4:-2]]

            # 保存未转义的标签
            (start_think_teacher_tag_no_esc, end_think_teacher_no_esc,
             start_think_student_tag_no_esc, end_think_student_tag_no_esc,
             ) = batch[4:8]

            think_prefix, think_solution_delimiter = batch[-2:]
            reward_match = 0.0
            
            # 创建思考模式的正则表达式
            think_pattern = (
                start_think_teacher_tag + r"(.*?)" + end_think_teacher_tag
            )

            # 在完成文本中搜索教师思考内容
            teacher_thought_match = re.search(
                think_pattern, completion, flags=re.DOTALL)

            if not teacher_thought_match:
                # 如果没有找到匹配，添加惩罚并尝试修复
                reward_match += self.not_matched_penalty
                completion = completion + end_think_teacher_no_esc
                teacher_thought_match = re.search(
                    think_pattern, completion, flags=re.DOTALL)
                if not teacher_thought_match:
                    # 再次尝试修复
                    completion = start_think_teacher_tag_no_esc + completion
                    teacher_thought_match = re.search(
                        think_pattern, completion, flags=re.DOTALL)
                    assert teacher_thought_match

            match_reward.append(reward_match)

            # 获取思考内容的起始和结束位置
            start_teacher_thought = teacher_thought_match.start(1)
            end_teacher_thought = teacher_thought_match.end(1)

            # 对完成文本进行编码
            completion_tokens = self.tokenizer.encode(completion)

            # 提取思考内容并编码
            thought_content = completion[
                start_teacher_thought:end_teacher_thought]
            thought_tokens_orig = self.tokenizer.encode(thought_content)

            # 在完成文本中找到有效的思考token子序列
            thought_tokens = find_valid_subsequence(
                sub=thought_tokens_orig, seq=completion_tokens)

            # 找到思考token在完成文本中的位置
            start_end_teacher_thought_idxs = find_sublist_start_end(
                completion_tokens,
                thought_tokens,
                from_end=True,
                reverse_search=False,
            )
            
            # 处理tokenization错误
            while start_end_teacher_thought_idxs is None:
                print('Tokenization error: Missing thought tokens in teacher '
                      'completion.')
                print(completion_tokens)
                print(thought_tokens)
                for _ in range(3):
                    print('='*20)
                print('completion')
                print(completion)
                for _ in range(3):
                    print('='*20)
                print('thought_content')
                print(thought_content)
                
                # 尝试缩短思考token
                thought_tokens_orig = thought_tokens_orig[:-1]
                thought_tokens = find_valid_subsequence(
                    sub=thought_tokens_orig, seq=completion_tokens)

                start_end_teacher_thought_idxs = find_sublist_start_end(
                    completion_tokens,
                    thought_tokens,
                    from_end=True,
                    reverse_search=False,
                )
                
                # 重新构建内容
                thought_content = self.tokenizer.decode(thought_tokens_orig)
                completion = (
                    start_think_teacher_tag + thought_content +
                    end_think_teacher_tag
                )

            start_end_teacher_thought_idxs_list.append(
                start_end_teacher_thought_idxs)

            teacher_completion_list.append(completion)

            # 构建学生完成文本
            student_completion = (
                think_prefix + start_think_student_tag_no_esc + thought_content
                + end_think_student_tag_no_esc + think_solution_delimiter
                + solution)

            # 构建学生对话消息
            student_chat_messages = [
                {
                    "role": "system",
                    "content": student_system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": student_completion,
                },
            ]
            
            # 应用对话模板
            student_chat = self.tokenizer.apply_chat_template(
                student_chat_messages,
                tokenize=False,
                continue_final_message=False,
            )

            # 编码学生对话
            student_chat_tokens = self.tokenizer.encode(student_chat)

            # 在学生对话中找到思考token的位置
            start_end_student_thought_idxs = find_sublist_start_end(
                student_chat_tokens,
                thought_tokens,
                from_end=True,
                reverse_search=False,
            )

            if start_end_student_thought_idxs is None:
                print('Tokenization error: Missing thought tokens in student '
                      'chat.')
                print(student_chat_tokens)
                print(thought_tokens)
                raise NotImplementedError

            start_end_student_thought_idxs_list.append(
                start_end_student_thought_idxs)

            # 处理解决方案标签
            solution_pattern = (
                start_solution_tag + r"(.*?)" + end_solution_tag)
            student_solution_match = re.search(
                solution_pattern, solution, flags=re.DOTALL)

            assert student_solution_match

            # 提取解决方案内容
            sol_start = student_solution_match.start(1)
            sol_end = student_solution_match.end(1)

            solution_without_tags = solution[sol_start:sol_end]
            solution_tokens = self.tokenizer.encode(solution_without_tags)
            solution_tokens = find_valid_subsequence(
                sub=solution_tokens, seq=student_chat_tokens,)
            
            # 在学生对话中找到解决方案token的位置
            start_end_student_solution_idxs = find_sublist_start_end(
                student_chat_tokens,
                solution_tokens,
                from_end=True,
                # 从后往前搜索解决方案
                reverse_search=True,
            )

            if start_end_student_solution_idxs is None:
                print('Tokenization error: Missing solution tokens in student '
                      'chat.')
                print(student_chat_tokens)
                print(solution_tokens)
                raise NotImplementedError

            start_end_student_solution_idxs_list.append(
                start_end_student_solution_idxs)

            chats.append(student_chat)

        return (chats,
                match_reward,
                teacher_completion_list,
                start_end_teacher_thought_idxs_list,
                start_end_student_thought_idxs_list,
                start_end_student_solution_idxs_list)

    def link_with_trainer(
            self, trainer, student_model, teacher_model, tokenizer,):
        """
        与训练器建立链接
        
        Args:
            trainer: 训练器实例
            student_model: 学生模型
            teacher_model: 教师模型
            tokenizer: 分词器
        """
        TeacherReward.link_with_trainer(
            self=self,
            trainer=trainer,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
        )
        
        # 设置教师对数概率去偏参数
        if self.unbias_teacher_log_probs is None:
            self.unbias_teacher_log_probs = True

        if self.unbias_teacher_log_probs:
            # 使用训练器的生成温度
            self.teacher_gen_temperature = trainer.gen_temperature
        else:
            # 使用默认温度
            self.teacher_gen_temperature = 1

        # 将系数移动到正确的设备
        if is_tensor(self.kl_penalty_reward_coeff):
            self.kl_penalty_reward_coeff = self.kl_penalty_reward_coeff.to(
                self.teacher_model.device)
        if is_tensor(self.answer_log_prob_coeff):
            self.answer_log_prob_coeff = self.answer_log_prob_coeff.to(
                self.teacher_model.device)

    def get_mask_for_spans(self, start_end_idxs_list, seq_len, device):
        """
        为指定的span创建掩码
        
        Args:
            start_end_idxs_list: 包含(start, end)索引对的列表
            seq_len: 序列长度
            device: 设备
            
        Returns:
            布尔掩码张量
        """
        # 处理负索引
        start_end_idxs_list = [
            (s if s >= 0 else seq_len + s, e if e >= 0 else seq_len + e)
            for s, e in start_end_idxs_list
        ]

        bsz = len(start_end_idxs_list)
        # 创建位置张量
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(dim=0).expand(bsz, seq_len)
        
        # 创建起始和结束位置张量
        starts = torch.tensor(
            [s for s, _ in start_end_idxs_list], device=device)
        ends = torch.tensor([e for _, e in start_end_idxs_list], device=device)
        
        # 创建掩码：位置在[start, end)范围内的为True
        mask = ((positions >= starts.unsqueeze(1)) &
                (positions < ends.unsqueeze(1)))
        return mask

    def estimate_kl(self, p_log_probs, q_log_probs, use_schulman_kl_estimation):
        """
        估计KL散度
        
        Args:
            p_log_probs: 分布P的对数概率
            q_log_probs: 分布Q的对数概率
            use_schulman_kl_estimation: 是否使用Schulman的KL估计方法
            
        Returns:
            KL散度估计值
        """
        if use_schulman_kl_estimation is None:
            use_schulman_kl_estimation = self.use_schulman_kl_estimation

        # 基本KL散度计算
        kl = p_log_probs - q_log_probs
        
        if use_schulman_kl_estimation:
            # Schulman的无偏KL估计：KL(p||q) ≈ log(p/q) - 1 + q/p
            kl = kl - 1 + torch.exp(-kl)
        return kl

    @torch.no_grad()
    def compute_batch_log_probs(
            self, text, student_model=True, cached_log_probs=None, temperature=1.0):
        """
        计算批次文本的对数概率
        
        Args:
            text: 输入文本列表
            student_model: 是否使用学生模型
            cached_log_probs: 缓存的对数概率
            temperature: 温度参数
            
        Returns:
            token级别的对数概率张量
        """
        if student_model:
            model = self.student_model
        else:
            model = self.teacher_model

        # 编码文本
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(model.device)

        # 如果有缓存的对数概率，直接使用
        if cached_log_probs is not None:
            encoding_shape = encoding.input_ids.shape
            expected_num_tokens = encoding_shape[-1] - 1
            assert expected_num_tokens == cached_log_probs.shape[-1]
            cached_log_probs_tensor = to_torch_tensor(
                cached_log_probs, device=model.device)
            return cached_log_probs_tensor.view(
                *encoding_shape[:-1], encoding_shape[-1] - 1)

        # 前向传播获取logits
        outputs = model(**encoding)
        logits = outputs.logits[:, :-1, :]  # 去掉最后一个位置
        labels = encoding.input_ids[:, 1:]  # 去掉第一个位置（通常是BOS token）
        single_token_log_probs = []

        # 为每个样本计算对数概率
        for i in range(logits.size(0)):
            # 应用温度并计算log softmax
            b_log_probs = F.log_softmax(logits[i]/temperature, dim=-1)
            b_labels = labels[i].unsqueeze(-1)
            # 收集对应token的对数概率
            b_token_log_probs = b_log_probs.gather(1, b_labels).squeeze(-1)
            single_token_log_probs.append(b_token_log_probs)

        # 堆叠所有样本的结果
        token_log_probs = torch.stack(single_token_log_probs, dim=0)
        return token_log_probs

    @torch.no_grad()
    def compute_batch_log_probs_with_logits(
            self, text, student_model=True, cached_logits=None, temperature=1.0):
        """
        计算批次文本的对数概率，同时返回logits
        
        Args:
            text: 输入文本列表
            student_model: 是否使用学生模型
            cached_logits: 缓存的logits
            temperature: 温度参数
            
        Returns:
            (对数概率张量, logits张量)的元组
        """
        if student_model:
            model = self.student_model
        else:
            model = self.teacher_model

        # 编码文本
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(model.device)

        if cached_logits is not None:
            # 使用缓存的logits
            encoding_shape = encoding.input_ids.shape
            expected_num_tokens = encoding_shape[-1] - 1
            assert expected_num_tokens == cached_logits.shape[-1]
            logits = to_torch_tensor(
                cached_logits, device=model.device)
        else:
            # 计算logits
            outputs = model(**encoding)
            logits = outputs.logits[:, :-1, :]

        labels = encoding.input_ids[:, 1:]
        single_token_log_probs = []

        # 为每个样本计算对数概率
        for i in range(logits.size(0)):
            scaled_logits = logits[i] / temperature
            # 将logits移动到CPU以节省GPU内存
            logits = logits.detach().cpu()
            b_log_probs = F.log_softmax(scaled_logits, dim=-1)
            b_labels = labels[i].unsqueeze(-1)
            b_token_log_probs = b_log_probs.gather(1, b_labels).squeeze(-1)
            single_token_log_probs.append(b_token_log_probs)

        token_log_probs = torch.stack(single_token_log_probs, dim=0)
        return token_log_probs, logits

    @torch.no_grad()
    def compute_split_batch_log_probs(
        self, text, student_model=True, cached_log_probs=None,
        max_sequence_tokens_to_process=4096,
    ):
        """
        分块计算批次文本的对数概率（用于处理长序列）
        
        Args:
            text: 输入文本列表
            student_model: 是否使用学生模型
            cached_log_probs: 缓存的对数概率
            max_sequence_tokens_to_process: 每次处理的最大token数
            
        Returns:
            token级别的对数概率张量
        """
        if student_model:
            model = self.student_model
        else:
            model = self.teacher_model

        # 编码文本
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(model.device)

        # 如果有缓存，直接返回
        if cached_log_probs is not None:
            expected_num_tokens = encoding.shape[-1] - 1
            assert expected_num_tokens == cached_log_probs.shape[-1]
            return to_torch_tensor(cached_log_probs, device=model.device)

        input_ids = encoding.input_ids
        attention_mask = encoding.get('attention_mask', None)
        batch_size, seq_length = input_ids.shape

        token_log_probs_list = []
        offset = 0
        current_pos = 0
        total_valid = seq_length - 1

        # 判断是否需要分块处理
        multiple_chunks = max_sequence_tokens_to_process < seq_length

        use_cache = False  # 是否使用KV缓存
        past_key_values = None

        # 分块处理序列
        while current_pos < seq_length:
            end_pos = min(
                seq_length, current_pos + max_sequence_tokens_to_process)
            
            if use_cache:
                # 使用缓存时只处理新的token
                chunk_ids = input_ids[:, current_pos:end_pos]
            else:
                # 不使用缓存时处理从开始到当前位置的所有token
                chunk_ids = input_ids[:, :end_pos]
            
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :end_pos]
            else:
                chunk_mask = None

            chunk_len = end_pos - current_pos
            
            # 前向传播
            outputs = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=None if current_pos == 0 else past_key_values,
                use_cache=use_cache,
                num_logits_to_keep=chunk_len,
            )
            
            if hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values

            # 计算有效的token数量
            available = total_valid - offset
            used = min(chunk_len, available)

            # 获取有效的logits
            valid_chunk_logits = outputs.logits[:, :used, :]

            # 获取对应的标签
            valid_labels = input_ids[:, current_pos+1:current_pos+used+1]

            # 计算对数概率
            chunk_log_probs = F.log_softmax(valid_chunk_logits, dim=-1)
            gathered = chunk_log_probs.gather(2, valid_labels.unsqueeze(-1))
            gathered = gathered.squeeze(-1)
            token_log_probs_list.append(gathered)

            # 清理内存
            outputs = None
            valid_chunk_logits = None
            chunk_log_probs = None
            gc.collect()
            torch.cuda.empty_cache()

            offset += used
            current_pos = end_pos

        # 清理KV缓存
        if past_key_values is not None:
            del past_key_values
        gc.collect()
        torch.cuda.empty_cache()

        # 拼接所有块的结果
        token_log_probs = torch.cat(token_log_probs_list, dim=1)
        return token_log_probs

    def process_single_reward(
            self,
            chat,
            match_reward,
            teacher_completion,
            start_end_teacher_thought_idxs,
            start_end_student_thought_idxs,
            start_end_student_solution_idxs,
            use_schulman_unbiased_estimate=None,
            include_teacher_think_entropy=True,
            # 返回选项
            return_info_dict=False,
            return_raw_tensors=False,
            # 缓存选项
            cached_student_log_probs=None,
            cached_teacher_log_probs=None,
            cached_thought_tokens_kl=None,):
        """
        处理单个样本的奖励计算
        
        Args:
            chat: 学生对话文本
            match_reward: 匹配奖励
            teacher_completion: 教师完成文本
            start_end_teacher_thought_idxs: 教师思考索引
            start_end_student_thought_idxs: 学生思考索引
            start_end_student_solution_idxs: 学生解决方案索引
            use_schulman_unbiased_estimate: 是否使用Schulman无偏估计
            include_teacher_think_entropy: 是否包含教师思考熵
            return_info_dict: 是否返回信息字典
            return_raw_tensors: 是否返回原始张量
            cached_student_log_probs: 缓存的学生对数概率
            cached_teacher_log_probs: 缓存的教师对数概率
            cached_thought_tokens_kl: 缓存的思考token KL散度
            
        Returns:
            奖励列表，可选的日志字典和张量字典
        """
        self._print_debugging_logs('computing student logprobs')

        # 计算学生对数概率
        student_log_probs = self.compute_batch_log_probs(
            text=chat,
            student_model=True,
            cached_log_probs=cached_student_log_probs,
            temperature=self.unbias_student_log_probs_temp,
        )

        student_device = self.student_model.device

        self._print_debugging_logs('computing student solution masks')
        # 创建学生解决方案掩码
        student_solution_masks = self.get_mask_for_spans(
            start_end_student_solution_idxs,
            seq_len=student_log_probs.shape[-1],
            device=student_device,
        )

        # 初始化返回张量字典
        tensors_to_return_dict = dict(chats=chat)
        if return_raw_tensors:
            tensors_to_return_dict.update(dict(
                student_log_probs=student_log_probs.clone().detach().squeeze(
                    dim=0).cpu(),
                student_solution_masks=(
                    student_solution_masks.clone().detach().squeeze(
                        dim=0).cpu()),
            ))

        # 计算答案对数概率奖励
        if self.use_answer_log_prob_coeff:
            self._print_debugging_logs('processing student logprobs')
            # 归一化对数概率， 将原始的对数概率转换为更适合计算奖励的形式
            processed_log_probs = self.normalize_log_prob_fn(
                x=student_log_probs)

            # 只计算解决方案部分的平均概率
            log_prob_scores = self.reduction_log_prob_fn(
                x=processed_log_probs, mask=student_solution_masks)

            # 处理NaN值，将所有NaN值替换为惩罚值
            log_prob_scores = torch.nan_to_num(
                log_prob_scores,
                nan=self.not_matched_penalty,
            )

            # 生成最终的对数概率奖励
            log_prob_reward = (
                log_prob_scores*self.answer_log_prob_coeff).sum(-1)

        else:
            raise NotImplementedError

        # 获取用于日志记录的组件字典
        unprocessed_dict = self.get_components_dictionaries_to_log(
            kl=None,
            log_probs=student_log_probs,
            teacher_mask=None,
            student_solution_masks=student_solution_masks,
        )

        self._print_debugging_logs('computing student thought masks')
        # 创建学生思考掩码
        student_thought_masks = self.get_mask_for_spans(
            start_end_student_thought_idxs,
            seq_len=student_log_probs.shape[-1],
            device=student_device,
        )
        
        # 提取并左对齐学生思考token的对数概率
        student_log_probs, student_mask = extract_and_left_align_from_mask(
            student_log_probs, student_thought_masks)

        # 计算KL惩罚奖励
        if self.use_kl_penalty_reward_coeff or return_raw_tensors:

            teacher_device = self.teacher_model.device

            self._print_debugging_logs('computing teacher log probs')

            # 检查是否重用缓存的KL
            reused_cached_kl = cached_thought_tokens_kl is None

            reused_cached_kl = False
            if not reused_cached_kl:
                # 计算教师对数概率
                teacher_log_probs = self.compute_batch_log_probs(
                    text=teacher_completion, student_model=False,
                    cached_log_probs=cached_teacher_log_probs,
                    temperature=self.teacher_gen_temperature,
                )

                if return_raw_tensors:
                    tensors_to_return_dict.update(dict(
                        teacher_log_probs=(
                            teacher_log_probs.clone().detach().squeeze(
                                dim=0).cpu()),
                    ))

                self._print_debugging_logs(
                    'computing teacher thought masks')
                # 创建教师思考掩码
                teacher_thought_masks = self.get_mask_for_spans(
                    start_end_teacher_thought_idxs,
                    seq_len=teacher_log_probs.shape[-1],
                    device=teacher_device,
                )

                self._print_debugging_logs('aligning and extracting tokens')

                # 提取并左对齐教师思考token的对数概率
                teacher_log_probs, teacher_mask = (
                    extract_and_left_align_from_mask(
                        teacher_log_probs, teacher_thought_masks))

                self._print_debugging_logs('computing KL')
                # 计算思考token的KL散度
                thought_tokens_kl = self.estimate_kl(
                    p_log_probs=teacher_log_probs,
                    q_log_probs=student_log_probs,
                    use_schulman_kl_estimation=(
                        use_schulman_unbiased_estimate),
                )

                # 确保教师和学生掩码一致
                assert torch.all(teacher_mask == student_mask)

            else:
                # 使用缓存的KL散度
                thought_tokens_kl = to_torch_tensor(
                    cached_thought_tokens_kl, device=student_log_probs.device)
                assert (
                    thought_tokens_kl.shape[-1] == student_log_probs.shape[-1])
                thought_tokens_kl = thought_tokens_kl.view_as(
                    student_log_probs)
                teacher_mask = student_mask

            # 更新日志字典
            unprocessed_dict.update(self.get_components_dictionaries_to_log(
                # KL散度和掩码信息
                kl=thought_tokens_kl,
                log_probs=None,
                teacher_mask=teacher_mask,
                student_solution_masks=None,
            ))

            if return_raw_tensors:
                tensors_to_return_dict.update(dict(
                    thought_tokens_kl=(
                        thought_tokens_kl.clone().detach().squeeze(
                            dim=0).cpu()),
                    teacher_mask=teacher_mask.clone().detach().squeeze(
                        dim=0).cpu(),
                ))

            self._print_debugging_logs('processing KL')
            # 处理KL散度
            processed_kl = self.normalize_kl_fn(x=thought_tokens_kl)
            kl_scores = self.reduction_kl_fn(x=processed_kl, mask=teacher_mask)
            kl_scores = torch.nan_to_num(
                kl_scores,
                nan=-1*self.not_matched_penalty,
            )
            kl_reward = (kl_scores*self.kl_penalty_reward_coeff).sum(-1)
        else:
            # 如果不使用KL惩罚，设置默认值
            thought_tokens_kl = None
            processed_kl = 0.0
            kl_scores = 0.0
            kl_reward = torch.zeros_like(log_prob_reward)

        # KL奖励取负值（因为我们想要最小化KL散度）
        kl_reward = kl_reward*-1
        match_reward = torch.tensor(
            match_reward, device=log_prob_reward.device)

        # 确保所有奖励组件的形状一致
        assert log_prob_reward.shape == match_reward.shape
        assert kl_reward.shape == match_reward.shape
        
        # 计算总奖励
        reward = log_prob_reward + kl_reward + match_reward

        # 准备日志记录的分数
        log_prob_scores_to_log = {
            n: log_prob_scores[..., i] for i, n in enumerate(self.log_lp_names)}
        kl_scores_to_log = {
            n: kl_scores[..., i] for i, n in enumerate(self.log_kl_names)}

        # 计算无熵版本的奖励（用于比较）
        processed_kl_no_entropy = self.normalize_kl_fn(x=-student_log_probs)
        kl_scores_no_entropy = self.reduction_kl_fn(
            x=processed_kl_no_entropy, mask=student_mask)
        kl_scores_no_entropy = torch.nan_to_num(
            kl_scores_no_entropy,
            nan=self.not_matched_penalty,
        )
        kl_reward_no_entropy = -1*(
            kl_scores_no_entropy*self.kl_penalty_reward_coeff).sum(-1)
        reward_no_entropy = (
            log_prob_reward + kl_reward_no_entropy + match_reward)

        # 准备无熵版本的日志字典
        kl_dict_no_entropy = self.get_components_dictionaries_to_log(
            kl=-student_log_probs,
            log_probs=None,
            teacher_mask=teacher_mask,
            student_solution_masks=None,
        )
        kl_dict_no_entropy = {
            f'no_entropy_{k}': v for k, v in kl_dict_no_entropy.items()}

        kl_scores_no_entropy_to_log = {
            f'no_entropy_{n}': kl_scores_no_entropy[..., i]
            for i, n in enumerate(self.log_kl_names)}

        self._print_debugging_logs('logging')
        # 记录所有指标
        logged_dict = self.trainer.log_metric(
            **unprocessed_dict,
            **log_prob_scores_to_log,
            solution_log_prob_reward=log_prob_reward,
            **kl_scores_to_log,
            thought_processed_kl=processed_kl,
            thought_kl_scores=kl_scores,
            kl_reward=kl_reward,
            match_reward=match_reward,
            total_teacher_likelihood_reward=reward,
            # 无熵版本的指标
            processed_kl_no_entropy=processed_kl_no_entropy,
            kl_scores_no_entropy=kl_scores_no_entropy,
            kl_reward_no_entropy=kl_reward_no_entropy,
            total_tl_reward_no_entropy=reward_no_entropy,
            **kl_dict_no_entropy,
            **kl_scores_no_entropy_to_log,
        )
        
        # 记录对话到文件
        if self.trainer.accelerator.is_main_process:
            self._print_debugging_logs('logging chats')
            for chat in chat:
                self.trainer.log_to_file(chat)

        # 根据设置选择返回哪个奖励版本
        if include_teacher_think_entropy:
            rw_list = reward.tolist()
        else:
            rw_list = reward_no_entropy.tolist()
        
        # 根据返回选项决定返回内容
        if return_info_dict or return_raw_tensors:
            return rw_list, logged_dict, tensors_to_return_dict
        return rw_list

    @torch.no_grad()
    def __call__(
            self,
            prompts,
            completions,
            student_system_prompts,
            start_think_teacher_tags,
            end_think_teacher_tags,
            start_think_student_tags,
            end_think_student_tags,
            start_solution_tags,
            end_solution_tags,
            think_prefixes,
            think_solution_delimiters,
            questions,
            solutions,
            masked_out_think_prefix,
            use_schulman_unbiased_estimate=None,
            include_teacher_think_entropy=None,
            # 返回选项
            return_info_dict=False,
            return_raw_tensors=False,
            # 缓存选项
            cached_student_log_probs=None,
            cached_teacher_log_probs=None,
            cached_thought_tokens_kl=None,
            **kwargs,):
        """
        主要的奖励计算函数
        
        Args:
            prompts: 提示列表
            completions: 完成文本列表
            student_system_prompts: 学生系统提示列表
            各种标签参数: 用于标记思考和解决方案的开始/结束标签
            questions: 问题列表
            solutions: 解决方案列表
            masked_out_think_prefix: 掩码思考前缀列表
            use_schulman_unbiased_estimate: 是否使用Schulman无偏估计
            include_teacher_think_entropy: 是否包含教师思考熵
            return_info_dict: 是否返回信息字典
            return_raw_tensors: 是否返回原始张量
            cached_student_log_probs: 缓存的学生对数概率
            cached_teacher_log_probs: 缓存的教师对数概率
            cached_thought_tokens_kl: 缓存的思考token KL散度
            
        Returns:
            奖励列表，可选的日志字典列表和张量字典列表
        """
        if include_teacher_think_entropy is None:
            include_teacher_think_entropy = self.include_teacher_think_entropy

        # 构建完整的教师解决方案
        full_teacher_solutions = [
            p + c for p, c in zip(prompts, completions)]

        # 构建带掩码前缀的完成文本
        completions = [
            p + c for p, c in zip(masked_out_think_prefix, completions)]

        # 移除前缀得到原始提示
        prompts_no_prefix = [
            p.removesuffix(pre) for pre, p in zip(
                masked_out_think_prefix, prompts)
        ]

        self._print_debugging_logs('inside teacher reward, extracting chats')
        # 获取学生对话和相关信息
        (chats,
         match_reward,
         teacher_completion_list,
         start_end_teacher_thought_idxs_list,
         start_end_student_thought_idxs_list,
         start_end_student_solution_idxs_list) = (
            self.get_student_chats_and_relevant_num_tokens(
                completions=completions,
                student_system_prompts=student_system_prompts,
                questions=questions,
                solutions=solutions,
                start_think_teacher_tags=start_think_teacher_tags,
                end_think_teacher_tags=end_think_teacher_tags,
                start_think_student_tags=start_think_student_tags,
                end_think_student_tags=end_think_student_tags,
                start_solution_tags=start_solution_tags,
                end_solution_tags=end_solution_tags,
                think_prefixes=think_prefixes,
                think_solution_delimiters=think_solution_delimiters,
            )
        )

        # 重构教师解决方案
        rec_teacher_solutions = [
            p + c for p, c in zip(
                prompts_no_prefix, teacher_completion_list)]

        # 验证教师和学生思考token的一致性
        for i, (ft, rec_ft, mr) in enumerate(zip(
                full_teacher_solutions, rec_teacher_solutions, match_reward)):

            teacher_se = start_end_teacher_thought_idxs_list[i]
            teacher_enc = self.tokenizer(
                rec_ft,
                return_tensors='pt',
                padding=True,
                truncation=True
            )['input_ids'][0]
            teacher_enc_f = teacher_enc[teacher_se[0]:teacher_se[1]]
            
            student_se = start_end_student_thought_idxs_list[i]
            student_enc = self.tokenizer(
                chats[i],
                return_tensors='pt',
                padding=True,
                truncation=True
            )['input_ids'][0]
            student_enc_f = student_enc[student_se[0]:student_se[1]]
            
            # 检查学生和教师思考token是否一致
            if not torch.all(student_enc_f == teacher_enc_f):
                print(
                    f'Warning - student teacher tokens, match reward sc {mr}')
            
            # 验证对齐和提取的正确性
            teacher_thought_masks = self.get_mask_for_spans(
                [teacher_se],
                seq_len=teacher_enc.shape[-1],
                device=teacher_enc.device,
            )

            teacher_enc_al, teacher_mask = (
                extract_and_left_align_from_mask(
                    teacher_enc.unsqueeze(0), teacher_thought_masks))

            if not torch.all(teacher_enc_al == teacher_enc_f):
                print(f'Warning - aligned and extracted encs, mismatch'
                      f' match reward sc {mr}')

        # 处理每个样本
        num_chats = len(chats)
        out_values = []
        for i in range(num_chats):
            out = self.process_single_reward(
                chat=chats[i:i+1],
                match_reward=match_reward[i:i+1],
                # 教师完成文本
                teacher_completion=rec_teacher_solutions[i:i+1],
                start_end_teacher_thought_idxs=(
                    start_end_teacher_thought_idxs_list[i:i+1]),
                start_end_student_thought_idxs=(
                    start_end_student_thought_idxs_list[i:i+1]),
                start_end_student_solution_idxs=(
                    start_end_student_solution_idxs_list[i:i+1]),
                use_schulman_unbiased_estimate=(
                    use_schulman_unbiased_estimate),
                include_teacher_think_entropy=include_teacher_think_entropy,
                return_info_dict=return_info_dict,
                return_raw_tensors=return_raw_tensors,
                # 缓存参数
                cached_student_log_probs=(
                    cached_student_log_probs[i]
                    if cached_student_log_probs is not None else None),
                cached_teacher_log_probs=(
                    cached_teacher_log_probs[i]
                    if cached_teacher_log_probs is not None else None),
                cached_thought_tokens_kl=(
                    cached_thought_tokens_kl[i]
                    if cached_thought_tokens_kl is not None else None),
            )
            out_values.append(out)
            # 清理内存
            gc.collect()
            torch.cuda.empty_cache()

        # 整理返回结果
        rw_list = []
        if return_info_dict or return_raw_tensors:
            logged_dicts, tensors_to_return_dicts = [], []
            for out in out_values:
                rw, logged_dict, tensors_to_return_dict = out
                rw_list += rw
                logged_dicts.append(logged_dict)
                tensors_to_return_dicts.append(tensors_to_return_dict)
            return rw_list, logged_dicts, tensors_to_return_dicts
        
        # 只返回奖励列表
        for rw in out_values:
            rw_list += rw
        return rw_list

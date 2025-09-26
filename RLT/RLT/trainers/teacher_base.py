import os
import abc
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Sequence
from transformers import PreTrainedTokenizer
import re
import random
from itertools import islice


def is_subsequence(sub, seq):
    """
    检查子序列是否存在于序列中
    
    Args:
        sub: 子序列
        seq: 主序列
        
    Returns:
        bool: 如果子序列存在于主序列中则返回True
    """
    sub_len = len(sub)
    return any(sub == list(islice(seq, i, i + sub_len))
               for i in range(len(seq) - sub_len + 1))


def find_valid_subsequence(sub, seq):
    """
    在序列中查找有效的子序列，如果完整子序列不存在，则尝试查找部分匹配
    
    Args:
        sub: 要查找的子序列
        seq: 主序列
        
    Returns:
        找到的有效子序列，如果没有找到则返回None
    """
    # 首先检查完整的子序列
    if is_subsequence(sub, seq):
        return sub
    # 如果子序列长度大于1，尝试去掉首尾元素
    if len(sub) > 1:
        # 尝试去掉第一个元素
        if is_subsequence(sub[1:], seq):
            return sub[1:]
        # 尝试去掉最后一个元素
        if is_subsequence(sub[:-1], seq):
            return sub[:-1]
    # 如果子序列长度大于2，尝试去掉首尾元素
    if len(sub) > 2 and is_subsequence(sub[1:-1], seq):
        return sub[1:-1]
    return None


def is_tensor(t):
    """
    检查对象是否为PyTorch张量
    
    Args:
        t: 要检查的对象
        
    Returns:
        bool: 如果是张量则返回True
    """
    if isinstance(t, torch.Tensor):
        return True
    return False


def log_tensor_info(tensor):
    """
    打印张量的基本信息，包括形状、最大值、最小值、均值和是否包含NaN
    
    Args:
        tensor: 要分析的张量
    """
    print(f"Shape: {tensor.shape}")
    print(f"Max: {tensor.max().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Mean: {tensor.float().mean().item()}")
    print(f"Nan: {torch.isnan(tensor).any().item()}")


def find_first_last_one_idxs(tensor, negative_indices=False):
    """
    在二维张量中找到每行第一个和最后一个非零元素的索引
    
    Args:
        tensor: 二维张量 (batch_size, sequence_length)
        negative_indices: 是否返回负索引（从末尾开始计算）
        
    Returns:
        tuple: (first_indices, last_indices) 每行第一个和最后一个非零元素的索引
    """
    b, m = tensor.shape
    tensor = tensor.to(torch.long)
    # 找到每行第一个非零元素的索引
    first_indices = torch.where(tensor.any(dim=1), tensor.argmax(dim=1), -1)
    # 找到每行最后一个非零元素的索引
    last_indices = torch.where(
        tensor.any(dim=1), m - 1 - tensor.flip(dims=[1]).argmax(dim=1), -1)

    # 如果需要负索引，转换为从末尾开始的索引
    if negative_indices:
        first_indices = torch.where(
            first_indices != -1, first_indices - m, -1)
        last_indices = torch.where(
            last_indices != -1, last_indices - m, -1)

    return first_indices, last_indices


def extract_and_left_align_from_mask(matrix, mask):
    """
    根据掩码从矩阵中提取元素并左对齐
    
    Args:
        matrix: 输入矩阵
        mask: 布尔掩码，指示要提取的元素
        
    Returns:
        tuple: (data_out, mask_out) 提取并左对齐的数据和对应的掩码
    """
    bsz = matrix.size(0)
    rows = []
    max_len = 0
    
    # 为每个批次提取掩码为True的元素
    for i in range(bsz):
        row = matrix[i][mask[i].bool()]
        rows.append(row)
        max_len = max(max_len, row.size(0))

    padded_rows, padded_masks = [], []
    # 对提取的行进行填充以达到相同长度
    for row in rows:
        pad_len = max_len - row.size(0)
        # 填充数据
        padded_rows.append(F.pad(row, (0, pad_len)))
        # 创建对应的掩码
        row_mask = torch.ones_like(row, dtype=torch.bool)
        padded_masks.append(F.pad(row_mask, (0, pad_len)))

    # 堆叠所有行
    data_out = torch.stack(padded_rows, dim=0)
    mask_out = torch.stack(padded_masks, dim=0)
    return data_out, mask_out


def find_sublist_start_end(lst, sublst, from_end=False, reverse_search=False):
    """
    在列表中查找子列表的开始和结束位置
    
    Args:
        lst: 主列表
        sublst: 要查找的子列表
        from_end: 是否返回从末尾开始的索引
        reverse_search: 是否从后向前搜索
        
    Returns:
        tuple: (start_index, end_index) 子列表的开始和结束位置，如果未找到则返回None
    """
    if sublst is not None:
        sub_len = len(sublst)
        # 根据搜索方向确定索引范围
        indices_range = (
            range(len(lst) - sub_len, -1, -1) if reverse_search else
            range(len(lst) - sub_len + 1)
        )

        # 在指定范围内搜索子列表
        for i in indices_range:
            if lst[i:i + sub_len] == sublst:
                if from_end:
                    # 返回从末尾开始的索引
                    return i - len(lst), i + sub_len - len(lst)
                else:
                    # 返回正常索引
                    return i, i + sub_len
    return None


def find_indices_between_tags(content, start_tag, end_tag):
    """
    在内容中查找开始和结束标签之间的索引位置
    
    Args:
        content: 要搜索的内容字符串
        start_tag: 开始标签
        end_tag: 结束标签
        
    Returns:
        tuple: (start_content, end_content, failures) 开始位置、结束位置和失败次数
    """
    failures = 1
    # 查找开始标签的位置
    start_content = content.find(start_tag)
    if start_content == -1:
        failures += 1
        start_content = 0
    # 查找结束标签的位置
    end_content = content.find(end_tag, start_content + len(start_tag))
    if end_content == -1:
        failures += 1
        end_content = -1
    return start_content, end_content, failures


def replace_text_between_tags(content, content2, start_tag, end_tag):
    """
    用第二个内容中标签之间的文本替换第一个内容中标签之间的文本
    
    Args:
        content: 要被替换的原始内容
        content2: 提供替换文本的内容
        start_tag: 开始标签
        end_tag: 结束标签
        
    Returns:
        tuple: (new_content, replaced_range_c2, replaced_range_new) 
               新内容、在content2中的替换范围、在新内容中的替换范围
    """
    # 在第一个内容中查找标签位置
    start_content = content.find(start_tag)
    if start_content == -1:
        raise NotImplementedError
    end_content = content.find(end_tag, start_content + len(start_tag))
    if end_content == -1:
        raise NotImplementedError

    # 在第二个内容中查找标签位置
    start2 = content2.find(start_tag)
    if start2 == -1:
        raise NotImplementedError
    end2 = content2.find(end_tag, start2 + len(start_tag))
    if end2 == -1:
        raise NotImplementedError

    # 提取第二个内容中标签之间的文本
    sub2 = content2[start2 + len(start_tag):end2]

    # 构建新内容
    new_content = (content[:start_content + len(start_tag)]
                   + sub2
                   + content[end_content:])

    # 计算替换范围
    replaced_start_new = start_content + len(start_tag)
    replaced_end_new = replaced_start_new + len(sub2)
    replaced_start_c2 = start2 + len(start_tag)
    replaced_end_c2 = replaced_start_c2 + len(sub2)

    return (new_content,
            (replaced_start_c2, replaced_end_c2),
            (replaced_start_new, replaced_end_new))


class TeacherTrainer(abc.ABC):
    """
    教师训练器的抽象基类，提供学生-教师模型训练的基础功能
    """
    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        reward_functions=None,
        output_dir=None,
        logging_prob=0.0
    ):
        """
        初始化教师训练器
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            tokenizer: 分词器
            reward_functions: 奖励函数列表
            output_dir: 输出目录
            logging_prob: 日志记录概率
        """
        # 如果没有提供奖励函数，使用默认的奖励函数
        if reward_functions is None:
            reward_functions = self.reward_funcs
        
        # 为每个奖励函数建立与训练器的链接
        for rw in reward_functions:
            if isinstance(rw, TeacherReward):
                rw.link_with_trainer(
                    trainer=self,
                    student_model=student_model,
                    teacher_model=teacher_model,
                    tokenizer=tokenizer,
                )
        
        # 初始化日志相关属性
        self.logging_dir = output_dir
        self.logging_prob = logging_prob
        self.last_logged_iter = -1
        self.do_log_to_file = False
        
        # 如果设置了日志概率和输出目录，启用文件日志
        if logging_prob > 0.0 and output_dir is not None:
            self.do_log_to_file = True
            self.logging_dir = output_dir + '/teacher_chats'
            os.makedirs(self.logging_dir, exist_ok=True)
            self.logging_file = self.logging_dir + '/log.txt'

    def log_to_file(self, *args, **kwargs):
        """
        将日志信息写入文件
        
        Args:
            *args: 要记录的字符串参数
            **kwargs: 要记录的命名参数，键为文件名，值为内容
        """
        if self.do_log_to_file:
            # 如果是新的迭代步骤，记录步骤信息
            if not (self.state.global_step == self.last_logged_iter):
                with open(f"{self.logging_file}", "a") as f:
                    f.write("\n\n============================\n" +
                            f"Global step: {self.state.global_step}")
                self.last_logged_iter = self.state.global_step
            
            # 根据概率决定是否记录日志
            if random.random() < self.logging_prob:
                # 记录位置参数
                for log_value in args:
                    assert isinstance(log_value, str)
                    with open(f"{self.logging_file}", "a") as f:
                        f.write("\n\n==============\n" + log_value)
                
                # 记录命名参数到单独的文件
                for log_name, log_value in kwargs.items():
                    assert isinstance(log_value, str)
                    with open(f"{self.logging_dir}/{log_name}.txt", "a") as f:
                        f.write("\n\n==============\n" + log_value)

    def log_metric(self, **kwargs):
        """
        记录指标数据
        
        Args:
            **kwargs: 要记录的指标，键为指标名，值为指标值
            
        Returns:
            dict: 处理后的指标字典
        """
        logged_dict = {}
        for log_name, log_value in kwargs.items():
            # 处理张量类型的指标
            if is_tensor(log_value):
                log_value = log_value.mean().item()
            # 处理列表或元组类型的指标
            elif isinstance(log_value, (list, tuple)):
                log_value = np.mean(log_value)
            else:
                log_value = float(log_value)
            
            logged_dict[log_name] = log_value
            # 在主进程中记录指标
            if self.accelerator.is_main_process:
                self._metrics[log_name].append(log_value)
        return logged_dict

    def obtain_vllm_completions(
        self,
        inputs,
    ):
        """
        使用VLLM获取补全结果（抽象方法，需要子类实现）
        
        Args:
            inputs: 输入数据
            
        Raises:
            NotImplementedError: 需要在子类中实现
        """
        raise NotImplementedError


class TeacherReward(abc.ABC):
    """
    教师奖励函数的抽象基类，用于计算学生模型的奖励信号
    """
    def link_with_trainer(
            self, trainer, student_model, teacher_model, tokenizer,):
        """
        将奖励函数与训练器建立链接
        
        Args:
            trainer: 训练器实例
            student_model: 学生模型
            teacher_model: 教师模型
            tokenizer: 分词器
        """
        self.__name__ = self.__class__.__name__
        self.trainer: TeacherTrainer = trainer
        self.student_model: torch.Module = student_model
        self.teacher_model: torch.Module = teacher_model
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def _make_normalize_fn(
            self, normalize_fn, temp=1, clip_min=None, clip_max=None):
        """
        创建归一化函数
        
        Args:
            normalize_fn: 归一化方法名称或函数
            temp: 温度参数
            clip_min: 最小裁剪值
            clip_max: 最大裁剪值
            
        Returns:
            callable: 归一化函数
        """
        # 如果是字符串，转换为小写
        if isinstance(normalize_fn, str):
            normalize_fn = normalize_fn.lower()
        # 如果已经是函数，直接返回
        elif isinstance(normalize_fn, Callable):
            return normalize_fn

        def apply_clipping(x):
            """应用裁剪操作"""
            if clip_min is not None:
                x = torch.clamp(x, min=clip_min)
            if clip_max is not None:
                x = torch.clamp(x, max=clip_max)
            return x

        # 根据归一化方法创建相应的函数
        if normalize_fn is None or normalize_fn == 'none':
            def f(x):
                return apply_clipping(x / temp)
        elif normalize_fn == 'exp':
            def f(x):
                return apply_clipping(torch.exp(x / temp))
        elif normalize_fn == 'exp_norm':
            def f(x):
                return apply_clipping(1 - torch.exp(-x / temp))
        else:
            raise NotImplementedError
        return f

    def _make_reduction_fn(self, reduction_fn, function_log_name=None):
        """
        创建降维函数，用于将序列数据降维为标量
        
        Args:
            reduction_fn: 降维方法名称、函数或方法列表
            function_log_name: 用于日志的函数名称
            
        Returns:
            callable 或 tuple: 降维函数，如果提供了function_log_name则返回(函数, 日志名称列表)
        """
        # 如果已经是函数，直接返回
        if isinstance(reduction_fn, Callable):
            if function_log_name is not None:
                log_names_to_store = [function_log_name + '_custom']
                return reduction_fn, log_names_to_store
            return reduction_fn

        def _flatten(seq):
            """递归展平序列"""
            for i in seq:
                if isinstance(i, Sequence) and not isinstance(i, str):
                    yield from _flatten(i)
                else:
                    yield i

        # 处理输入参数，转换为操作列表
        if isinstance(reduction_fn, str):
            ops = [reduction_fn.lower()]
        elif isinstance(reduction_fn, Sequence):
            ops = [op.lower() for op in _flatten(reduction_fn)]
        else:
            raise NotImplementedError

        def f(x, mask):
            """
            执行降维操作
            
            Args:
                x: 输入张量
                mask: 掩码张量
                
            Returns:
                torch.Tensor: 降维后的结果
            """
            out = []
            for op in ops:
                try:
                    if op == 'mean':
                        # 计算掩码区域的平均值
                        o = torch.sum(x * mask, dim=-1) / \
                            torch.sum(mask, dim=-1)
                    elif op == 'sum':
                        # 计算掩码区域的和
                        o = torch.sum(x * mask, dim=-1)
                    elif op == 'min':
                        # 计算掩码区域的最小值
                        tmp = x.masked_fill(mask == 0,
                                            torch.finfo(x.dtype).max)
                        o = torch.min(tmp, dim=-1).values
                    elif op == 'max':
                        # 计算掩码区域的最大值
                        tmp = x.masked_fill(mask == 0,
                                            torch.finfo(x.dtype).min)
                        o = torch.max(tmp, dim=-1).values
                    elif op == 'median':
                        # 计算掩码区域的中位数
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanmedian(tmp, dim=-1).values
                    elif op == 'first_quartile':
                        # 计算掩码区域的第一四分位数
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanquantile(tmp.float(),
                                              0.25, dim=-1)
                    elif op == 'last_quartile':
                        # 计算掩码区域的第三四分位数
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanquantile(tmp.float(),
                                              0.75, dim=-1)
                    else:
                        raise NotImplementedError
                except Exception:
                    # 如果计算失败，返回NaN值
                    print(f'Invalid completion when reducing {x}...')
                    o = torch.full(
                        x.shape[:-1], float('nan'),
                        dtype=x.dtype, device=x.device)
                out.append(o)
            return torch.stack(out, dim=-1)

        # 如果提供了日志名称，返回函数和日志名称列表
        if function_log_name is not None:
            log_names_to_store = [
                function_log_name + '_' + op for op in ops]
            return f, log_names_to_store
        return f

    def _make_elementwise_normalize_fn(self, normalize_fn, temp=1):
        """
        创建逐元素归一化函数
        
        Args:
            normalize_fn: 归一化方法名称、函数或方法列表
            temp: 温度参数
            
        Returns:
            callable: 逐元素归一化函数
        """
        # 如果没有指定归一化方法，返回简单的温度缩放
        if normalize_fn is None:
            return lambda x: x / temp
        # 如果已经是函数，直接返回
        if callable(normalize_fn):
            return normalize_fn

        import torch

        def build_transform(fn, t):
            """
            根据方法名构建变换函数
            
            Args:
                fn: 变换方法名称
                t: 温度参数
                
            Returns:
                callable: 变换函数
            """
            if fn is None or fn == 'none':
                return lambda x: x / t
            elif fn == 'exp':
                return lambda x: torch.exp(x / t)
            elif fn == 'exp_norm':
                return lambda x: 1 - torch.exp(-x / t)
            elif fn == 'sym_log':
                # 对称对数变换
                return lambda x: torch.where(
                    x > 0,
                    torch.log(x / t),
                    torch.where(x < 0, -torch.log(-x / t),
                                torch.zeros_like(x))
                )
            else:
                raise NotImplementedError

        # 如果是列表，为每个通道创建不同的变换
        if isinstance(normalize_fn, list):
            trans_list = []
            for fn in normalize_fn:
                if isinstance(fn, str):
                    fn = fn.lower()
                trans_list.append(build_transform(fn, temp))

            def f(x):
                """对多通道数据应用不同的变换"""
                if x.shape[-1] != len(trans_list):
                    raise ValueError()
                channels = [trans_list[i](x[..., i])
                            for i in range(x.shape[-1])]
                return torch.stack(channels, dim=-1)
            return f
        elif isinstance(normalize_fn, str):
            return build_transform(normalize_fn.lower(), temp)
        else:
            raise TypeError()

    @abc.abstractmethod
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
        计算奖励的抽象方法（需要子类实现）
        
        Args:
            prompts: 提示文本列表
            completions: 补全文本列表
            student_system_prompts: 学生系统提示列表
            start_think_teacher_tags: 教师思考开始标签列表
            end_think_teacher_tags: 教师思考结束标签列表
            start_think_student_tags: 学生思考开始标签列表
            end_think_student_tags: 学生思考结束标签列表
            start_solution_tags: 解决方案开始标签列表
            end_solution_tags: 解决方案结束标签列表
            think_prefixes: 思考前缀列表
            think_solution_delimiters: 思考-解决方案分隔符列表
            questions: 问题列表
            solutions: 解决方案列表
            **kwargs: 其他关键字参数
            
        Raises:
            NotImplementedError: 需要在子类中实现
        """
        raise NotImplementedError

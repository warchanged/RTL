import abc
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import copy
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import timedelta
from dataclasses import dataclass, field
from unittest.mock import patch
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict, Counter

from accelerate.utils import broadcast_object_list, gather, gather_object


from torch import distributed as dist, nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import accelerate
from transformers.trainer_utils import TrainOutput
from torch.utils.data import Subset


from datasets import Dataset, IterableDataset
from .utils_trl_15 import prepare_deepspeed
from .teacher_base import TeacherReward, TeacherTrainer


# 从数据集中获取得分最高的样本
def get_top_hf_dataset(dataset, score_column, number_to_keep, reverse=False):
    """
    从HuggingFace数据集中根据得分列选择得分最高的样本
    
    Args:
        dataset: HuggingFace数据集
        score_column: 得分列名
        number_to_keep: 要保留的样本数量
        reverse: 是否反向排序（保留得分最低的样本）
    
    Returns:
        包含得分最高样本的数据集子集
    """
    scores = dataset[score_column]

    # 根据得分对索引进行排序
    sorted_indices = sorted(range(len(dataset)),
                            key=lambda i: np.max([s.item() for s in scores[i]])
                            if hasattr(scores[i], "item") else scores[i],
                            reverse=True)
    if reverse:
        # 如果reverse为True，选择得分最低的样本
        top_indices = sorted_indices[-number_to_keep:]
    else:
        # 选择得分最高的样本
        top_indices = sorted_indices[:number_to_keep]

    # 对索引进行排序以保持原始顺序
    top_indices.sort()

    return dataset.select(top_indices)


def get_top_dataset(dataset, score_column, number_to_keep):
    """
    从普通数据集中根据得分列选择得分最高的样本
    
    Args:
        dataset: 数据集
        score_column: 得分列名
        number_to_keep: 要保留的样本数量
    
    Returns:
        包含得分最高样本的数据集子集
    """
    scores = []
    # 提取所有样本的得分
    for i in range(len(dataset)):
        item = dataset[i]
        score = item[score_column]
        # 如果得分是tensor，转换为标量
        if torch.is_tensor(score):
            score = score.item()
        scores.append(score)

    # 根据得分对索引进行排序（降序）
    sorted_indices = sorted(range(len(scores)),
                            key=lambda i: scores[i],
                            reverse=True)
    # 选择得分最高的样本索引
    top_indices = sorted_indices[:number_to_keep]

    # 对索引进行排序以保持原始顺序
    top_indices = sorted(top_indices)

    return Subset(dataset, top_indices)


def instantiate_from_target(cfg, **kwargs):
    """
    从配置中实例化对象，支持target和_target_两种配置格式
    
    Args:
        cfg: 配置字典或DictConfig
        **kwargs: 额外的关键字参数
    
    Returns:
        实例化的对象
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # 兼容target和_target_两种配置格式
    if 'target' in cfg and '_target_' not in cfg:
        cfg['_target_'] = cfg.pop('target')

    if '_target_' not in cfg:
        raise KeyError

    return hydra.utils.instantiate(cfg, **kwargs)


def gather_lists(local_list, accelerator, log_name=None):
    """
    在多进程环境中收集所有进程的列表数据
    
    Args:
        local_list: 本地进程的列表数据
        accelerator: Accelerate加速器对象
        log_name: 日志名称（可选）
    
    Returns:
        合并后的完整列表
    """
    # 收集所有进程的列表
    gathered_lists = accelerator.gather_object(local_list)

    # 将所有列表合并为一个
    combined = []
    for lst in gathered_lists:
        combined.extend(lst)
    return combined


def value_frequencies(values):
    """
    计算值的频率分布
    
    Args:
        values: 值的列表
    
    Returns:
        包含频率百分比的字典
    """
    total = len(values)
    counts = Counter(values)
    return {f'freq_{v}': (c / total) * 100 for v, c in counts.items()}


def get_mean_std_max_min_dict(array, prefix):
    """
    计算数组的统计信息（均值、标准差、最小值、最大值）
    
    Args:
        array: 数值数组
        prefix: 结果字典键的前缀
    
    Returns:
        包含统计信息的字典
    """
    res = {}
    res[prefix + '/mean'] = np.mean(array)
    res[prefix + '/std'] = np.std(array)
    res[prefix + '/min'] = np.amin(array)
    res[prefix + '/max'] = np.amax(array)
    return res


def is_unique_sequential(lst, orig_idx=None):
    """
    检查列表是否包含唯一的连续值
    
    Args:
        lst: 要检查的列表
        orig_idx: 原始索引列表（可选）
    
    Returns:
        布尔值，表示列表是否包含唯一的连续值
    """
    idxs_for_seq = range(len(lst))
    if orig_idx is not None:
        idxs_for_seq = [orig_idx[i] for i in idxs_for_seq]
    return set(lst) == set(idxs_for_seq)


def list_of_dicts_to_dict_of_lists(lst):
    """
    将字典列表转换为列表字典
    
    Args:
        lst: 字典列表
    
    Returns:
        列表字典
    """
    result = defaultdict(list)
    for d in lst:
        for key, value in d.items():
            result[key].append(value)
    return dict(result)


def merge_samples(old_dataset: Dataset, new_samples: List[dict]):
    """
    合并旧数据集和新样本，如果索引相同则合并completions
    
    Args:
        old_dataset: 旧数据集
        new_samples: 新样本列表
    
    Returns:
        合并后的样本列表
    """
    # 将旧数据集转换为字典，以__index为键
    dataset_dict = {sample["__index"]: sample for sample in old_dataset}
    
    # 合并新样本
    for new_sample in new_samples:
        new_index = new_sample["__index"]
        if new_index in dataset_dict:
            # 如果索引已存在，合并completions
            dataset_dict[new_index]["completions"] += new_sample["completions"]
        else:
            # 如果索引不存在，添加新样本
            dataset_dict[new_index] = new_sample
    return list(dataset_dict.values())


def save_pickle(fname, directory, **kwargs):
    """
    将数据保存为pickle文件
    
    Args:
        fname: 文件名（不包含扩展名）
        directory: 保存目录
        **kwargs: 要保存的数据
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{fname}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(kwargs, f)


def load_pickle(fname, directory, *args):
    """
    从pickle文件加载数据
    
    Args:
        fname: 文件名（不包含扩展名）
        directory: 文件目录
        *args: 要加载的数据键名
    
    Returns:
        加载的数据元组
    """
    file_path = os.path.join(directory, f"{fname}.pickle")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return tuple(data[arg] for arg in args if arg in data)


@dataclass
class DataScorerArgs(TrainingArguments):
    """
    数据评分器的参数配置类，继承自TrainingArguments
    """
    # 排序数据保存目录
    ranked_data_dir: Optional[str] = None
    # 生成的得分文件名
    generated_scores_fname: str = 'reward_scores'
    # 从哪里检索得分
    retrieve_scores_from: Optional[str] = None
    # PEFT配置
    peft_config: Optional[Dict[str, Any]] = field(default=None)
    # 从哪个数据分割进行评分
    score_from_split: str = 'train'
    # 目标提示列名
    target_prompt_column: str = field(default="prompt")
    # 目标完成列名
    target_completions_column: str = field(default="completions")
    # 更新列名
    update_column: Optional[str] = field(default="completion")
    # 奖励列名
    rewards_column: Optional[str] = field(default="completions_rewards")

    # 是否存储原始得分
    store_raw_scores: bool = True
    # 随机种子
    seed: int = 42
    # 生成温度
    temperature: float = field(default=0.9,)
    # 是否去偏置对数概率
    unbias_log_probabilities: bool = field(default=True,)
    # 是否激活调试日志
    activate_debugging_logs: bool = field(default=False,)

    # 限制评分样本数量
    limit_scoring_samples: Optional[int] = None
    # 格式化条目索引
    formatted_entry_idx: Optional[int] = None


class DataTeacherRewardScorer(Trainer, TeacherTrainer):
    """
    数据教师奖励评分器类，用于对数据集中的完成进行评分和排序
    """
    def __init__(
            self,
            model: Union[str, PreTrainedModel],  # 教师模型
            args: DataScorerArgs,  # 评分器参数
            reward_funcs: List[TeacherReward],  # 奖励函数列表
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,  # 训练数据集
            eval_dataset: Optional[Union[Dataset, IterableDataset,
                                         dict[str, Union[Dataset, IterableDataset]]]] = None,  # 评估数据集
            processing_class: Optional[PreTrainedTokenizerBase] = None,  # 处理类（tokenizer）
            reward_funcs_names: Optional[List[str]] = None,  # 奖励函数名称列表
            student_model: Union[str, PreTrainedModel] = None,  # 学生模型
            teacher_model_init_kwargs=None,  # 教师模型初始化参数
            student_model_init_kwargs=None,  # 学生模型初始化参数
            logging_prob=0.0,  # 日志记录概率
            offload_models=False,  # 是否卸载模型到CPU
            **kwargs):

        # 初始化模型初始化参数
        if teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        if student_model_init_kwargs is None:
            student_model_init_kwargs = {}

        # 初始化基本属性
        self._metrics = defaultdict(list)
        self.args = args
        self.seed = self.args.seed
        self.reward_funcs = reward_funcs
        self.formatted_entry_idx = args.formatted_entry_idx

        # 处理格式化条目索引
        if self.formatted_entry_idx is not None:
            assert int(self.formatted_entry_idx) == self.formatted_entry_idx
            self.formatted_entry_idx = int(self.formatted_entry_idx)
            self.has_reserved_completion_index = True
        else:
            self.has_reserved_completion_index = False

        # 设置概率相关参数
        self.unbias_log_probabilities = args.unbias_log_probabilities
        self.gen_temperature = args.temperature

        if self.unbias_log_probabilities:
            assert self.gen_temperature > 0.0

        # 验证奖励函数
        assert len(self.reward_funcs) > 0
        assert len(self.reward_funcs) == 1

        # 设置奖励函数名称
        self.reward_funcs_names = (
            reward_funcs_names if reward_funcs_names
            else [reward_func.__name__ for reward_func in self.reward_funcs]
        )
        assert len(self.reward_funcs) == len(self.reward_funcs_names)

        # 设置配置参数
        self.peft_config = args.peft_config
        self.score_from_split = args.score_from_split
        
        # 根据分割类型选择数据集
        if self.score_from_split == 'train':
            self.dataset = train_dataset
        elif self.score_from_split in ['val', 'eval', 'test']:
            self.dataset = eval_dataset
        else:
            raise NotImplementedError

        # 设置列名
        self.target_prompt_column = args.target_prompt_column
        self.target_completions_column = args.target_completions_column
        self.update_column = args.update_column
        self.rewards_column = args.rewards_column
        self.store_raw_scores = args.store_raw_scores
        
        # 设置随机种子并创建加速器
        self.set_seed(seed=self.seed)
        self.create_accelerator_and_postprocess()

        self.processing_class = processing_class

        # 初始化教师模型
        teacher_model = model
        if isinstance(teacher_model, str):
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model, **teacher_model_init_kwargs)
            if self.is_deepspeed_enabled:
                # 如果启用DeepSpeed，准备模型
                self.teacher_model = prepare_deepspeed(
                    self.teacher_model,
                    self.accelerator,
                    offload_to_cpu=offload_models)
            else:
                # 使用Accelerate准备模型
                self.teacher_model = self.accelerator.prepare_model(
                    self.teacher_model, evaluation_mode=True)

                if offload_models:
                    # 如果需要，将模型卸载到CPU
                    self.teacher_model = accelerate.cpu_offload(
                        model=self.teacher_model)
            
            # 如果没有提供处理类，自动创建tokenizer
            if processing_class is None:
                self.processing_class = AutoTokenizer.from_pretrained(
                    model, padding_side="left")
        else:
            # 如果直接提供模型对象（暂未实现）
            raise NotImplementedError
            self.teacher_model = teacher_model

        # 确保处理类存在并设置padding方向
        assert self.processing_class is not None
        self.processing_class.padding_side = 'left'

        # 初始化学生模型
        if student_model is None:
            # 如果没有提供学生模型，使用参考模型
            self.student_model = self.ref_model
        if isinstance(student_model, str):
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model, **student_model_init_kwargs)
            if self.is_deepspeed_enabled:
                self.student_model = prepare_deepspeed(
                    self.student_model,
                    self.accelerator,
                    offload_to_cpu=offload_models)
            else:
                self.student_model = self.accelerator.prepare_model(
                    self.student_model, evaluation_mode=True)

                if offload_models:
                    self.student_model = accelerate.cpu_offload(
                        model=self.student_model)
        else:
            # 如果直接提供模型对象（暂未实现）
            raise NotImplementedError
            self.student_model = student_model

        # 初始化TeacherTrainer
        TeacherTrainer.__init__(
            self,
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            tokenizer=self.processing_class,
            reward_functions=self.reward_funcs,
            output_dir=self.args.output_dir,
            logging_prob=logging_prob,
        )

        # 设置输出目录和文件路径
        self.ranked_data_dir = args.ranked_data_dir
        os.makedirs(self.ranked_data_dir, exist_ok=True)
        self.generated_scores_fname = args.generated_scores_fname
        self.full_generated_file_path = os.path.join(
            self.ranked_data_dir, f"{self.generated_scores_fname}.pickle")

        # 设置奖励函数
        self.reward_fn = self.reward_funcs[0]

        # 初始化缓存的奖励张量
        self.cached_reward_raw_tensors = None

        # 设置检索得分的路径
        self.retrieve_scores_from = self.args.retrieve_scores_from
        if self.retrieve_scores_from is None:
            self.retrieve_scores_from = self.ranked_data_dir

        self.out_file_path = os.path.join(self.ranked_data_dir, "ranked.json")

        # 检查是否存在缓存的得分文件
        check_picke_file = os.path.join(
            self.retrieve_scores_from, f"{self.generated_scores_fname}.pickle")
        if os.path.exists(check_picke_file):
            # 加载缓存的奖励张量
            self.cached_reward_raw_tensors = load_pickle(
                self.generated_scores_fname,
                self.retrieve_scores_from,
                'cached_reward_raw_tensors',
            )
            if isinstance(self.cached_reward_raw_tensors, tuple) and len(
                    self.cached_reward_raw_tensors) == 1:
                self.cached_reward_raw_tensors = (
                    self.cached_reward_raw_tensors[0])
            print('Loaded cached reward tensor!')

        # 设置样本限制
        self.limit_scoring_samples = self.args.limit_scoring_samples
        self.limited_scoring_run = self.limit_scoring_samples is not None

    def _print_for_process(self, to_print: str, only_main=False):
        """
        为特定进程打印信息
        
        Args:
            to_print: 要打印的字符串
            only_main: 是否只在主进程打印
        """
        if only_main:
            if self.accelerator.is_main_process:
                print(f'Main process: {to_print}')
        else:
            print(f'Process {self.accelerator.process_index}: {to_print}')

    def _print_debugging_logs(self, to_print: str):
        """
        打印调试日志
        
        Args:
            to_print: 要打印的字符串
        """
        if self.args.activate_debugging_logs:
            print(f'Process {self.accelerator.process_index}: {to_print}')

    def set_seed(self, seed: int) -> None:
        """
        设置随机种子以确保结果可重现
        
        Args:
            seed: 随机种子值
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def rank_prompt(self, sample_idx, log_completion=False):
        """
        对单个样本的完成进行排序
        
        Args:
            sample_idx: 样本索引
            log_completion: 是否记录完成信息
        
        Returns:
            奖励列表、奖励信息列表、缓存张量列表
        """
        # 获取样本数据
        sample = self.dataset[sample_idx]
        prompt = sample[self.args.target_prompt_column]
        completions = sample[self.args.target_completions_column]
        num_completions = len(completions)

        # 获取样本真实索引
        sample_true_idx = sample.get('__index', sample_idx)
        reward_kwargs = {}

        # 提取除了提示和完成之外的其他字段作为奖励函数参数
        for k in sample.keys():
            if k not in [self.args.target_prompt_column, 'prompts',
                         self.args.target_completions_column, 'completions',
                         'completion', 'prompt']:
                reward_kwargs[k] = [sample[k]]
        
        # 处理缓存的张量
        if self.cached_reward_raw_tensors is not None:
            cached_tensors_for_completions = self.cached_reward_raw_tensors[
                sample_idx]
            assert cached_tensors_for_completions[0][
                '__index'] == sample_true_idx
            assert len(cached_tensors_for_completions) == len(completions)
        else:
            cached_tensors_for_completions = [
                {} for _ in range(num_completions)]
            assert True
        
        # 初始化结果列表
        rws = []
        rw_infos = []
        
        # 确保completions是列表格式
        if isinstance(completions, str):
            completions = [completions]
        
        # 对每个完成计算奖励
        for i, completion in enumerate(completions):
            # 准备缓存张量参数
            cached_tensors = {
                f'cached_{k}': [v] for k, v in
                cached_tensors_for_completions[i].items()}
            reward_kwargs.update(cached_tensors)

            # 计算奖励
            rw, rw_info, raw_tensors = self.reward_fn(
                prompts=[prompt],
                completions=[completion],
                **reward_kwargs,
                return_info_dict=True,
                return_raw_tensors=self.store_raw_scores,
            )

            # 更新缓存张量
            cached_tensors_for_completions[i].update(raw_tensors[0])
            cached_tensors_for_completions[i]['rw'] = rw
            cached_tensors_for_completions[i]['__index'] = sample_true_idx
            rws.append(rw[0])
            rw_infos.extend(rw_info)
        return rws, rw_infos, cached_tensors_for_completions

    def train(self, resume_from_checkpoint=None):
        """
        执行训练过程，实际上是对数据进行评分和排序
        
        Args:
            resume_from_checkpoint: 从检查点恢复（未使用）
        
        Returns:
            训练输出结果
        """
        # 获取数据集的提示和完成
        all_data_prompts = self.dataset[self.args.target_prompt_column]
        data_completions = self.dataset[self.args.target_completions_column]

        data_prompts = all_data_prompts

        # 确定要处理的样本数量
        all_num_samples = len(data_prompts)
        if self.limited_scoring_run:
            all_num_samples = min(all_num_samples, self.limit_scoring_samples)

        all_sample_indices = list(range(all_num_samples))

        # 计算多进程分配
        num_processes = self.accelerator.num_processes

        process_batch_size = (
            all_num_samples + num_processes - 1)//num_processes

        # 随机打乱样本索引
        perm = np.random.permutation(len(all_sample_indices))
        all_sample_indices_shuffled = [all_sample_indices[i] for i in perm]
        
        # 将样本分配给不同进程
        all_sample_idxs_split = [
            all_sample_indices_shuffled[
                process_idx*process_batch_size:
                (process_idx + 1)*process_batch_size]
            for process_idx in range(num_processes)
        ]
        inv_perm = np.argsort(perm)

        # 广播分配结果到所有进程
        all_sample_idxs_split = broadcast_object_list(
            object_list=all_sample_idxs_split, from_process=0)
        inv_perm = broadcast_object_list(
            object_list=inv_perm, from_process=0)

        first_sample_idxs = all_sample_idxs_split[0]

        # 获取当前进程要处理的样本
        sample_idxs = all_sample_idxs_split[self.accelerator.process_index]

        # 处理样本数量不均匀的情况
        if len(first_sample_idxs) > len(sample_idxs):
            self.padded_idxs = len(first_sample_idxs) - len(sample_idxs)
            sample_idxs.extend(first_sample_idxs[-self.padded_idxs:])
        else:
            self.padded_idxs = 0

        if len(sample_idxs) > 0:
            self._print_for_process(
                f'Ranking prompts {sample_idxs[0]}-{sample_idxs[-1]}',
                only_main=False)

        # 初始化结果存储
        num_samples = len(sample_idxs)
        all_completion_rewards = []
        best_completions = []
        line_contents = []
        all_best_completion_indices = []
        cached_raw_tensors_to_store = [{} for _ in range(num_samples)]
        
        # 如果有缓存张量，使用缓存数据
        if self.cached_reward_raw_tensors is not None:
            cached_raw_tensors_to_store = [
                self.cached_reward_raw_tensors[ix] for ix in sample_idxs]
            assert len(cached_raw_tensors_to_store) == num_samples, (
                f'Cached length {len(cached_raw_tensors_to_store)} does not '
                f'match data length {num_samples}')

        # 处理每个样本
        for process_prompt_idx, sample_idx in enumerate(sample_idxs):
            self._print_for_process(
                f'Sample {process_prompt_idx+1}/{num_samples}', only_main=True)
            
            # 对样本进行排序
            rws, rw_infos, cached_tensors_for_completions = self.rank_prompt(
                sample_idx=sample_idx, log_completion=True)
            completions = data_completions[sample_idx]
            sample = self.dataset[sample_idx]
            
            # 确保样本有索引
            if '__index' not in sample:
                sample['__index'] = sample_idx
            
            # 存储缓存张量
            cached_raw_tensors_to_store[
                process_prompt_idx] = cached_tensors_for_completions
            all_completion_rewards.append(rws)
            
            # 选择最佳完成
            if self.has_reserved_completion_index:
                # 如果有保留的完成索引，特殊处理
                formatted_entry_idx = self.formatted_entry_idx
                if formatted_entry_idx < 0:
                    formatted_entry_idx = formatted_entry_idx + len(rws)
                rws_wo_reserved_idx = [rew_t for rew_t_i, rew_t in enumerate(
                    rws) if rew_t_i != formatted_entry_idx]
                best_completion_idx = np.nanargmax(rws_wo_reserved_idx)

                format_value_best = rw_infos[
                    best_completion_idx]["match_reward"]
                if format_value_best < 0:
                    best_completion_idx = formatted_entry_idx
            else:
                # 选择奖励最高的完成
                best_completion_idx = np.nanargmax(rws)
            
            all_best_completion_indices.append(best_completion_idx)
            best_completion = completions[best_completion_idx]
            best_completions.append(best_completion)

            # 构建输出行内容
            line_content = {**sample}
            new_line_contents = {
                self.update_column: best_completion,
                self.rewards_column: rws,
            }
            line_content.update(new_line_contents)
            line_contents.append(line_content)

        # 移除填充的样本
        if self.padded_idxs > 0:
            all_completion_rewards = all_completion_rewards[:-self.padded_idxs]
            best_completions = best_completions[:-self.padded_idxs]
            all_best_completion_indices = all_best_completion_indices[
                :-self.padded_idxs]
            line_contents = line_contents[:-self.padded_idxs]
            cached_raw_tensors_to_store = cached_raw_tensors_to_store[
                :-self.padded_idxs]

        # 等待所有进程完成
        self.accelerator.wait_for_everyone()
        
        # 收集所有进程的结果
        self._print_for_process(
            'Gathering all_completion_rewards...', only_main=True)
        all_completion_rewards = gather_object(all_completion_rewards)

        # 恢复原始顺序
        all_completion_rewards = [all_completion_rewards[i] for i in inv_perm]
        self._print_for_process(f"{all_completion_rewards}", only_main=True)

        self._print_for_process(
            f'Number of gathered rws: {len(all_completion_rewards)}',
            only_main=True)

        self._print_for_process(
            'Gathering best_completions...', only_main=True)
        best_completions = gather_object(best_completions)
        best_completions = [best_completions[i] for i in inv_perm]

        self._print_for_process(
            'Gathering line_contents...', only_main=True)
        line_contents = gather_object(line_contents)
        line_contents = [line_contents[i] for i in inv_perm]
        
        # 进行完整性检查
        test_idx = 27
        if test_idx is not None and test_idx < all_num_samples:
            self._print_for_process(
                'Comparing idxs sanity check: ', only_main=True)
            self._print_for_process(
                line_contents[test_idx]['__index'], only_main=True)
            self._print_for_process(
                self.dataset[test_idx].get('__index', test_idx), only_main=True)

        self._print_for_process(
            'Gathering all_best_completion_indices...', only_main=True)
        all_best_completion_indices = gather_object(
            all_best_completion_indices)
        all_best_completion_indices = [
            all_best_completion_indices[i] for i in inv_perm]

        self._print_for_process(
            'Gathering cached_raw_tensors_to_store...', only_main=True)
        cached_raw_tensors_to_store = gather_object(
            cached_raw_tensors_to_store)
        cached_raw_tensors_to_store = [
            cached_raw_tensors_to_store[i] for i in inv_perm]
        
        # 计算完成选择的频率分布
        completion_wins = value_frequencies(values=all_best_completion_indices)

        # 主进程保存结果
        if self.accelerator.is_main_process:
            print('Win index percentages: ')
            print(completion_wins)
            print('Storing data to disk...')
            ds = Dataset.from_list(line_contents)

            # 保存排序后的数据
            ds.to_json(self.out_file_path)

            # 如果需要，保存原始得分
            if self.store_raw_scores:
                print('Storing pickle of raw tensors...')
                save_pickle(
                    fname=self.generated_scores_fname,
                    directory=self.ranked_data_dir,
                    cached_reward_raw_tensors=cached_raw_tensors_to_store,
                )

        # 等待所有进程完成
        self.accelerator.wait_for_everyone()
        
        # 记录到wandb
        if wandb.run is not None:
            res_to_log = completion_wins
            res_to_log['step'] = 1
            wandb.log(res_to_log)
        
        self._print_for_process(
            f'Ranking complete, results stored at {self.out_file_path}',
            only_main=False)

        return TrainOutput(
            global_step=1, training_loss=0.0, metrics=completion_wins)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        记录日志信息
        
        Args:
            logs: 日志字典
            start_time: 开始时间（可选）
        """
        # 计算指标的平均值
        metrics = {key: sum(val) / len(val)
                   for key, val in self._metrics.items()}

        # 如果是评估指标，添加eval_前缀
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs)

        # 清空指标缓存
        self._metrics.clear()

    def log_metrics(self, split, metrics):
        """记录指标（空实现）"""
        pass

    def save_metrics(self, split, metrics):
        """保存指标（空实现）"""
        pass

    def save_state(self):
        """保存状态（空实现）"""
        pass

    def save_model(self, output_dir):
        """保存模型（空实现）"""
        pass

    def create_model_card(self, metadata):
        """创建模型卡片（空实现）"""
        pass

    def push_to_hub(self):
        """推送到Hub（空实现）"""
        pass


@dataclass
class DataConcatenatorArgs(TrainingArguments):
    """
    数据连接器的参数配置类
    """
    # 连接后数据保存目录
    concatenated_data_dir: Optional[str] = None
    # 目标完成列名（可以是字符串或字符串列表）
    target_completions_column: str | List[str] = field(default="completions")
    # 连接后的完成列名
    concatenated_completions_column: str = field(default="completions")


class DataCompletionConcatenator(Trainer, TeacherTrainer):
    """
    数据完成连接器类，用于将多个数据集的完成字段连接在一起
    """
    def __init__(
            self,
            args: DataConcatenatorArgs = None,  # 连接器参数
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,  # 训练数据集
            eval_dataset: Optional[Union[Dataset, IterableDataset,
                                         dict[str, Union[Dataset, IterableDataset]]]] = None,  # 评估数据集
            tokenizer: Optional[PreTrainedTokenizerBase] = None,  # tokenizer
            datasets_to_concatenate: List[
                Dataset | IterableDataset | DictConfig] = None,  # 要连接的数据集列表
            **kwargs):

        self.args = args
        self.concatenated_data_dir = self.args.concatenated_data_dir
        self.out_file_path = os.path.join(
            self.concatenated_data_dir, "merged.json")
        
        # 处理目标完成列名
        self.target_completions_column = []
        self.concatenated_completions_column = (
            self.args.concatenated_completions_column)
        
        if isinstance(self.args.target_completions_column, str):
            # 如果是字符串，为每个数据集使用相同的列名
            self.target_completions_column = [
                self.args.target_completions_column
                for _ in datasets_to_concatenate]
        else:
            # 如果是列表，直接使用
            self.target_completions_column = self.args.target_completions_column
        
        # 确保数据集数量和列名数量匹配
        assert len(datasets_to_concatenate) == len(
            self.target_completions_column)
        
        # 实例化数据集
        self.datasets_to_concatenate = []
        if isinstance(tokenizer, (dict, DictConfig)):
            tokenizer = instantiate_from_target(tokenizer)
        
        for data, col in zip(
                datasets_to_concatenate, self.target_completions_column):
            if isinstance(data, (Dataset, IterableDataset)):
                # 如果已经是数据集对象，直接使用
                inst_data = data
            else:
                # 如果是配置，实例化数据集
                assert isinstance(data, (DictConfig, dict)), f'{type(data)}'
                inst_data = instantiate_from_target(data, tokenizer=tokenizer)
            
            # 获取训练数据集
            inst_data = inst_data['train_dataset']
            print(inst_data)
            print(col)
            
            # 确保列存在
            assert col in inst_data.column_names
            self.datasets_to_concatenate.append(inst_data)

        # 设置基础数据集和完成数据
        self.dataset = self.datasets_to_concatenate[0]
        self.dataset_completions = [data[compl_col] for data, compl_col in zip(
            self.datasets_to_concatenate, self.target_completions_column)]
        
        # 验证所有数据集长度相同
        self.data_len = len(self.dataset)
        print(f'target data len {self.data_len}')
        for data in self.datasets_to_concatenate:
            print(f'other data len {len(data)}')
            assert len(data) == self.data_len, (
                "Trying to concatenate different data sizes")
        
        # 获取所有数据索引
        self.all_data_indices = [
            data['__index'] for data in self.datasets_to_concatenate if
            '__index' in data]

    def get_all_completions_for_idx(self, idx):
        """
        获取指定索引的所有完成
        
        Args:
            idx: 样本索引
        
        Returns:
            所有完成的列表
        """
        all_completions = []
        for dataset_comp in self.dataset_completions:
            dataset_comp_idx = dataset_comp[idx]
            if isinstance(dataset_comp_idx, str):
                # 如果是字符串，直接添加
                all_completions.append(dataset_comp_idx)
            else:
                # 如果是列表或元组，扩展到结果中
                assert isinstance(dataset_comp_idx, (tuple, list)), (
                    f'{type(dataset_comp_idx)}')
                all_completions.extend(list(dataset_comp_idx))
        return all_completions

    def train(self, resume_from_checkpoint=None):
        """
        执行训练过程，实际上是连接数据集
        
        Args:
            resume_from_checkpoint: 从检查点恢复（未使用）
        
        Returns:
            训练输出结果
        """
        line_contents = []

        # 统计完成数量
        all_num_original_completions = []
        all_num_new_completions = []
        
        # 处理每个样本
        for sample_idx in tqdm(range(self.data_len)):
            # 获取原始完成数量
            completions = self.dataset_completions[0][sample_idx]
            num_completions = len(completions)
            all_num_original_completions.append(num_completions)
            
            # 获取样本数据
            sample = self.dataset[sample_idx]
            
            # 获取所有数据集的完成
            new_completions = self.get_all_completions_for_idx(idx=sample_idx)
            num_new_completions = len(new_completions)
            all_num_new_completions.append(num_new_completions)
            
            # 获取样本真实索引
            if '__index' in sample:
                sample_true_idx = sample['__index']
            
            # 构建新的行内容
            line_content = {**sample}
            line_content.update(
                {self.concatenated_completions_column: new_completions})
            line_contents.append(line_content)

        # 计算平均完成数量
        mean_num_original_completions = np.mean(all_num_original_completions)
        mean_num_new_completions = np.mean(all_num_new_completions)

        print('Made new dataset with '
              f'{mean_num_original_completions}-->'
              f'{mean_num_new_completions} average completions')

        # 创建新数据集并保存
        ds = Dataset.from_list(line_contents)

        ds.to_json(self.out_file_path)

        print(f'Merging complete, results stored at {self.out_file_path}')
        return TrainOutput(global_step=1, training_loss=0.0, metrics={})

    def log_metrics(self, split, metrics):
        """记录指标（空实现）"""
        pass

    def save_metrics(self, split, metrics):
        """保存指标（空实现）"""
        pass

    def save_state(self):
        """保存状态（空实现）"""
        pass

    def save_model(self, output_dir):
        """保存模型（空实现）"""
        pass

    def create_model_card(self, metadata):
        """创建模型卡片（空实现）"""
        pass

    def push_to_hub(self):
        """推送到Hub（空实现）"""
        pass

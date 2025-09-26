import os
import abc
import torch
import accelerate
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
from transformers import PreTrainedTokenizer
from .grpo import GRPOTrainer
from .grpo_config import GRPOConfig
from .teacher_base import TeacherReward, TeacherTrainer
from .utils_trl_15 import prepare_deepspeed
from transformers import AutoModelForCausalLM


class TeacherGRPOTrainer(GRPOTrainer, TeacherTrainer):
    """
    教师GRPO训练器类，继承自GRPOTrainer和TeacherTrainer
    
    该类结合了GRPO（Group Relative Policy Optimization）训练器和教师训练器的功能，
    用于实现基于教师-学生模型架构的强化学习训练。
    """
    
    def __init__(
            self,
            *args,
            student_model=None,  # 学生模型，可以是None、字符串路径或模型实例

            use_reference_teacher_model=False,  # 是否使用参考模型作为教师模型
            student_model_init_kwargs=None,  # 学生模型初始化参数
            logging_prob=0.0,  # 日志记录概率

            disable_student_offloading=False,  # 是否禁用学生模型卸载
            **kwargs):
        """
        初始化教师GRPO训练器
        
        Args:
            *args: 传递给GRPOTrainer的位置参数
            student_model: 学生模型，可以是None（使用参考模型）、模型路径字符串或模型实例
            use_reference_teacher_model: 是否使用参考模型作为教师模型，默认False使用主模型
            student_model_init_kwargs: 学生模型初始化时的关键字参数
            logging_prob: 日志记录的概率，用于控制日志输出频率
            disable_student_offloading: 是否禁用学生模型的CPU卸载功能
            **kwargs: 传递给GRPOTrainer的其他关键字参数
        """

        # 初始化GRPO训练器
        GRPOTrainer.__init__(self, *args, **kwargs)
        
        # 如果没有提供学生模型初始化参数，使用存储的模型初始化参数
        if student_model_init_kwargs is None:
            student_model_init_kwargs = self._stored_model_init_kwargs

        # 确定是否需要将学生模型卸载到CPU
        # 只有在启用了未训练模型卸载且没有禁用学生模型卸载时才进行卸载
        offload_student_model = self.offload_untrained_models and (
            not disable_student_offloading)
        
        # 根据student_model参数的类型进行不同的处理
        if student_model is None:
            # 如果没有提供学生模型，使用参考模型作为学生模型
            self.student_model = self.ref_model
        elif isinstance(student_model, str):
            # 如果提供的是模型路径字符串，从预训练模型加载
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model, **student_model_init_kwargs)
            
            # 根据是否启用DeepSpeed进行不同的模型准备
            if self.is_deepspeed_enabled:
                # 使用DeepSpeed准备模型
                self.student_model = prepare_deepspeed(
                    self.student_model,
                    self.accelerator,
                    offload_to_cpu=offload_student_model)
            else:
                # 使用Accelerate准备模型，设置为评估模式
                self.student_model = self.accelerator.prepare_model(
                    self.student_model, evaluation_mode=True)

                # 如果需要卸载，将模型卸载到CPU
                if offload_student_model:
                    self.student_model = accelerate.cpu_offload(
                        model=self.student_model)
        else:
            # 如果提供的是模型实例，目前未实现此功能
            raise NotImplementedError
            self.student_model = student_model

        # 根据use_reference_teacher_model参数选择教师模型
        if use_reference_teacher_model:
            # 使用参考模型作为教师模型
            teacher_model = self.ref_model
        else:
            # 使用主模型作为教师模型
            teacher_model = self.model

        # 初始化教师训练器
        TeacherTrainer.__init__(
            self,
            student_model=self.student_model,  # 学生模型
            teacher_model=teacher_model,       # 教师模型
            tokenizer=self.processing_class,   # 分词器
            reward_functions=self.reward_funcs, # 奖励函数
            output_dir=self.args.output_dir,   # 输出目录
            logging_prob=logging_prob,         # 日志记录概率
        )

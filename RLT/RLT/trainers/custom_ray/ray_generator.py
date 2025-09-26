import torch
import ray
import os
import time

from typing import Optional, Dict, Any, Callable, List

from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from ray.util.placement_group import placement_group
from .vllm_engine import (
    create_vllm_engines, batch_vllm_engine_call, get_resource_info)
from .vllm_worker_wrap import stateless_init_process_group
from vllm.utils import get_ip, get_open_port


def get_rank_safe():
    """
    安全地获取当前进程的分布式训练rank
    如果分布式训练未初始化，返回0
    """
    if (torch.distributed.is_available() and
            torch.distributed.is_initialized()):
        return torch.distributed.get_rank()
    return 0


def get_world_size_safe():
    """
    安全地获取分布式训练的总进程数
    如果分布式训练未初始化，返回1
    """
    if (torch.distributed.is_available() and
            torch.distributed.is_initialized()):
        return torch.distributed.get_world_size()
    return 1


def barrier_safe():
    """
    安全地执行分布式训练同步屏障
    如果分布式训练未初始化，则跳过
    """
    if (torch.distributed.is_available() and
            torch.distributed.is_initialized()):
        torch.distributed.barrier()


def _extract_cuda_metadata(tensor: torch.Tensor):
    """
    提取CUDA张量的元数据信息，用于跨进程共享张量
    
    Args:
        tensor: 需要提取元数据的CUDA张量
        
    Returns:
        包含张量元数据的字典，包括存储信息、设备信息、引用计数等
    """
    storage = tensor._typed_storage()
    (
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()

    return {
        "dtype": tensor.dtype,  # 张量数据类型
        "tensor_size": tensor.size(),  # 张量尺寸
        "tensor_stride": tensor.stride(),  # 张量步长
        "tensor_offset": tensor.storage_offset(),  # 张量在存储中的偏移
        "storage_cls": type(storage),  # 存储类型
        "storage_device": storage_device,  # 存储设备
        "storage_handle": storage_handle,  # 存储句柄
        "storage_size_bytes": storage_size_bytes,  # 存储大小（字节）
        "storage_offset_bytes": storage_offset_bytes,  # 存储偏移（字节）
        "requires_grad": tensor.requires_grad,  # 是否需要梯度
        "ref_counter_handle": ref_counter_handle,  # 引用计数句柄
        "ref_counter_offset": ref_counter_offset,  # 引用计数偏移
        "event_handle": event_handle,  # 事件句柄
        "event_sync_required": event_sync_required,  # 是否需要事件同步
    }


class RayGeneratorActor:
    """
    基于Ray的分布式文本生成器Actor类
    使用vLLM引擎进行高效的大语言模型推理
    """
    
    def __init__(
        self,
        model: str,  # 模型名称或路径
        revision: str = None,  # 模型版本
        tokenizer: Optional[Any | str] = None,  # 分词器
        seed: int = 42,  # 随机种子

        # Ray分布式配置
        ray_num_nodes: int = 1,  # Ray节点数量
        ray_tensor_parallelism: int = 1,  # 张量并行度
        ray_data_parallelism: int = 1,  # 数据并行度
        
        # vLLM引擎配置
        vllm_gpu_memory_utilization: float = 0.9,  # GPU内存利用率
        vllm_dtype: str = "auto",  # 数据类型
        enable_prefix_caching: bool = False,  # 是否启用前缀缓存
        enforce_eager: bool = True,  # 是否强制eager模式
        sleep_level: int = 0,  # 睡眠级别（用于资源管理）

        # 长度限制配置
        max_prompt_length: int = 32768,  # 最大提示长度
        max_tokens: int = 32768,  # 最大token数
        max_completion_length: Optional[int] = 32768,  # 最大补全长度

        # 采样参数配置
        temperature: float = 1.0,  # 温度参数
        top_k: int = -1,  # top-k采样
        top_p: float = 1.0,  # top-p采样
        repetition_penalty: float = 1.0,  # 重复惩罚
        presence_penalty: float = 0.0,  # 存在惩罚
        frequency_penalty: float = 0.0,  # 频率惩罚
        
        # 通信配置
        collective_rpc_mode: Optional[str] = 'nccl',  # 集合通信模式

        # 调试和显示配置
        verbose_generator: bool = True,  # 是否详细输出
        reserved_gpus: int = 0,  # 保留的GPU数量
        activate_debugging_logs: bool = False,  # 是否激活调试日志
        sampling_params=None,  # 自定义采样参数
        show_progress: bool = False,  # 是否显示进度
    ):
        """初始化Ray生成器Actor"""
        self.sampling_params = sampling_params
        self.show_progress = show_progress

        self.activate_debugging_logs = activate_debugging_logs
        if reserved_gpus is None:
            reserved_gpus = 0

        # 检查GPU共享模式
        if reserved_gpus > 0:
            self._print_debugging_logs('检查睡眠级别...')
            self.shared_gpus = False  # 不共享GPU
            assert sleep_level == 0  # 保留GPU时不允许睡眠
        else:
            self.shared_gpus = True  # 共享GPU模式
            
        self.model = model
        
        # 初始化分词器
        if tokenizer is None:
            tokenizer = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        # 保存配置参数
        self.seed = seed
        self.ray_num_nodes = ray_num_nodes
        self.ray_tensor_parallelism = ray_tensor_parallelism
        self.ray_data_parallelism = ray_data_parallelism
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_dtype = vllm_dtype
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        
        # 睡眠模式配置
        self.sleep_level = int(sleep_level)
        self.enable_sleep = sleep_level > 0
        if self.enable_sleep:
            assert self.sleep_level in [1, 2]
            if self.sleep_level == 2:
                # 睡眠级别2暂未实现
                raise NotImplementedError

        # 长度限制配置
        self.max_prompt_length = max_prompt_length
        self.max_tokens = max_tokens
        self.max_completion_length = max_completion_length
        if self.max_completion_length is None:
            self.max_completion_length = self.max_tokens

        # 采样参数配置
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        # 计算GPU配置
        self.num_gpus_per_node = (
            self.ray_tensor_parallelism*self.ray_data_parallelism)
        self.total_devices = self.ray_num_nodes*self.num_gpus_per_node

        # 设置vLLM使用的GPU设备
        if not self.shared_gpus:
            vllm_devices = [
                i + reserved_gpus for i in range(self.total_devices)]
            if self.ray_num_nodes > 1:
                # 多节点模式暂未实现
                raise NotImplementedError
        else:
            vllm_devices = None

        pg = None
        self.model_awoken = False

        # Ray运行时环境配置
        runtime_env = {
            'env_vars': {
                "RAY_memory_monitor_refresh_ms": "0",  # 内存监控刷新间隔
                "RAY_memory_usage_threshold": "3"  # 内存使用阈值
            }
        }

        # 设置CUDA可见设备
        if vllm_devices is not None:
            print(
                f'为ray actors初始化设置可见设备为 {vllm_devices}.')
            assert 0 not in vllm_devices
            vllm_devices_str = ",".join(str(d) for d in vllm_devices)
            original_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices_str
        else:
            print(
                f'Vllm设备 {vllm_devices} 为None，默认可见性为:')
            print(f'{os.environ.get("CUDA_VISIBLE_DEVICES", None)}.')

        # 初始化Ray
        ray.init(runtime_env=runtime_env)
        self._print_debugging_logs(ray.available_resources())
        
        # 创建vLLM引擎
        self.vllm_engines = create_vllm_engines(
            num_engines=ray_data_parallelism,  # 引擎数量
            tensor_parallel_size=ray_tensor_parallelism,  # 张量并行大小
            pretrain=model,  # 预训练模型
            revision=revision,  # 模型版本
            seed=seed,  # 随机种子
            enable_prefix_caching=enable_prefix_caching,  # 前缀缓存
            enforce_eager=enforce_eager,  # 强制eager模式
            max_model_len=max_tokens,  # 最大模型长度
            num_total_actors=1,  # 总actor数量
            dtype=self.vllm_dtype,  # 数据类型
            shared_pg=pg,  # 共享放置组
            gpu_memory_utilization=vllm_gpu_memory_utilization,  # GPU内存利用率
            vllm_enable_sleep=self.enable_sleep,  # 启用睡眠
            sleep_level=self.sleep_level,  # 睡眠级别
            vllm_devices=vllm_devices,  # vLLM设备
            show_progress=show_progress,  # 显示进度
        )
        
        self.asleep = False
        self.sleep_if_needed()  # 如果需要则进入睡眠状态
        print(f'已初始化: {len(self.vllm_engines)} 个引擎')

        # 恢复CUDA可见设备设置
        if vllm_devices is not None:
            print(f'将可见设备设置回 {original_devices}')
            if original_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_devices
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # 初始化集合通信
        self.collective_rpc_mode = collective_rpc_mode
        self.use_collective_rpc_mode = (collective_rpc_mode is not None) and (
            collective_rpc_mode.lower() != 'none')
        if self.use_collective_rpc_mode:
            assert collective_rpc_mode == 'nccl'  # 目前只支持NCCL
            
        if self.use_collective_rpc_mode:
            # 获取权重更新的主地址和端口
            self.weight_update_master_addr = get_ip()
            self.weight_update_master_port = get_open_port()
            
            # 计算rank偏移
            if reserved_gpus > 0:
                rank_offsets_shift = 1
            else:
                rank_offsets_shift = 0
            rank_offsets = [i*self.ray_tensor_parallelism + rank_offsets_shift
                            for i in range(self.ray_data_parallelism)]

            print('用于初始化权重更新进程组的rank偏移 '
                  f'{rank_offsets}, {type(rank_offsets)}.')
            
            self.total_update_devices = self.total_devices
            if not self.shared_gpus:
                self.total_update_devices += 1

            # 准备引擎初始化参数
            engines_init_args = [(
                self.weight_update_master_addr,
                self.weight_update_master_port,
                rank_offset,
                self.total_update_devices)
                for rank_offset in rank_offsets]

            # 初始化权重更新组
            initialization_handles = [
                engine.init_weight_update_group_s.remote(*init_args) for
                engine, init_args in zip(self.vllm_engines, engines_init_args)
            ]
            
            if not self.shared_gpus:
                self.main_engine_idx = None
                # 初始化无状态进程组
                self.model_update_group = stateless_init_process_group(
                    master_address=self.weight_update_master_addr,
                    master_port=self.weight_update_master_port,
                    rank=0,
                    world_size=self.total_update_devices,
                    device=torch.device("cuda:0"),
                )
                print('无状态进程组初始化完成')
            else:
                self.main_engine_idx = 0
                self.model_update_group = None

            print('获取进程更新组初始化句柄....')
            update_groups_infos = ray.get(initialization_handles)
            if self.shared_gpus:
                self.update_group_info = update_groups_infos[
                    self.main_engine_idx][0]
            else:
                self.update_group_info = None
            print('成功初始化进程更新组!')

    def _print_debugging_logs(self, to_print: str):
        """打印调试日志"""
        if self.activate_debugging_logs:
            print(f'Ray生成器: {to_print}')

    def update_state_dict(
            self, state_dict, clone_weight=True, main_engine_idx=0,):
        """
        更新模型状态字典到所有vLLM引擎
        
        Args:
            state_dict: 要更新的状态字典
            clone_weight: 是否克隆权重
            main_engine_idx: 主引擎索引
        """
        if self.main_engine_idx is not None:
            main_engine = self.vllm_engines[self.main_engine_idx]
            other_engines = [self.vllm_engines[i] for i in range(
                len(self.vllm_engines)) if i != self.main_engine_idx]
            device = f'cuda:{self.main_engine_idx}'
        else:
            main_engine = None
            device = 'cuda:0'
            
        params_names = list(state_dict.keys())
        for k in params_names:
            p = state_dict[k]
            dtype = p.dtype
            shape = p.shape
            p_class = type(p)
            
            if self.shared_gpus:
                # 共享GPU模式：使用CUDA元数据共享
                p_metadata = _extract_cuda_metadata(tensor=p)

                update_ref = main_engine.update_self_weight_from_metadata.remote(
                    name=k, p_metadata=p_metadata, clone=True)

                handles = [
                    engine.update_weight_s.remote(
                        name=k, dtype=dtype, shape=shape)
                    for engine in other_engines
                ]
                handles = [update_ref] + handles

                self._print_debugging_logs(f'同步: {len(handles)} 个模型')
                ray.get(handles)
            else:
                # 非共享GPU模式：使用广播通信
                self.model_update_group.broadcast(
                    p,
                    src=0, stream=torch.cuda.current_stream())

                handles = [
                    engine.update_weight_s.remote(
                        name=k, dtype=dtype, shape=shape)
                    for engine in self.vllm_engines
                ]
                ray.get(handles)

    def generate(self, all_prompts: List[str], return_only_completions=False,
                 update_iteration=False, **kwargs):
        """
        生成文本
        
        Args:
            all_prompts: 所有输入提示列表
            return_only_completions: 是否只返回补全部分
            update_iteration: 是否更新迭代
            **kwargs: 其他关键字参数
            
        Returns:
            生成的文本输出
        """
        self.wake_if_needed()  # 如果需要则唤醒引擎

        # 设置采样参数
        if self.sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_completion_length,
            )
        else:
            sampling_params = self.sampling_params

        rank = 0
        world_size = 1

        llms = self.vllm_engines

        # 将请求分发到各个引擎
        refs = []
        batch_size = (len(all_prompts) + len(llms) - 1)//len(llms)  # 计算批次大小
        for i, llm in enumerate(llms):
            prompts = all_prompts[
                i*batch_size:(i + 1)*batch_size]  # 分割提示
            refs.append(
                llm.add_requests.remote(
                    rank,
                    prompts=prompts,
                    sampling_params=sampling_params,
                ),
            )
        ray.get(refs)  # 等待所有请求添加完成

        # 获取所有引擎的响应
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])  # 合并所有输出

        self.sleep_if_needed()  # 如果需要则进入睡眠状态

        torch.cuda.synchronize()  # 同步CUDA操作

        # 根据需要返回不同格式的结果
        if return_only_completions:
            all_output_completions = []
            for output in all_outputs:
                all_output_completions.extend(output.outputs)
            return all_output_completions

        return all_outputs

    def wake_if_needed(self):
        """如果引擎处于睡眠状态则唤醒"""
        if self.enable_sleep and self.asleep:
            self._print_debugging_logs('等待唤醒...')
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.cuda.synchronize()
            self._print_debugging_logs('已唤醒!')
            self.asleep = False

    def sleep_if_needed(self):
        """如果启用睡眠且当前未睡眠则进入睡眠状态"""
        if self.enable_sleep and (not self.asleep):
            batch_vllm_engine_call(
                self.vllm_engines, "sleep", level=self.sleep_level)
            torch.cuda.synchronize()
            self.asleep = True

    def reset_prefix_cache(self,):
        """重置前缀缓存"""
        batch_vllm_engine_call(
            self.vllm_engines, "reset_prefix_cache")

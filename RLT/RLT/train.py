import logging
import os
import random
import re
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from transformers.trainer_utils import get_last_checkpoint

# 设置环境变量以启用HuggingFace Hub的快速传输功能
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 配置日志记录，设置日志级别为INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    """
    初始化Weights & Biases (wandb) 实验跟踪
    
    Args:
        cfg: 配置对象
        run_name: 运行名称
        group_name: 组名称
        log_dir: 日志目录
    
    Returns:
        wandb: 初始化后的wandb对象
    """
    import wandb
    from omegaconf import OmegaConf

    # 将OmegaConf配置转换为普通字典
    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,  # 解析变量引用
        throw_on_missing=False,  # 缺失值不抛出异常
    )
    # 添加额外的配置信息
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    # 初始化wandb运行，限制名称长度为127字符
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


def get_checkpoint(output_dir):
    """
    获取输出目录中的最新检查点
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        str or None: 最新检查点路径，如果不存在则返回None
    """
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def get_total_devices():
    """
    获取总设备数量（用于分布式训练）
    
    Returns:
        int: 总设备数量
    """
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1


def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    """
    计算梯度累积步数
    
    Args:
        train_batch_size: 总训练批次大小
        per_device_train_batch_size: 每个设备的训练批次大小
    
    Returns:
        int: 梯度累积步数
    
    Raises:
        ValueError: 当train_batch_size不能被(per_device_batch*total_devices)整除时
    """
    total_devices = get_total_devices()

    # 计算所有设备的总批次大小
    div = per_device_train_batch_size*total_devices
    steps = train_batch_size/div
    
    # 检查是否能整除
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={div}"
        )
    return int(steps)


@hydra.main(config_path="cfgs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """
    主训练函数
    
    Args:
        cfg: Hydra配置对象
    """
    # 打印配置信息
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 判断是否为主进程（用于分布式训练）
    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main_process = int(os.environ["RANK"]) == 0
    else:
        is_main_process = True

    # 如果配置中缺少梯度累积步数，则自动计算
    if OmegaConf.is_missing(cfg, "gradient_accumulation_steps"):
        accumulation_steps = compute_accumulation_steps(
            train_batch_size=cfg.train_batch_size,
            per_device_train_batch_size=cfg.per_device_train_batch_size)
        cfg.gradient_accumulation_steps = accumulation_steps

    logger.info(f"Accumulation steps {cfg.gradient_accumulation_steps} ----")

    # 检查是否使用wandb进行实验跟踪
    using_wandb = False
    if isinstance(cfg.report_to, str):
        using_wandb = cfg.report_to == 'wandb'
    elif cfg.report_to is not None:
        for v in cfg.report_to:
            using_wandb = using_wandb or (v == 'wandb')

    # 如果使用wandb且为主进程，则初始化wandb
    if using_wandb and is_main_process:
        wandb = wandb_init(
            cfg=cfg,
            group_name=cfg.wandb_group_name,
            run_name=cfg.wandb_run_name,
            log_dir=cfg.output_dir,
        )

    # 实例化分词器
    tokenizer = hydra.utils.instantiate(cfg.make_tokenizer_fn)

    # 实例化数据集
    datasets = hydra.utils.instantiate(
        cfg.make_dataset_fn, tokenizer=tokenizer)

    # 实例化训练器
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        **datasets,
    )

    print('Model initialized!!!')

    # 查找现有的检查点
    last_checkpoint = get_checkpoint(cfg.output_dir)
    if not last_checkpoint and cfg.resume_from is not None:
        last_checkpoint = get_checkpoint(cfg.resume_from)
    
    if last_checkpoint:
        logger.info("Found checkpoint, resuming training run from "
                    f"{last_checkpoint}.")
    else:
        logger.info("No existing checkpoint, initializing new model")

    # 开始训练
    logger.info(f"Training  {datetime.now()}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info(f"Training complete {datetime.now()}")

    # 记录和保存训练指标
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 如果配置要求保存最终模型
    if cfg.save_final_model:
        logger.info(f"Saving final model at {cfg.output_dir}")
        # 启用缓存以提高推理性能
        trainer.model.config.use_cache = True
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        logger.info(f"Done saving {datetime.now()}")

    # 如果是主进程且需要推送到hub，创建模型卡片
    if is_main_process and cfg.push_to_hub:
        tags = cfg.tags if cfg.tags is not None else []
        trainer.create_model_card({"tags": tags})
    
    # 推送模型到HuggingFace Hub
    if cfg.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    # 如果是主进程且配置了训练后回调，则执行
    if is_main_process and cfg.call_post_training is not None:
        hydra.utils.instantiate(cfg.call_post_training)


if __name__ == "__main__":
    main()

from datasets import load_dataset, concatenate_datasets
from .utils import make_masked_sft_collator
from .reasoning_datasets_info import (
    DATA_CONFIGS, wrap_string_between_tag, grab_text_between_tag, get_tags,
    ReasoningData)


def add_indices(ds):
    """
    为数据集添加索引列
    
    Args:
        ds: 数据集对象
        
    Returns:
        添加了__index列的数据集
    """
    if "__index" not in ds.column_names:
        ds = ds.map(lambda x, i: {"__index": i}, with_indices=True)
    return ds


def get_process_line_fn(dataset_id_or_path):
    """
    根据数据集路径获取对应的数据处理函数
    
    Args:
        dataset_id_or_path (str): 数据集ID或路径
        
    Returns:
        function: 用于处理单行数据的函数
    """
    # 从配置中获取对应数据集的配置信息
    data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
    system_prompt = data.system_prompt

    def process_line_fn(line, tokenizer):
        """
        处理单行数据，将其转换为聊天格式
        
        Args:
            line (dict): 单行数据
            tokenizer: 分词器
            
        Returns:
            dict: 包含格式化文本的字典
        """
        # 从数据行中提取问题和完整回答
        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        
        # 构建聊天消息格式
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question_content,
            },
            {
                "role": "assistant",
                "content": thought_process_and_solution,
            }
        ]
        
        # 使用分词器的聊天模板格式化消息
        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        return {"text": line_text}
    return process_line_fn


def load_formatted_sft_dataset(
        tokenizer,
        dataset_id_or_path,
        dataset_local_directory=None,
        train_split='train',
        val_split=None,
        process_line_fn=None,
        model_name_or_path=None,
        completion_only_training=True,
        custom_start_of_response=None,
        keep_columns=None,
        add_dataset_indices=False,
        artificial_epochs=None,
        **dataset_loading_kwargs,
):
    """
    加载并格式化SFT（监督微调）数据集
    
    Args:
        tokenizer: 分词器对象
        dataset_id_or_path (str): 数据集ID或路径
        dataset_local_directory (str, optional): 本地数据集目录，默认使用dataset_id_or_path
        train_split (str): 训练集分割名称，默认为'train'
        val_split (str, optional): 验证集分割名称，如果为None则不加载验证集
        process_line_fn (function or list, optional): 数据处理函数或函数列表
        model_name_or_path (str, optional): 模型名称或路径
        completion_only_training (bool): 是否只训练完成部分，默认为True
        custom_start_of_response (str, optional): 自定义响应开始标记
        keep_columns (list, optional): 要保留的列名列表
        add_dataset_indices (bool): 是否添加数据集索引，默认为False
        artificial_epochs (int, optional): 人工epoch数量
        **dataset_loading_kwargs: 数据集加载的额外参数
        
    Returns:
        dict: 包含训练数据集、验证数据集和数据整理器的字典
    """

    # 如果没有指定本地目录，使用数据集ID或路径
    if dataset_local_directory is None:
        dataset_local_directory = dataset_id_or_path
    
    # 加载数据集
    dataset = load_dataset(dataset_local_directory, **dataset_loading_kwargs)
    train_dataset = dataset[train_split]
    
    # 如果需要，为训练数据集添加索引
    if add_dataset_indices:
        train_dataset = add_indices(train_dataset)
    
    # 如果提供了数据处理函数，则处理训练数据集
    if process_line_fn is not None:
        if isinstance(process_line_fn, (list, tuple)):
            # 如果是函数列表，分别处理并合并结果
            processed_train_datasets = []
            for fn in process_line_fn:
                processed = train_dataset.map(
                    lambda x, fn=fn: fn(x, tokenizer))
                processed_train_datasets.append(processed)
            train_dataset = concatenate_datasets(
                processed_train_datasets)
        else:
            # 如果是单个函数，直接处理
            print('not loading from cache')
            train_dataset = train_dataset.map(
                lambda x: process_line_fn(x,  tokenizer))
    
    # 处理验证数据集
    if val_split is None:
        val_dataset = None
    else:
        val_dataset = dataset[val_split]
        # 如果需要，为验证数据集添加索引
        if add_dataset_indices:
            val_dataset = add_indices(val_dataset)
        # 如果提供了数据处理函数，则处理验证数据集
        if process_line_fn is not None:
            if isinstance(process_line_fn, (list, tuple)):
                # 如果是函数列表，分别处理并合并结果
                processed_val_datasets = []
                for fn in process_line_fn:
                    processed = val_dataset.map(
                        lambda x, fn=fn: fn(x, tokenizer))
                    processed_val_datasets.append(processed)
                val_dataset = concatenate_datasets(
                    processed_val_datasets)
            else:
                # 如果是单个函数，直接处理
                val_dataset = val_dataset.map(
                    lambda x: process_line_fn(x,  tokenizer))
    
    # 如果指定了要保留的列，则移除其他列
    if keep_columns is not None:
        train_dataset = train_dataset.remove_columns(
            [col for col in train_dataset.column_names
             if col not in keep_columns])

    # 处理人工epoch（用于数据增强）
    if artificial_epochs is not None:
        assert artificial_epochs == 1, (
            'Artificial epoch, moved to GRPO to avoid shuffling samples between'
            ' different epochs.')

        # 复制数据集指定次数
        train_dataset = concatenate_datasets(
            [train_dataset]*artificial_epochs)
    
    # 构建输出数据字典
    out_data = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 如果启用了仅完成部分训练，添加数据整理器
    if completion_only_training:
        out_data['data_collator'] = make_masked_sft_collator(
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            custom_start_of_response=custom_start_of_response,
        )
    return out_data

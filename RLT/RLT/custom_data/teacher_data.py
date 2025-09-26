from datasets import load_dataset
import re
from typing import Optional
from .reasoning_datasets_info import (
    DATA_CONFIGS, wrap_string_between_tag, grab_text_between_tag, get_tags,
    ReasoningData, extract_until)

# Stratos风格的教师提示词模板
# 定义了教师模型的角色和响应格式要求
STRATOS_STYLE_TEACHER_PROMPT = str(
    "Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. "
    "Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Solution and Explanation. "
    "In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. "
    "In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\\n\\n'} <|end_of_explanation|> Each step should show detailed considerations leading to your solutions such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
)

# 教师系统提示词和标签配置字典
# 包含不同风格的系统提示词、解决方案标签、解释标签、前缀、分隔符和填充
TEACHER_SYSTEM_PROMPTS_AND_TAGS = {
    'stratos': (
        STRATOS_STYLE_TEACHER_PROMPT,  # 系统提示词
        'solution',                    # 解决方案标签
        'explanation',                 # 解释标签
        '',                           # 前缀
        '\n\n',                       # 分隔符
        '\n\n'                        # 填充
    ),
}


def process_assistant_message(
    solution_content, think_content, new_solution_tag, explanation_tag
):
    """
    处理助手消息，将解决方案内容和思考内容用指定标签包装
    
    Args:
        solution_content (str): 解决方案内容
        think_content (str): 思考过程内容
        new_solution_tag (str): 新的解决方案标签名称
        explanation_tag (str): 解释标签名称
        
    Returns:
        tuple: (包装后的解决方案, 包装后的解释)
    """
    # 用解决方案标签包装解决方案内容
    solution = wrap_string_between_tag(
        s=solution_content, tag=new_solution_tag)
    # 用解释标签包装思考内容
    explanation = wrap_string_between_tag(s=think_content, tag=explanation_tag)
    return solution, explanation


def get_teacher_data_mask_delimiter(
        mask_teacher_answer,
        system_prompt_style='stratos',
        include_end_of_solution_tag=False,
        dataset_id_or_path=None,
):
    """
    获取教师数据的掩码分隔符，用于在训练时掩盖教师答案的特定部分
    
    Args:
        mask_teacher_answer (bool): 是否掩盖教师答案
        system_prompt_style (str): 系统提示词风格，默认为'stratos'
        include_end_of_solution_tag (bool): 是否包含解决方案结束标签
        dataset_id_or_path (str): 数据集ID或路径
        
    Returns:
        str: 自定义掩码分隔符
        
    Raises:
        NotImplementedError: 当mask_teacher_answer为False时抛出
    """
    custom_mask_delimiter = None
    if mask_teacher_answer:
        # 从配置中获取系统提示词相关信息
        _, _solution_tag, explanation_tag, _, delimiter, paddings = (
            TEACHER_SYSTEM_PROMPTS_AND_TAGS[system_prompt_style])
        
        # 如果指定了数据集路径，使用数据集特定的填充配置
        if dataset_id_or_path is not None:
            data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
            paddings = data.delimiters_padding
            
        # 获取解释标签的开始标记
        start_expl_tag, _ = get_tags(tag=explanation_tag)
        custom_mask_delimiter = start_expl_tag + paddings
        
        # 如果需要包含解决方案结束标签
        if include_end_of_solution_tag:
            _, end_soln_tag = get_tags(tag=_solution_tag)
            custom_mask_delimiter = (
                end_soln_tag + delimiter + custom_mask_delimiter)
    else:
        raise NotImplementedError
    return custom_mask_delimiter


def get_process_line_fn(
        dataset_id_or_path,
        system_prompt_style='stratos',
        return_student_prompt_info: bool = True,
        add_text_completions: bool = False):
    """
    获取数据行处理函数，用于教师强化学习和监督微调的数据格式化
    
    Args:
        dataset_id_or_path (str): 数据集ID或路径
        system_prompt_style (str): 系统提示词风格，默认为'stratos'
        return_student_prompt_info (bool): 是否返回学生提示信息
        add_text_completions (bool): 是否添加文本补全
        
    Returns:
        function: 用于处理单行数据的函数
    """
    # 打印数据格式创建信息
    print(f'Creating data for format {dataset_id_or_path}')
    
    # 获取数据集配置
    data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
    
    # 从教师系统提示词配置中获取相关参数
    (system_msg, new_solution_tag, explanation_tag, new_pre, new_delimiter,
     new_paddings) = (TEACHER_SYSTEM_PROMPTS_AND_TAGS[system_prompt_style])
    
    # 使用数据集特定的分隔符填充
    new_paddings = data.delimiters_padding

    def process_line_fn(line, tokenizer):
        """
        处理单行数据的内部函数
        
        Args:
            line (dict): 单行数据
            tokenizer: 分词器对象
            
        Returns:
            dict: 处理后的数据行，包含格式化的文本和相关信息
        """
        # 构建消息列表，首先添加系统消息
        messages = [{
            "role": "system",
            "content": system_msg,
        },
        ]
        
        # 从数据行中提取问题内容和思考过程及解决方案
        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        
        # 添加用户消息（问题）
        messages.append(
            {"role": 'user', "content": question_content},)
        
        # 提取推理消息的各个组成部分
        (think_prefix, think_content, think_solution_delimiter,
            solution_content) = data.extract_reasoning_message_content(
            s=thought_process_and_solution)
        
        # 处理助手消息，将解决方案和思考内容用新标签包装
        solution, explanation = process_assistant_message(
            solution_content=solution_content,
            think_content=think_content,
            new_solution_tag=new_solution_tag,
            explanation_tag=explanation_tag,
        )
        
        # 构建新的助手回复内容
        new_content = new_pre + solution + new_delimiter + explanation
        messages.append(
            {"role": 'assistant', "content": new_content},)

        # 使用分词器应用聊天模板，生成最终的文本格式
        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        
        # 创建输出行，包含格式化的文本
        out_line = {'text': line_text}
        
        # 如果需要添加文本补全，必须同时返回学生提示信息
        if add_text_completions:
            assert return_student_prompt_info
            
        # 如果需要返回学生提示信息（用于强化学习）
        if return_student_prompt_info:
            # 获取解释标签的开始和结束标记
            start_explanation_tag, end_explanation_tag = get_tags(
                tag=explanation_tag)
            # 获取解决方案标签的开始和结束标记
            start_solution_tag, end_solution_tag = get_tags(
                tag=new_solution_tag)
            
            # 构建完成分隔符，用于分割提示和解决方案
            completion_div = (
                end_solution_tag + new_delimiter + start_explanation_tag +
                new_paddings)

            # 使用分隔符分割文本，得到强化学习的提示和解决方案
            rl_prompts, rl_solutions = line_text.split(completion_div)
            
            # 获取学生数据的标签信息
            (start_think_student_tag, end_think_student_tag,
             start_solution_student_tag, end_solution_student_tag
             ) = data.get_tags()

            # 用学生解决方案标签包装解决方案内容
            solution_s = wrap_string_between_tag(
                s=solution_content, tag=data.solution_tag)

            # 构建强化学习输出行，包含奖励函数所需的所有信息
            rl_out_line = {
                "prompt": rl_prompts + completion_div,                    # 强化学习提示
                "student_system_prompts": data.system_prompt,            # 学生系统提示词
                "start_think_teacher_tags": start_explanation_tag,       # 教师思考开始标签
                "end_think_teacher_tags": end_explanation_tag,           # 教师思考结束标签
                "start_think_student_tags": start_think_student_tag,     # 学生思考开始标签
                "end_think_student_tags": end_think_student_tag,         # 学生思考结束标签
                "start_solution_tags": start_solution_student_tag,       # 解决方案开始标签
                "end_solution_tags": end_solution_student_tag,           # 解决方案结束标签
                "think_prefixes": think_prefix,                          # 思考前缀
                "think_solution_delimiters": think_solution_delimiter,   # 思考解决方案分隔符
                "questions": question_content,                           # 问题内容
                "solutions": solution_s,                                 # 包装后的解决方案
                "solutions_content": solution_content,                   # 原始解决方案内容
                "delimiters_padding": new_paddings,                      # 分隔符填充
                "masked_out_think_prefix": start_explanation_tag + new_paddings,  # 掩码思考前缀
            }
            
            # 如果需要添加文本补全
            if add_text_completions:
                rl_out_line['completion'] = rl_solutions
                
            # 将强化学习相关信息添加到输出行
            out_line.update(rl_out_line)
        return out_line
    return process_line_fn

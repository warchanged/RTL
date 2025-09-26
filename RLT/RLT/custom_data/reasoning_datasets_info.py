from dataclasses import dataclass
from typing import Optional
import re
from collections import defaultdict


def extract_until(s1, s2):
    """
    从字符串s1中提取直到找到s2为止的部分
    
    Args:
        s1 (str): 源字符串
        s2 (str): 要查找的子字符串
        
    Returns:
        str: 从s1开头到s2之前的部分，如果未找到s2则返回完整的s1
    """
    idx = s1.find(s2)
    return s1[:idx] if idx != -1 else s1


def get_tags(tag, format='bespoke'):
    """
    根据指定格式生成标签的开始和结束标记
    
    Args:
        tag (str): 标签名称
        format (str): 标签格式，支持'bespoke'和'r1'两种格式
        
    Returns:
        tuple: (开始标记, 结束标记)
        
    Raises:
        NotImplementedError: 当format不是支持的格式时抛出
    """
    if format == 'bespoke':
        start = f"<|begin_of_{tag}|>"
        end = f"<|end_of_{tag}|>"
    elif format == 'r1':
        start = f"<{tag}>"
        end = f"</{tag}>"
    else:
        raise NotImplementedError
    return start, end


def grab_text_between(s, start, end):
    """
    从字符串中提取位于开始标记和结束标记之间的文本
    
    Args:
        s (str): 源字符串
        start (str): 开始标记
        end (str): 结束标记
        
    Returns:
        str: 位于开始和结束标记之间的文本
        
    Raises:
        ValueError: 当无法找到匹配的标记时抛出
    """
    # 使用正则表达式匹配开始和结束标记之间的内容
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)

    matches = re.findall(pattern, s, re.DOTALL)
    if matches is None:
        print(f'Unable to find content in {start} {end}')
        raise ValueError
    return matches[0]


def grab_text_between_tag(s, tag, tag2=None, format='bespoke'):
    """
    从字符串中提取指定标签之间的文本内容
    
    Args:
        s (str): 源字符串
        tag (str): 主标签名称
        tag2 (str, optional): 第二个标签名称，如果提供则提取从tag结束到tag2开始之间的内容
        format (str): 标签格式，默认为'bespoke'
        
    Returns:
        str: 提取的文本内容
    """
    start, end = get_tags(tag, format=format)

    # 如果提供了第二个标签，则提取两个标签之间的内容
    if tag2 is not None:
        start = end
        end, _ = get_tags(tag2, format=format)
    return grab_text_between(s, start, end)


def wrap_string_between_tag(s: str, tag: str, format='bespoke'):
    """
    用指定标签包装字符串，如果字符串已经被标签包装则直接返回
    
    Args:
        s (str): 要包装的字符串
        tag (str): 标签名称
        format (str): 标签格式，默认为'bespoke'
        
    Returns:
        str: 被标签包装的字符串
    """
    start, end = get_tags(tag=tag, format=format)
    # 如果字符串已经以开始标签开头，确保它也以结束标签结尾，然后直接返回
    if s.startswith(start):
        assert s.endswith(end)
        return s
    # 否则用标签包装字符串
    return f"{start}{s}{end}"


@dataclass
class ReasoningData:
    """
    推理数据配置类，用于定义推理数据集的格式和处理方式
    
    Attributes:
        system_prompt (str): 系统提示词
        think_tag (str): 思考过程标签名称
        solution_tag (str): 解决方案标签名称
        think_prefix (Optional[str]): 思考过程前缀
        think_solution_delimiter (Optional[str]): 思考过程和解决方案之间的分隔符
        delimiters_padding (Optional[str]): 分隔符填充
        tag_format (str): 标签格式，默认为'bespoke'
    """
    system_prompt: str
    think_tag: str
    solution_tag: str

    think_prefix: Optional[str]
    think_solution_delimiter: Optional[str]
    delimiters_padding: Optional[str]
    tag_format: str = 'bespoke'

    def extract_question_and_completion_from_line(self, line):
        """
        从数据行中提取问题和完整回答
        
        Args:
            line (dict): 包含对话数据的字典
            
        Returns:
            tuple: (问题内容, 思考过程和解决方案)
        """
        # 提取用户消息（问题）
        user_message = line["conversations"][0]
        assert user_message['from'] == 'user'
        question_content = user_message["value"]
        
        # 提取助手消息（回答）
        assistant_message = line["conversations"][1]
        assert assistant_message['from'] == 'assistant'
        thought_process_and_solution = assistant_message["value"]
        return question_content, thought_process_and_solution

    def get_tags(self,):
        """
        获取思考和解决方案标签的开始和结束标记
        
        Returns:
            tuple: (思考开始标记, 思考结束标记, 解决方案开始标记, 解决方案结束标记)
        """
        start_think_tag, end_think_tag = get_tags(
            tag=self.think_tag, format=self.tag_format)
        start_solution_tag, end_solution_tag = get_tags(
            tag=self.solution_tag, format=self.tag_format)
        return (start_think_tag, end_think_tag,
                start_solution_tag, end_solution_tag)

    def format_reasoning_message(
            self,
            think_content,
            solution_content,
            think_prefix=None,
            think_solution_delimiter=None,
    ):
        """
        格式化推理消息，将思考内容和解决方案内容组合成完整的推理消息
        
        Args:
            think_content (str): 思考过程内容
            solution_content (str): 解决方案内容
            think_prefix (str, optional): 思考过程前缀，如果为None则使用实例配置
            think_solution_delimiter (str, optional): 思考和解决方案之间的分隔符
            
        Returns:
            tuple: (思考前缀, 思考字符串, 分隔符, 解决方案字符串)
        """
        # 使用实例配置的前缀，如果没有则使用传入的参数或空字符串
        if self.think_prefix is not None:
            think_prefix = self.think_prefix
        if think_prefix is None:
            think_prefix = ''
            
        # 使用实例配置的分隔符，如果没有则使用传入的参数或默认分隔符
        if self.think_solution_delimiter is not None:
            think_solution_delimiter = self.think_solution_delimiter
        if think_solution_delimiter is None:
            think_solution_delimiter = '\n\n'
            
        # 用标签包装思考内容和解决方案内容
        think_s = wrap_string_between_tag(
            s=think_content, tag=self.think_tag, format=self.tag_format)

        solution_s = wrap_string_between_tag(
            s=solution_content, tag=self.solution_tag, format=self.tag_format)
        return think_prefix, think_s, think_solution_delimiter, solution_s

    def extract_reasoning_message_content(self, s):
        """
        从推理消息中提取各个组成部分的内容
        
        Args:
            s (str): 完整的推理消息字符串
            
        Returns:
            tuple: (思考前缀, 思考内容, 分隔符, 解决方案内容)
        """
        # 获取思考前缀
        if self.think_prefix is not None:
            think_prefix = self.think_prefix
        else:
            think_prefix = ''
            
        # 获取思考和解决方案之间的分隔符
        if self.think_solution_delimiter is not None:
            think_solution_delimiter = self.think_solution_delimiter
        else:
            # 如果没有配置分隔符，则从字符串中提取思考标签结束到解决方案标签开始之间的内容
            think_solution_delimiter = grab_text_between_tag(
                s=s, tag=self.think_tag, tag2=self.solution_tag,
                format=self.tag_format)

        # 提取解决方案内容
        solution_content = grab_text_between_tag(
            s=s, tag=self.solution_tag, format=self.tag_format)
        # 提取思考内容
        think_content = grab_text_between_tag(
            s=s, tag=self.think_tag, format=self.tag_format)
        return (think_prefix, think_content, think_solution_delimiter,
                solution_content)


# 不包含答案的查询模板
S1_QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()


@dataclass
class CustomReasoningData(ReasoningData):
    """
    自定义推理数据配置类，继承自ReasoningData
    用于处理自定义格式的推理数据集
    """
    def extract_question_and_completion_from_line(self, line):
        """
        从自定义格式的数据行中提取问题和完整回答
        
        Args:
            line (dict): 包含自定义格式数据的字典，应包含'prompt'、'solution'和可选的'reasoning_trace'字段
            
        Returns:
            tuple: (问题内容, 思考过程和解决方案)
        """
        # 从数据行中提取问题内容
        question_content = line["prompt"]
        # 提取解决方案
        solution = line["solution"]
        # 提取推理轨迹，如果不存在则使用解决方案作为默认值
        thought_process = line.get("reasoning_trace", solution)
        
        # 将思考过程和解决方案分别用对应的标签包装，然后组合
        thought_process_and_solution = wrap_string_between_tag(
            s=thought_process, tag=self.think_tag) + wrap_string_between_tag(
                s=solution, tag=self.solution_tag)
        return question_content, thought_process_and_solution


# Stratos数据集的系统提示词
STRATOS_SYSTEM_PROMPT = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution. "
    "In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
)


# Stratos数据集的配置
STRATOS_CONFIG = ReasoningData(
    system_prompt=STRATOS_SYSTEM_PROMPT,
    think_tag="thought",
    solution_tag="solution",
    think_prefix="",
    think_solution_delimiter="\n\n",
    delimiters_padding="\n\n",
)


# 自定义Stratos风格的配置
CUSTOM_CONFIG_STRATOS_STYLE = CustomReasoningData(
    system_prompt=STRATOS_SYSTEM_PROMPT,
    think_tag="thought",
    solution_tag="solution",
    think_prefix="",
    think_solution_delimiter="\n\n",
    delimiters_padding="\n\n"
)


# 数据配置字典，根据数据集名称返回对应的配置
# 默认使用CUSTOM_CONFIG_STRATOS_STYLE配置
DATA_CONFIGS = defaultdict(
    lambda: CUSTOM_CONFIG_STRATOS_STYLE,
    {
        'bespokelabs/Bespoke-Stratos-17k': STRATOS_CONFIG,
        '/root/autodl-tmp/datasets/Bespoke-Stratos-17k': STRATOS_CONFIG  # 本地路径配置
    }
)

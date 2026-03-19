"""
LLM 生成用户评论
输入：agent persona + 原帖内容
输出：一句话评论（符合人物特征）
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .client import get_llm

_PROMPT = PromptTemplate.from_template(
    """你正在扮演一位社交平台用户，根据人物背景对品牌帖子发表评论。

人物背景：
{persona}

品牌帖子内容：
{post_text}

请以这位用户的口吻写一条评论，要求：
1. 1~2句话，口语化，自然真实
2. 符合人物背景和性格特征
3. 不要提及自己的职业或城市等身份信息
4. 不要使用"作为XXX"这类句式
5. 只输出评论内容本身，不要加引号或任何前缀

评论："""
)

_chain = _PROMPT | get_llm(temperature=0.9) | StrOutputParser()


async def generate_comment(persona: str, post_text: str) -> str:
    result = await _chain.ainvoke({"persona": persona, "post_text": post_text})
    return result.strip()

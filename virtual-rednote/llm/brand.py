"""
品牌 agent 巡查评论并选择性回复
输入：品牌人设 + 原帖 + 新评论列表
输出：[{agent_id, reply}, ...]，最多 3 条，可为空
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .client import get_llm


class BrandReply(BaseModel):
    agent_id: str = Field(description="要回复的 agent_id")
    reply: str = Field(description="回复内容，1~2句，口吻符合品牌风格")


class BrandRepliesOutput(BaseModel):
    replies: list[BrandReply] = Field(
        description="决定回复的列表，最多3条，不想回复则为空列表"
    )


_PROMPT = PromptTemplate.from_template(
    """你是一个品牌的社交媒体运营，负责在社区帖子下回复用户评论。

品牌信息：
{brand_persona}

原帖内容：
{post_text}

以下是用户的新评论，请决定回复哪些（最多3条），并写出回复内容：
{comments_text}

回复原则：
1. 优先回复有疑问、有购买意向或影响力较高（KOL/KOC）的评论
2. 回复语气符合品牌风格，自然真实，不要过于推销
3. 如果没有值得回复的评论，返回空列表
4. 不要重复回复同一个人

{format_instructions}
"""
)


async def brand_review_comments(
    brand_persona: str,
    post_text: str,
    new_comments: list[dict],  # [{agent_id, tier, text}, ...]
) -> list[dict]:
    """返回 [{agent_id, reply}, ...]"""
    if not new_comments:
        return []

    parser = PydanticOutputParser(pydantic_object=BrandRepliesOutput)
    chain = _PROMPT | get_llm(temperature=0.7) | parser

    comments_text = "\n".join(
        f"- [{c['tier'].upper()}] {c['agent_id']}: {c['text']}" for c in new_comments
    )

    try:
        result = await chain.ainvoke(
            {
                "brand_persona": brand_persona,
                "post_text": post_text,
                "comments_text": comments_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return [{"agent_id": r.agent_id, "reply": r.reply} for r in result.replies[:3]]
    except Exception:
        return []

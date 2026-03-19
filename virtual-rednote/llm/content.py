"""
内容向量投射：分析品牌文案，输出在维度空间中的坐标
"""

from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sim.models import Dimension

from .client import get_llm


class ContentVector(BaseModel):
    scores: list[float] = Field(description="每个维度的得分，0~1，顺序与输入维度一致")


_PROMPT = """你是一个内容分析专家，擅长分析营销内容对不同用户群体的吸引力。

社区维度定义：
{dimensions}

品牌内容：
{content}

请分析这条内容在各维度上的得分（0~1）：
- 得分接近 1 表示该内容对"高分端"用户有强吸引力
- 得分接近 0 表示该内容对"低分端"用户更有吸引力

按维度顺序输出 {count} 个得分。

{format_instructions}
"""


async def analyze_content_vector(
    content_text: str, dimensions: list[Dimension]
) -> list[float]:
    parser = PydanticOutputParser(pydantic_object=ContentVector)
    fixing_parser = OutputFixingParser.from_llm(
        parser=parser, llm=get_llm(temperature=0)
    )
    llm = get_llm(temperature=0.3)

    dim_text = "\n".join(
        f"{i+1}. {d.name}：{d.description}\n   低分端：{d.low_label}\n   高分端：{d.high_label}"
        for i, d in enumerate(dimensions)
    )
    response = await llm.ainvoke(
        _PROMPT.format(
            dimensions=dim_text,
            content=content_text,
            count=len(dimensions),
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        result = parser.parse(response.content)
    except Exception:
        result = fixing_parser.parse(response.content)

    scores = result.scores[: len(dimensions)]
    while len(scores) < len(dimensions):
        scores.append(0.5)
    return [max(0.0, min(1.0, s)) for s in scores]

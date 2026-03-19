"""
LLM 推导社区维度
输入：品牌/场景描述
输出：社区名称 + 5~6 个数值维度 + 3~4 个人口统计维度（含标签池）
"""

from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sim.models import BrandAgent, DemographicDimension, Dimension

from .client import get_llm


class DimensionItem(BaseModel):
    name: str = Field(description="维度名称，2~6个字，中文")
    description: str = Field(description="一句话描述该维度的含义")
    low_label: str = Field(description="低分端（0分）对应的用户特征，10~20字")
    high_label: str = Field(description="高分端（1分）对应的用户特征，10~20字")


class DemographicItem(BaseModel):
    name: str = Field(description="人口统计维度名称，例如：职业类型、年龄段、城市线级")
    labels: list[str] = Field(
        description="该维度的标签池，4~8个标签，覆盖该社区的典型用户群体"
    )


class BrandAgentItem(BaseModel):
    brand_name: str = Field(description="品牌名称，从描述中提取")
    tone_of_voice: str = Field(
        description="品牌沟通风格，例如：专业温和、活泼亲切、高冷克制，20字以内"
    )
    reply_style: str = Field(
        description="回复评论时的具体风格描述，30~50字，包含语气、侧重点等"
    )


class WorldOutput(BaseModel):
    community_name: str = Field(description="社区名称，简短，例如：成分党护肤社区")
    dimensions: list[DimensionItem] = Field(description="5~6个正交数值维度")
    demographic_dimensions: list[DemographicItem] = Field(
        description="3~4个人口统计维度"
    )
    brand_agent: BrandAgentItem = Field(description="品牌运营人设")


_PROMPT = """你是一个用户研究和社区分析专家。

品牌/场景描述：
{description}

请为这个场景设计一个虚拟社交社区，输出两类维度：

【数值维度】5~6个正交的用户特征维度（用于向量化表示用户偏好）：
- 维度之间尽量正交，能区分社区内不同类型用户
- 每个维度有明确的低分端和高分端语义

【人口统计维度】3~4个与该场景强相关的人口统计维度：
- 根据产品/场景特点选择最有区分度的维度（如职业、年龄、城市、消费能力等）
- 每个维度提供4~8个标签，覆盖该社区的典型用户群体
- 标签要具体，不要太宽泛（例如职业："护肤博主/皮肤科医生/美妆学生/上班族/全职妈妈"而非"学生/工作者"）

【品牌运营人设】根据产品描述提炼品牌名称、沟通风格和回复评论的具体方式。

同时给出社区名称。

{format_instructions}
"""


async def generate_dimensions(
    description: str,
) -> tuple[str, list[Dimension], list[DemographicDimension], BrandAgent]:
    """返回 (community_name, dimensions, demographic_dimensions)"""
    parser = PydanticOutputParser(pydantic_object=WorldOutput)
    fixing_parser = OutputFixingParser.from_llm(
        parser=parser, llm=get_llm(temperature=0)
    )
    llm = get_llm(temperature=0.7)

    response = await llm.ainvoke(
        _PROMPT.format(
            description=description,
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        result = parser.parse(response.content)
    except Exception:
        result = fixing_parser.parse(response.content)

    dimensions = [
        Dimension(
            name=item.name,
            description=item.description,
            low_label=item.low_label,
            high_label=item.high_label,
        )
        for item in result.dimensions
    ]
    demographic_dimensions = [
        DemographicDimension(name=item.name, labels=item.labels)
        for item in result.demographic_dimensions
    ]
    brand_agent = BrandAgent(
        brand_name=result.brand_agent.brand_name,
        tone_of_voice=result.brand_agent.tone_of_voice,
        reply_style=result.brand_agent.reply_style,
    )
    return result.community_name, dimensions, demographic_dimensions, brand_agent

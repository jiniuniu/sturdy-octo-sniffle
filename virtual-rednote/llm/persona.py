"""
批量生成 agent persona
从 DemographicDimension 标签池采样人口统计信息，连同数值向量一起喂给 LLM。
"""

import random

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from sim.models import Agent, AgentTier, DemographicDimension, Dimension

from .client import get_llm

TIER_LABEL = {
    AgentTier.KOL: "KOL（领域达人，高粉丝量，内容质量高，是社区意见领袖）",
    AgentTier.KOC: "KOC（活跃用户，爱分享真实体验，有一定圈内影响力）",
    AgentTier.NORMAL: "普通用户（偶尔浏览，较少主动发帖）",
}


class PersonaOutput(BaseModel):
    persona: str = Field(description="人物描述，2~3句话，中文，不包含任何人名")


_PROMPT = PromptTemplate.from_template(
    """你是一个用户研究专家，正在为虚拟社交社区创建用户画像。

产品/场景背景：
{product_desc}

社区用户特征维度：
{dimensions_desc}

请为以下用户生成一段人物描述（2~3句话）：
- 身份：{tier_label}
- 人口统计信息（必须体现）：{demographics}
- 该用户在各维度上的特征：
{vector_desc}

要求：
1. 必须在描述中体现上述人口统计信息
2. 自然融入维度特征，不要直接提"维度"或"分值"
3. 符合{tier_label}的社区定位
4. 不要出现任何人名

{format_instructions}
"""
)


def _sample_demographics(
    demographic_dimensions: list[DemographicDimension],
    rng: random.Random,
) -> str:
    """从每个人口统计维度随机抽一个标签，拼成一行描述"""
    parts = [f"{d.name}：{rng.choice(d.labels)}" for d in demographic_dimensions]
    return "，".join(parts)


def _build_vector_desc(agent: Agent, dimensions: list[Dimension]) -> str:
    lines = []
    for i, dim in enumerate(dimensions):
        val = agent.vector[i] if i < len(agent.vector) else 0.5
        if val >= 0.7:
            label = dim.high_label
        elif val <= 0.3:
            label = dim.low_label
        else:
            label = f"介于两端：{dim.low_label} ↔ {dim.high_label}"
        lines.append(f"  - {dim.name}：{label}")
    return "\n".join(lines)


def _build_dimensions_desc(dimensions: list[Dimension]) -> str:
    return "\n".join(
        f"  - {d.name}：{d.description}（低：{d.low_label} / 高：{d.high_label}）"
        for d in dimensions
    )


async def generate_personas(
    agents: list[Agent],
    dimensions: list[Dimension],
    demographic_dimensions: list[DemographicDimension],
    product_desc: str,
    batch_size: int = 10,
    seed: int = 42,
) -> list[Agent]:
    """批量为 agents 生成 persona，原地修改并返回"""
    rng = random.Random(seed)
    parser = PydanticOutputParser(pydantic_object=PersonaOutput)
    chain = _PROMPT | get_llm(temperature=0.9) | parser

    dimensions_desc = _build_dimensions_desc(dimensions)

    inputs = [
        {
            "product_desc": product_desc,
            "dimensions_desc": dimensions_desc,
            "tier_label": TIER_LABEL[agent.tier],
            "vector_desc": _build_vector_desc(agent, dimensions),
            "demographics": _sample_demographics(demographic_dimensions, rng),
            "format_instructions": parser.get_format_instructions(),
        }
        for agent in agents
    ]

    for i in range(0, len(inputs), batch_size):
        batch_agents = agents[i : i + batch_size]
        results = await chain.abatch(
            inputs[i : i + batch_size],
            config={"max_concurrency": batch_size},
        )
        for agent, result in zip(batch_agents, results):
            agent.persona = result.persona
        print(f"  persona 生成进度：{min(i + batch_size, len(agents))}/{len(agents)}")

    return agents

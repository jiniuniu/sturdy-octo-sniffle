import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from chains.client import get_llm
from models.summary import SummaryOutput

_PROMPT = """你是一名资深品牌风险顾问，正在为品牌决策者撰写营销活动风险评估报告的执行摘要。

## 活动信息
{campaign_description}

## 风险维度
{dimensions_text}

## 各 Persona 评论与风险信号
{comments_text}

## 任务
基于以上所有信息，从品牌决策者的视角，生成一份简洁有力的风险总结。要求：

1. **整体风险等级**：综合所有 persona 的升级风险和传播风险，给出整体判断（高/中/低）
2. **核心风险摘要**：用 2-4 句话说清楚这个活动最大的风险是什么、为什么、会怎么演变
3. **最值得关注的风险点**：提炼 2-3 个具体风险，每个需要说明是什么人、会做什么、为什么危险
4. **建议动作**：针对上述风险，给出 3-5 条可执行的建议

语气专业但直接，避免模糊措辞，决策者需要在 30 秒内看懂核心风险。

{format_instructions}
"""


def _build_dimensions_text(dimensions: list[dict]) -> str:
    lines = []
    for dim in dimensions:
        segs = "、".join(s["label"] for s in dim["segments"])
        lines.append(f"· {dim['name']}：{segs}")
    return "\n".join(lines)


def _build_comments_text(comments: list[dict], personas_map: dict) -> str:
    lines = []
    for c in comments:
        persona = personas_map.get(c["persona_id"], {})
        identity = persona.get("identity_summary", c["persona_id"])
        lines.append(
            f"[{c['persona_id']}] {identity}\n"
            f"  平台：{c['platform']} | 情绪：{c['tone']} | "
            f"升级风险：{c['escalation_risk']} | 传播风险：{c['spread_likelihood']}\n"
            f"  评论：{c['text'][:100]}{'...' if len(c['text']) > 100 else ''}\n"
            f"  触发词：{', '.join(c['trigger_keywords'])}\n"
            f"  传播原因：{c['spread_reason']}"
        )
    return "\n\n".join(lines)


async def generate_summary(
    campaign_description: str,
    dimensions: list[dict],
    comments: list[dict],
    personas_map: dict,
) -> SummaryOutput:
    parser = PydanticOutputParser(pydantic_object=SummaryOutput)
    llm = get_llm(temperature=0.3)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=get_llm(temperature=0))

    response = await llm.ainvoke(
        _PROMPT.format(
            campaign_description=campaign_description,
            dimensions_text=_build_dimensions_text(dimensions),
            comments_text=_build_comments_text(comments, personas_map),
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        return parser.parse(response.content)
    except Exception:
        return fixing_parser.parse(response.content)

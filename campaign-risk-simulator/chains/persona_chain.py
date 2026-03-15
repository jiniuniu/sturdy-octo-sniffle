from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from chains.client import get_llm
from models.persona import PersonaOutput

_PROMPT = """你正在为消费者研究生成一个合成消费者 persona。
目标市场为：{target_market}，请确保 persona 的地域、文化背景与该市场一致。

## 研究背景
{stimulus_description}

## 该 Persona 的维度定位
以下是该 persona 在各差异维度上的位置，每条描述了一种特定的消费者类型。
请将这些描述作为硬性约束——persona 的背景、价值观和人生经历必须自然地解释为何他/她持有这些立场。

{segments_text}

## 生成要求
生成一个与上述维度描述内部一致的、真实可信的消费者画像。
persona 的背景和生活处境应该自然解释他/她为何持有这些立场，而不是与之矛盾。
人可以持有看似矛盾的观点，请在复杂性中发掘真实感。

包含以下内容：
- 基本人口信息：年龄、性别、城市、职业、收入范围
- 生活处境：居住状况、家庭结构、日常生活
- 价值观与世界观：2-3 条影响其消费行为的核心信念
- 与品牌的关系：他/她如何对待营销内容
- 心理触发点：这个研究对象中具体什么内容会激发他/她的强烈反应
- 身份摘要：一行话（年龄、职业、地点、一个定义性特质）
- 初始反应提示：看到研究对象时第一感受的一句话描述

Persona ID 为：{persona_id}

{format_instructions}
"""


async def generate_persona(
    campaign_description: str,
    segments_text: str,
    persona_id: str,
    target_market: str = "中国大陆",
) -> PersonaOutput:
    parser = PydanticOutputParser(pydantic_object=PersonaOutput)
    llm = get_llm(temperature=0.9)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=get_llm(temperature=0))

    response = await llm.ainvoke(
        _PROMPT.format(
            stimulus_description=campaign_description,
            segments_text=segments_text,
            persona_id=persona_id,
            target_market=target_market,
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        return parser.parse(response.content)
    except Exception:
        return fixing_parser.parse(response.content)

import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from chains.client import get_llm
from models.comment import CommentOutput

_PROMPT = """你正在模拟一位真实消费者在中国社交媒体上对品牌营销活动的反应。

## 活动内容
{campaign_description}

## Persona 画像
{persona_json}

## 适当性逻辑
在写评论之前，请从该 persona 的视角依次思考以下三个问题：

1. "这是什么情况？"
   → 这个 persona 认为这个活动想要传达什么？

2. "我是什么样的人？"
   → 结合该 persona 的价值观和心理触发点，这个活动让他/她处于怎样的情绪状态？

3. "像我这样的人在这种情况下会怎么做？"
   → 他/她会划走、留短评、写长文、转发，还是引发争论？

用中文写评论，符合该 persona 的语气、平台风格和情绪状态。

{format_instructions}
"""


async def generate_comment(
    campaign_description: str,
    persona: dict,
) -> CommentOutput:
    parser = PydanticOutputParser(pydantic_object=CommentOutput)
    llm = get_llm(temperature=0.9)

    persona_for_llm = {
        k: v for k, v in persona.items()
        if k not in ("_id", "campaign_id", "dimension_scores", "dimension_segments", "completed")
    }

    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=get_llm(temperature=0))

    response = await llm.ainvoke(
        _PROMPT.format(
            campaign_description=campaign_description,
            persona_json=json.dumps(persona_for_llm, ensure_ascii=False, indent=2),
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        return parser.parse(response.content)
    except Exception:
        return fixing_parser.parse(response.content)

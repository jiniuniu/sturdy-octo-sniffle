import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from chains.client import get_llm
from models.response import ResponseOutput
from models.study import ResponseMode

# ── Prompt 模板（按 response_mode）────────────────────────────

_PROMPT_COMMENT = """你正在模拟一位真实消费者在社交媒体上对以下内容的反应。

## 研究对象
{stimulus_description}

## Persona 画像
{persona_json}

## 推理步骤
在生成反应之前，请从该 persona 的视角依次思考：
1. "这是什么情况？" → 这个 persona 认为这个内容想传达什么？
2. "我是什么样的人？" → 结合其价值观和心理触发点，处于怎样的情绪状态？
3. "像我这样的人会怎么做？" → 划走、留短评、写长文，还是引发争论？

用中文写评论，符合该 persona 的语气和平台风格。
content 字段请填写：platform（平台名）、text（评论内容）、tone（正面/负面/中立/复杂）
signals 请填写 spread_risk（传播风险）和 escalation_risk（升级风险），level 为 低/中/高。

{format_instructions}
"""

_PROMPT_SURVEY = """你正在模拟一位真实消费者回答一份关于以下内容的问卷。

## 研究对象
{stimulus_description}

## 研究目标
{research_objective}

## Persona 画像
{persona_json}

## 任务
以该 persona 的视角，对以下问题作答：
1. 你对这个内容的整体评价（1-5分）是多少？为什么？
2. 最吸引你的地方是什么？
3. 你最大的顾虑或不满是什么？
4. 这个内容是否符合你的期望？

content 字段请填写：score（整体评分，如"4/5"）、reason（评分理由）、appeal（吸引点）、concern（顾虑）
signals 请填写 acceptance（接受度）和 purchase_barrier（购买/采纳阻力），level 为 低/中/高。

{format_instructions}
"""

_PROMPT_INTERVIEW = """你正在模拟一位真实消费者接受深度访谈，话题是以下内容。

## 研究对象
{stimulus_description}

## 研究目标
{research_objective}

## Persona 画像
{persona_json}

## 访谈问题
1. 当你第一次看到/了解这个的时候，你的第一反应是什么？
2. 这让你联想到了什么？
3. 这和你目前的生活/需求有什么关系？
4. 你觉得它最大的问题是什么？
5. 如果你要向朋友介绍它，你会怎么说？

请给出自然、口语化、符合该 persona 身份和语言习惯的完整访谈回答。
content 字段请填写：answer（完整访谈回答，可分段）
signals 请填写 engagement（参与意愿）和 confusion_point（认知障碍），level 为 低/中/高。

{format_instructions}
"""

_PROMPT_PURCHASE_INTENT = """你正在模拟一位真实消费者对以下内容做出购买/采纳决策。

## 研究对象
{stimulus_description}

## 研究目标
{research_objective}

## Persona 画像
{persona_json}

## 任务
以该 persona 的视角，描述你的决策过程：
1. 你的第一反应是什么？
2. 你会认真考虑购买/使用吗？为什么？
3. 如果会，你的决策路径是什么（什么情况下会下单）？
4. 如果不会，最大的阻力是什么？能被什么说服？

content 字段请填写：will_buy（是/否/考虑中）、path（决策路径或下单条件）、barrier（最大阻力）、persuasion（什么能改变你的想法）
signals 请填写 purchase_intent（购买意愿）和 price_sensitivity（价格敏感度），level 为 低/中/高。

{format_instructions}
"""

_PROMPT_REACTION = """你正在模拟一位真实消费者看到以下内容时的即时情绪反应。

## 研究对象
{stimulus_description}

## Persona 画像
{persona_json}

## 任务
以该 persona 的视角，描述你在看到这个内容的第一秒内的直觉感受：
- 第一眼的情绪是什么？（惊喜/困惑/反感/共鸣/无感……）
- 这个情绪来自哪里？是哪个具体元素触发的？
- 你会停下来仔细看，还是直接滑走？

content 字段请填写：emotion（情绪词）、trigger（触发元素）、attention（停留/滑走，及原因）
signals 请填写 attention_capture（注意力捕获）和 emotional_resonance（情感共鸣），level 为 低/中/高。

{format_instructions}
"""

_PROMPTS = {
    ResponseMode.comment:         _PROMPT_COMMENT,
    ResponseMode.survey:          _PROMPT_SURVEY,
    ResponseMode.interview:       _PROMPT_INTERVIEW,
    ResponseMode.purchase_intent: _PROMPT_PURCHASE_INTENT,
    ResponseMode.reaction:        _PROMPT_REACTION,
}


async def generate_response(
    stimulus_description: str,
    persona: dict,
    response_mode: ResponseMode = ResponseMode.comment,
    research_objective: str = "",
) -> ResponseOutput:
    parser = PydanticOutputParser(pydantic_object=ResponseOutput)
    llm = get_llm(temperature=0.9)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=get_llm(temperature=0))

    persona_for_llm = {
        k: v for k, v in persona.items()
        if k not in ("_id", "study_id", "campaign_id", "dimension_scores", "dimension_segments", "completed")
    }

    prompt_template = _PROMPTS.get(response_mode, _PROMPT_COMMENT)
    prompt = prompt_template.format(
        stimulus_description=stimulus_description,
        research_objective=research_objective,
        persona_json=json.dumps(persona_for_llm, ensure_ascii=False, indent=2),
        format_instructions=parser.get_format_instructions(),
    )

    response = await llm.ainvoke(prompt)
    try:
        return parser.parse(response.content)
    except Exception:
        return fixing_parser.parse(response.content)

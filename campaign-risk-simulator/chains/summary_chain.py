import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from chains.client import get_llm
from models.summary import StudySummary, SummaryOutput
from models.study import AnalysisFramework

# ── 各分析框架的 prompt 模板 ──────────────────────────────────

_FRAMEWORK_INSTRUCTIONS = {
    AnalysisFramework.risk: """
你是一名资深品牌风险顾问，正在为决策者撰写风险评估执行摘要。
请从以下角度生成 findings：
- "整体风险等级"（高/中/低）及判断依据
- "核心风险"：最大的风险是什么、为什么、会怎么演变
- "高危人群"：哪类消费者最危险，他们会做什么
- "触发因素"：哪些具体元素是风险触发点
语气专业直接，决策者需在 30 秒内看懂核心风险。
""",
    AnalysisFramework.acceptance: """
你是一名消费者洞察分析师，正在评估消费者对研究对象的接受程度。
请从以下角度生成 findings：
- "整体接受度"：高/中/低，及分布估算
- "主要驱动因素"：哪些因素促使消费者接受
- "主要阻力"：哪些因素阻碍接受
- "价格/价值感知"：消费者认为值不值
- "意见领袖效应"：哪类人群的接受会影响其他人
""",
    AnalysisFramework.segmentation: """
你是一名市场细分分析师，正在识别对研究对象反应显著不同的消费者群体。
请从以下角度生成 findings：
- "核心细分群体"：2-4个反应差异最大的群体及其特征
- "最支持群体"：谁最可能成为早期采用者/支持者
- "最抵制群体"：谁最可能反对或流失
- "潜力人群"：目前中立但有转化可能的群体
- "群体规模估算"：各群体的大致市场占比
""",
    AnalysisFramework.decision_path: """
你是一名用户行为研究员，正在还原消费者的决策路径。
请从以下角度生成 findings：
- "典型决策路径"：从接触到决策的关键步骤
- "关键决策节点"：在哪个环节消费者会做出去留判断
- "信息需求"：消费者在决策过程中缺少什么信息
- "情绪曲线"：各阶段的主导情绪变化
- "放弃触发点"：什么会导致消费者中途放弃
""",
    AnalysisFramework.fit_assessment: """
你是一名品牌策略顾问，正在评估研究对象与目标消费者的匹配程度。
请从以下角度生成 findings：
- "整体匹配度"：高/中/低，及综合判断
- "价值观契合度"：研究对象传递的价值观是否与目标人群一致
- "认知一致性"：消费者对品牌/产品的已有认知是否与内容匹配
- "期望落差"：内容与消费者期望之间的主要差距
- "错位风险"：哪些元素可能让消费者感到不协调
""",
}

_PROMPT = """## 研究对象
{stimulus_description}

## 研究目标
{research_objective}

## 研究维度
{dimensions_text}

## 各 Persona 反应
{responses_text}

## 分析任务
{framework_instructions}

基于以上所有信息，生成一份完整的研究总结。
- overall_conclusion：一句话概括最重要的发现
- confidence_level：基于 persona 覆盖度和反应一致性，判断结论可信度（高/中/低）
- findings：按上方分析框架要求填写，每条 finding 必须有具体的 evidence 引用
- segment_differences：识别 2-4 个反应显著不同的人群，给出规模估算
- suggested_actions：3-5 条可执行的具体建议
- open_questions：1-3 个本次研究无法回答、需要进一步验证的问题

{format_instructions}
"""


def _build_dimensions_text(dimensions: list[dict]) -> str:
    lines = []
    for dim in dimensions:
        if dim.get("source") == "hidden":
            continue
        segs = "、".join(s["label"] for s in dim["segments"])
        lines.append(f"· {dim['name']}：{segs}")
    return "\n".join(lines)


def _build_responses_text(responses: list[dict], personas_map: dict) -> str:
    lines = []
    for r in responses:
        persona = personas_map.get(r["persona_id"], {})
        identity = persona.get("identity_summary", r["persona_id"])
        signals_str = "、".join(
            f"{s['key']}:{s['level']}" for s in r.get("signals", [])
        )
        content_preview = json.dumps(r.get("content", {}), ensure_ascii=False)[:120]
        lines.append(
            f"[{r['persona_id']}] {identity}\n"
            f"  立场：{r['primary_stance']} | 信号：{signals_str}\n"
            f"  核心观点：{r['key_quote']}\n"
            f"  内容摘要：{content_preview}"
        )
    return "\n\n".join(lines)


async def generate_study_summary(
    stimulus_description: str,
    dimensions: list[dict],
    responses: list[dict],
    personas_map: dict,
    analysis_framework: AnalysisFramework = AnalysisFramework.risk,
    research_objective: str = "",
) -> StudySummary:
    parser = PydanticOutputParser(pydantic_object=StudySummary)
    llm = get_llm(temperature=0.3)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=get_llm(temperature=0))

    framework_instructions = _FRAMEWORK_INSTRUCTIONS.get(
        analysis_framework,
        _FRAMEWORK_INSTRUCTIONS[AnalysisFramework.risk],
    )

    response = await llm.ainvoke(
        _PROMPT.format(
            stimulus_description=stimulus_description,
            research_objective=research_objective,
            dimensions_text=_build_dimensions_text(dimensions),
            responses_text=_build_responses_text(responses, personas_map),
            framework_instructions=framework_instructions,
            format_instructions=parser.get_format_instructions(),
        )
    )
    try:
        return parser.parse(response.content)
    except Exception:
        return fixing_parser.parse(response.content)


# ── 向后兼容：旧 /campaigns 路由使用 ─────────────────────────

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


_LEGACY_PROMPT = """你是一名资深品牌风险顾问，正在为品牌决策者撰写营销活动风险评估报告的执行摘要。

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
        _LEGACY_PROMPT.format(
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

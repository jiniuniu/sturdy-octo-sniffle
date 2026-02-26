# models/schemas.py
from typing import Literal

from pydantic import BaseModel, Field

# ── 系统常量 ───────────────────────────────────────────────────────────────────

PREDEFINED_AXIS_NAMES = [
    "外部时机成熟度",
    "问题真实度（伪需求风险）",
    "用户切换成本",
    "变现可行性",
    "竞争防御能力",
    "网络效应潜力",
]

# 五大业务维度定义（轴索引、业务问题、最相关 TPB 维度）
# 竞争护城河合并了「竞争防御能力」和「网络效应潜力」两个轴
BUSINESS_DIMENSIONS: list[dict] = [
    {
        "name": "验证外部环境",
        "question": "现在进入市场时机成熟吗？",
        "axis_indices": [0],
        "axis_names": ["外部时机成熟度"],
        "key_tpb": "attitude",
    },
    {
        "name": "需求真实性",
        "question": "这是真实痛点还是伪需求？",
        "axis_indices": [1],
        "axis_names": ["问题真实度（伪需求风险）"],
        "key_tpb": "attitude",
    },
    {
        "name": "迁移成本",
        "question": "用户从现有方案切换过来容易吗？",
        "axis_indices": [2],
        "axis_names": ["用户切换成本"],
        "key_tpb": "perceived_control",
    },
    {
        "name": "商业模式",
        "question": "用户愿意为这个价值付费吗？",
        "axis_indices": [3],
        "axis_names": ["变现可行性"],
        "key_tpb": "attitude",
    },
    {
        "name": "竞争护城河",
        "question": "产品有没有持续的竞争优势？",
        "axis_indices": [4, 5],
        "axis_names": ["竞争防御能力", "网络效应潜力"],
        "key_tpb": "subjective_norm",
    },
]


# ── 用户输入 ───────────────────────────────────────────────────────────────────


class ProductInfo(BaseModel):
    name: str = Field(description="产品名称")
    description: str = Field(description="产品自然语言描述，1~3句")
    target_market: str = Field(description="目标市场，如'中国城市钓鱼爱好者，25~60岁'")


# ── Stage 1 输出：具体化维度 + 语义标签 ────────────────────────────────────────
# LLM 根据产品信息，将预设的6个维度名称具体化，并生成 L1~L4 语义标签


class DiversityAxis(BaseModel):
    name: str = Field(description="维度名称，与预设名称保持一致")
    description: str = Field(description="针对本产品具体化的维度说明，1句话")
    labels: list[str] = Field(
        min_length=4,
        max_length=4,
        description="从低到高的4个语义标签 [L1, L2, L3, L4]，具体、口语化",
    )


class AxesOutput(BaseModel):
    axes: list[DiversityAxis] = Field(
        min_length=6,
        max_length=6,
        description="6个多样性维度，顺序与 PREDEFINED_AXIS_NAMES 一致",
    )


# ── Stage 2a 输出：采样向量（纯计算，无 LLM）─────────────────────────────────
# Sobol 均匀采样生成，由 sampler.py 构造，不经过 LLM 解析


class PersonaVector(BaseModel):
    id: str = Field(description="唯一标识，如 'persona_000'")
    axis_indices: list[int] = Field(
        min_length=6,
        max_length=6,
        description="每个维度的 label index，值域 0~3 对应 L1~L4",
    )
    axis_labels: list[str] = Field(
        min_length=6,
        max_length=6,
        description="axis_indices 对应的语义标签，便于调试和 prompt 构造",
    )


# ── Stage 2b 输出：人物描述 + 行为目标 ────────────────────────────────────────
# LLM 根据 PersonaVector 生成自然语言人物描述，同时推导行为目标
# id 和 vector 由 persona_generator.py 通过 model_copy 注入，不由 LLM 输出


class PersonaProfile(BaseModel):
    id: str = Field(
        default="", description="由 pipeline 注入，与 PersonaVector.id 一致"
    )
    vector: PersonaVector = Field(
        default=None,  # type: ignore
        description="由 pipeline 注入，保留采样向量用于后续分析",
    )
    description: str = Field(
        description="2~3句自然语言人物描述，含姓名、年龄、职业、关键行为特征，不提维度或标签"
    )
    behavior_goal: str = Field(
        description="该用户对此产品最核心的行为目标，如'下载App并完成首次鱼塘预约'"
    )


# ── Stage 3a 输出：TPB 问卷 ────────────────────────────────────────────────────
# LLM 根据产品描述和行为目标生成，所有 persona 共用同一套问卷


class QuestionnaireItem(BaseModel):
    id: str = Field(description="题目编号，如 'q01'")
    tpb_dimension: Literal["attitude", "subjective_norm", "perceived_control"] = Field(
        description="所属 TPB 维度"
    )
    question_text: str = Field(description="第一人称、口语化的题目文本，不提产品名")
    reverse_scored: bool = Field(default=False, description="是否反向计分")


class Questionnaire(BaseModel):
    behavior_goal: str = Field(description="本问卷对应的行为目标")
    items: list[QuestionnaireItem] = Field(
        min_length=12,
        max_length=12,
        description="12道题，每个 TPB 维度各4题，每维度至少1道反向计分题",
    )


# ── Stage 3b 输出：问卷回答 ───────────────────────────────────────────────────
# LLM 代入 persona 逐题作答
# persona_id 由 responder.py 通过 model_copy 注入


class SingleResponse(BaseModel):
    score: int = Field(ge=1, le=5, description="1=完全不同意，5=完全同意")
    reasoning: str = Field(description="打分理由，1句话")


class QuestionnaireResponse(BaseModel):
    persona_id: str = Field(default="", description="由 pipeline 注入")
    responses: dict[str, SingleResponse] = Field(
        description="key 为 question id（如 'q01'），value 为打分和理由"
    )


# ── Stage 4a 输出：TPB 分数（纯计算，无 LLM）─────────────────────────────────
# scorer.py 根据 QuestionnaireResponse 计算，不经过 LLM


class TPBScore(BaseModel):
    attitude: float = Field(ge=0.0, le=1.0, description="态度维度均值，标准化到 0~1")
    subjective_norm: float = Field(
        ge=0.0, le=1.0, description="主观规范维度均值，标准化到 0~1"
    )
    perceived_control: float = Field(
        ge=0.0, le=1.0, description="感知行为控制维度均值，标准化到 0~1"
    )
    acceptance_score: float = Field(
        ge=0.0, le=1.0, description="三维度加权聚合，默认权重 0.4/0.3/0.3"
    )


# ── 聚合结构 ──────────────────────────────────────────────────────────────────


class PersonaEvaluation(BaseModel):
    persona: PersonaProfile
    response: QuestionnaireResponse
    tpb_score: TPBScore


class EvaluationMatrix(BaseModel):
    product: ProductInfo
    evaluations: list[PersonaEvaluation] = Field(
        description="N 个 persona 的完整评估结果"
    )


# ── Stage 4b 输出：洞察报告 ───────────────────────────────────────────────────


class DimensionInsight(BaseModel):
    """五大业务维度中单个维度的完整洞察。统计字段由 pipeline 计算注入，文字字段由 LLM 生成。"""

    dimension_name: str = Field(description="业务维度名称，如'需求真实性'")
    business_question: str = Field(description="该维度对应的业务问题")
    axis_names: list[str] = Field(description="对应的原始轴名称（可能多个）")

    # 统计字段（pipeline 计算注入）
    correlation: float = Field(description="轴值与接受度的 Spearman 相关系数，多轴取平均")
    pvalue: float = Field(description="相关系数的 p 值")
    label_means: dict[str, float] = Field(
        description="各标签层级的接受度均值，key 为 'L1'~'L4'"
    )
    key_tpb_barrier: str = Field(
        description="与该维度最相关的 TPB 障碍：attitude / subjective_norm / perceived_control"
    )
    signal: Literal["验证通过", "需关注", "风险较高", "数据噪声"] = Field(
        description="信号强度：由统计量自动计算，不由 LLM 生成"
    )

    # 文字字段（LLM 生成）
    finding: str = Field(description="基于数据的业务发现，2句：第1句描述现象，第2句给出判断")
    recommendation: str = Field(description="下一步最该做的一个具体验证动作，1句话")


class InsightReport(BaseModel):
    # 五大业务维度各一条，由 insight_engine 计算后注入
    dimension_insights: list[DimensionInsight] = Field(
        default_factory=list,
        description="五大业务维度的完整洞察，由 pipeline 注入",
    )

    # 以下字段由 LLM 生成
    ideal_persona_description: str = Field(
        description="高接受度用户的共同特征，自然语言描述"
    )
    dead_zone_description: str = Field(
        description="低接受度用户的共同特征及卡点分析，自然语言描述"
    )
    key_risks: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3~5条系统性风险，区分'价值未被认可'与'用户无法行动'两类",
    )
    summary: str = Field(description="整体洞察摘要，3~5句话")

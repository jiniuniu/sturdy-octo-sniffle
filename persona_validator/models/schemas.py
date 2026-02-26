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
# LLM 根据数值分析结果生成自然语言洞察
# top_correlations 由 insight_engine.py 通过 model_copy 注入


class InsightReport(BaseModel):
    top_correlations: list[dict] = Field(
        default_factory=list,
        description="由 pipeline 注入，相关系数最高的3个维度",
    )
    ideal_persona_description: str = Field(
        description="高接受度用户的共同特征，自然语言描述"
    )
    dead_zone_description: str = Field(
        description="低接受度用户的共同特征，自然语言描述"
    )
    key_risks: list[str] = Field(
        min_length=3,
        max_length=5,
        description="3~5条系统性风险，区分'价值未被认可'与'用户无法行动'两类",
    )
    summary: str = Field(description="整体洞察摘要，3~5句话")

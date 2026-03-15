from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
from enum import Enum


# ── 枚举 ────────────────────────────────────────────────────

class StudyType(str, Enum):
    risk_assessment        = "risk_assessment"        # 营销活动风险评估（原有）
    concept_test           = "concept_test"           # 新品/功能概念测试
    pricing_test           = "pricing_test"           # 定价测试
    creative_test          = "creative_test"          # 广告创意/文案测试
    policy_test            = "policy_test"            # 规则/权益变更测试
    market_entry           = "market_entry"           # 市场进入评估
    brand_fit              = "brand_fit"              # 品牌/代言人匹配度
    user_journey           = "user_journey"           # 用户旅程还原
    competitive_perception = "competitive_perception" # 竞品对比认知


class ResponseMode(str, Enum):
    comment         = "comment"         # 社媒评论模拟（原有）
    survey          = "survey"          # 结构化问卷回答
    interview       = "interview"       # 深度访谈模拟
    purchase_intent = "purchase_intent" # 购买决策模拟
    reaction        = "reaction"        # 即时情绪反应


class AnalysisFramework(str, Enum):
    risk           = "risk"           # 风险识别（原有）
    acceptance     = "acceptance"     # 接受度测量
    segmentation   = "segmentation"   # 人群细分洞察
    decision_path  = "decision_path"  # 决策路径分析
    fit_assessment = "fit_assessment" # 匹配度评估


class StimulusType(str, Enum):
    campaign     = "campaign"     # 营销活动
    product      = "product"      # 产品/功能
    pricing      = "pricing"      # 定价方案
    concept      = "concept"      # 商业概念
    message      = "message"      # 传播内容/文案
    policy       = "policy"       # 规则/政策
    spokesperson = "spokesperson" # 代言人/KOL
    experience   = "experience"   # 用户体验/旅程


class StudyStatus(str, Enum):
    idle                 = "idle"
    processing_visual    = "processing_visual"
    extracting_dimensions = "extracting_dimensions"
    sampling             = "sampling"
    generating_personas  = "generating_personas"
    generating_responses = "generating_responses"
    completed            = "completed"
    error                = "error"


TERMINAL_STATUSES = {StudyStatus.completed, StudyStatus.error}
RUNNING_STATUSES = {
    StudyStatus.processing_visual,
    StudyStatus.extracting_dimensions,
    StudyStatus.sampling,
    StudyStatus.generating_personas,
    StudyStatus.generating_responses,
}


# ── 子模型 ───────────────────────────────────────────────────

class Attachment(BaseModel):
    type: Literal["image", "document", "url"]
    content: str   # base64 / 文本内容 / URL
    label: str     # 如 "方案A"、"竞品对比"


class PresetDimension(BaseModel):
    name: str
    description: str
    segments: list[str] = []  # 可选，不填则 LLM 自动生成


class StudyDesign(BaseModel):
    study_type: StudyType
    research_objective: str            # 自由文本，如"了解25-35岁女性对新定价的接受度"
    response_mode: ResponseMode
    analysis_framework: AnalysisFramework
    structural_dimensions: list[str] = ["region", "occupation"]  # 必须覆盖的人口变量
    preset_dimensions: list[PresetDimension] = []                # 用户预设维度


# ── Request ──────────────────────────────────────────────────

class StudyCreate(BaseModel):
    title: str
    description: str
    stimulus_type: StimulusType = StimulusType.campaign
    context: str = ""                  # 背景信息
    study_design: StudyDesign
    n_personas: int = Field(default=8, ge=4, le=16)
    target_market: str = "中国大陆"
    callback_url: Optional[str] = None


# ── Response ─────────────────────────────────────────────────

class StudyCreated(BaseModel):
    study_id: str
    status: StudyStatus
    created_at: datetime


class PipelineProgress(BaseModel):
    personas_completed: int = 0
    personas_total: int = 0
    responses_completed: int = 0
    responses_total: int = 0


class StudyStatusResponse(BaseModel):
    study_id: str
    status: StudyStatus
    progress: PipelineProgress
    stages_completed: list[str]
    error: Optional[str] = None
    report_url: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ── DB Document ───────────────────────────────────────────────

class StudyDoc(BaseModel):
    title: str
    description: str
    stimulus_type: StimulusType = StimulusType.campaign
    context: str = ""
    study_design: StudyDesign
    n_personas: int = 8
    target_market: str = "中国大陆"
    # 附件（图片等）
    image_key: Optional[str] = None
    image_url: Optional[str] = None
    visual_description: Optional[str] = None
    # 状态
    status: StudyStatus = StudyStatus.idle
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ── 默认配置：向后兼容原 campaign 风险评估 ───────────────────

DEFAULT_RISK_ASSESSMENT_DESIGN = StudyDesign(
    study_type=StudyType.risk_assessment,
    research_objective="识别营销活动可能引发的舆论风险和文化敏感点",
    response_mode=ResponseMode.comment,
    analysis_framework=AnalysisFramework.risk,
    structural_dimensions=["region", "occupation"],
)

from pydantic import BaseModel
from typing import Literal, Optional


# ── 通用弱类型 Summary（所有分析框架共用）────────────────────

class Finding(BaseModel):
    key: str          # 发现维度，如"整体接受度"/"主要风险"/"价格敏感区间"
    value: str        # 结论值，如"中等偏高"/"文化挪用风险"/"150-200元"
    evidence: str     # 支撑证据，引用具体人设或反应
    importance: Literal["高", "中", "低"]


class SegmentDifference(BaseModel):
    segment_description: str   # 哪类人群
    stance: str                # 他们的立场/反应
    size_estimate: str         # 规模估算，如"约占目标市场30%"


class StudySummary(BaseModel):
    overall_conclusion: str                           # 一句话结论
    confidence_level: Literal["高", "中", "低"]
    findings: list[Finding]                           # 核心发现（各框架填充不同语义）
    segment_differences: list[SegmentDifference]      # 人群差异
    suggested_actions: list[str]                      # 建议动作
    open_questions: list[str]                         # 需要进一步研究的问题


# ── 向后兼容：原 SummaryOutput 保留供旧 /campaigns 路由使用 ──

class TopRisk(BaseModel):
    title: str
    description: str
    related_personas: list[str]


class SummaryOutput(BaseModel):
    overall_risk_level: str
    key_risk_summary: str
    top_risks: list[TopRisk]
    suggested_actions: list[str]

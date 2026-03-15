from pydantic import BaseModel
from typing import Literal


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
    overall_conclusion: str
    confidence_level: Literal["高", "中", "低"]
    findings: list[Finding]
    segment_differences: list[SegmentDifference]
    suggested_actions: list[str]
    open_questions: list[str]

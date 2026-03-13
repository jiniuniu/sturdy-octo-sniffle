from pydantic import BaseModel
from typing import Optional


class TopRisk(BaseModel):
    title: str
    description: str
    related_personas: list[str]    # persona_id 列表


class SummaryOutput(BaseModel):
    overall_risk_level: str        # "高" | "中" | "低"
    key_risk_summary: str          # 一段话，给决策者看的
    top_risks: list[TopRisk]       # 2-3 个最值得关注的风险点
    suggested_actions: list[str]   # 建议动作

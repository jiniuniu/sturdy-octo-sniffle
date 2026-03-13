from pydantic import BaseModel
from typing import Literal


class Reasoning(BaseModel):
    situation_reading: str
    emotional_state: str
    action_choice: str


class CommentContent(BaseModel):
    platform: str
    text: str
    tone: Literal["正面", "负面", "中立", "复杂"]
    length_type: str


class RiskSignals(BaseModel):
    spread_likelihood: Literal["低", "中", "高"]
    spread_reason: str
    trigger_keywords: list[str]
    escalation_risk: Literal["低", "中", "高"]


# LLM 结构化输出
class CommentOutput(BaseModel):
    persona_id: str
    reasoning: Reasoning
    comment: CommentContent
    risk_signals: RiskSignals


# DB Document
class CommentDoc(BaseModel):
    campaign_id: str
    persona_id: str
    situation_reading: str
    emotional_state: str
    action_choice: str
    platform: str
    text: str
    tone: str
    length_type: str
    spread_likelihood: str
    spread_reason: str
    trigger_keywords: list[str]
    escalation_risk: str


# 风险汇总（计算得出，不存 DB）
class RiskSummary(BaseModel):
    high_escalation_count: int
    high_spread_count: int
    top_trigger_keywords: list[str]
    tone_distribution: dict[str, int]
    riskiest_persona_id: str

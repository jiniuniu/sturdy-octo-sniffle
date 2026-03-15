from pydantic import BaseModel
from typing import Literal, Optional
from models.study import ResponseMode


# ── LLM 结构化输出（弱类型）────────────────────────────────

class Signal(BaseModel):
    key: str    # 如 "spread_risk"、"purchase_barrier"、"confusion_point"
    level: Literal["低", "中", "高"]
    reason: str


class ResponseOutput(BaseModel):
    persona_id: str
    primary_stance: Literal["正面", "负面", "中立", "复杂"]
    key_quote: str                  # 最能代表其立场的一句话
    reasoning: str                  # 三步推理过程（情境理解→情绪→行动）
    content: dict[str, str]         # 各 mode 填充不同 key：
                                    #   comment  → {"platform": ..., "text": ..., "tone": ...}
                                    #   survey   → {"scores": ..., "reason": ..., "concern": ...}
                                    #   purchase → {"will_buy": ..., "path": ..., "barrier": ...}
                                    #   interview → {"answer": ...}
                                    #   reaction  → {"reaction": ..., "emotion": ...}
    signals: list[Signal]


# ── DB Document ───────────────────────────────────────────────

class ResponseDoc(BaseModel):
    study_id: str
    persona_id: str
    response_mode: str
    primary_stance: str
    key_quote: str
    reasoning: str
    content: dict[str, str]
    signals: list[Signal]

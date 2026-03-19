from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentTier(str, Enum):
    KOL = "kol"
    KOC = "koc"
    NORMAL = "normal"


class ContentType(str, Enum):
    ORIGINAL = "original"
    REPOST = "repost"
    COMMENT = "comment"


class EventType(str, Enum):
    CONTENT_PUBLISHED = "content_published"
    CONTENT_EXPOSED = "content_exposed"


class ActionType(str, Enum):
    LIKE = "like"
    COMMENT = "comment"
    REPOST = "repost"
    IGNORE = "ignore"


@dataclass
class Dimension:
    name: str
    description: str
    low_label: str
    high_label: str


@dataclass
class DemographicDimension:
    name: str  # 例如 "城市线级"、"职业类型"、"年龄段"
    labels: list[str]  # 例如 ["一线城市", "新一线城市", "二线城市"]


@dataclass
class BrandAgent:
    brand_name: str
    tone_of_voice: str  # 例如：专业温和、活泼亲切
    reply_style: str  # 回复评论时的具体风格描述

    def persona(self) -> str:
        return f"品牌：{self.brand_name}\n风格：{self.tone_of_voice}\n回复方式：{self.reply_style}"


@dataclass
class Agent:
    id: str
    world_id: str
    tier: AgentTier
    vector: list[float]  # 在维度空间中的坐标 [0,1]^n
    activity: float  # 活跃度，影响 p_like
    expressiveness: float  # 表达欲，影响 p_comment
    sharing: float  # 分享倾向，影响 p_repost
    persona: str = ""  # LLM 生成的人物描述（后续用于 LLM 评论）
    following: list[str] = field(default_factory=list)
    followers: list[str] = field(default_factory=list)


@dataclass
class Content:
    id: str
    sim_id: str
    author_id: str
    type: ContentType
    vector: list[float]
    text: str = ""
    parent_id: Optional[str] = None
    sim_time: float = 0.0
    likes: int = 0
    comments: int = 0
    reposts: int = 0


@dataclass(order=True)
class Event:
    sim_time: float
    type: EventType = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)

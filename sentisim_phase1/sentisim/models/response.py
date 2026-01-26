"""
用户反应数据模型（LLM 输出）
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """用户行动类型"""

    IGNORE = "ignore"  # 划走，不互动
    LIKE = "like"  # 点赞
    FORWARD = "forward"  # 直接转发
    FORWARD_COMMENT = "forward_comment"  # 转发并评论
    CREATE = "create"  # 发原创内容


class UserResponse(BaseModel):
    """用户反应（LLM 结构化输出）"""

    first_reaction: str = Field(description="看到内容的第一反应，一句话")

    emotion: str = Field(
        description="情绪感受，如：开心、无感、反感、愤怒、担忧、想吐槽等"
    )

    impression_change: str = Field(
        description="对品牌印象的变化，如果没变化就回答'没有变化'"
    )

    action: ActionType = Field(description="决定采取的行动")

    content: Optional[str] = Field(
        default=None, description="如果选择转发评论或发原创，写的内容"
    )

    @property
    def has_impression_change(self) -> bool:
        """印象是否发生变化"""
        if not self.impression_change:
            return False
        no_change_keywords = ["没有变化", "没变化", "无变化", "不变", "unchanged"]
        return not any(
            kw in self.impression_change.lower() for kw in no_change_keywords
        )

    @property
    def will_spread(self) -> bool:
        """是否会产生传播"""
        return self.action in [
            ActionType.FORWARD,
            ActionType.FORWARD_COMMENT,
            ActionType.CREATE,
        ]

    @property
    def is_negative(self) -> bool:
        """是否为负面情绪（简单判断）"""
        negative_keywords = [
            "反感",
            "愤怒",
            "失望",
            "担忧",
            "不满",
            "厌恶",
            "怀疑",
            "质疑",
            "吐槽",
        ]
        return any(kw in self.emotion for kw in negative_keywords)

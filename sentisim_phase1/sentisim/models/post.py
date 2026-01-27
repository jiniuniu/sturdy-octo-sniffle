"""
帖子数据模型
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PostType(str, Enum):
    """帖子类型"""

    BRAND_ORIGINAL = "brand_original"  # 品牌原创
    FORWARD = "forward"  # 直接转发
    FORWARD_COMMENT = "forward_comment"  # 转发并评论


class Post(BaseModel):
    """帖子"""

    post_id: str = Field(description="帖子ID")

    author_id: str = Field(description="作者ID")

    content: str = Field(description="帖子内容")

    post_type: PostType = Field(default=PostType.BRAND_ORIGINAL, description="帖子类型")

    original_post_id: Optional[str] = Field(
        default=None, description="原帖ID（如果是转发）"
    )

    timestamp: int = Field(default=0, description="时间戳（模拟步数）")

    @property
    def is_original(self) -> bool:
        """是否为原创内容（品牌原帖）"""
        return self.post_type == PostType.BRAND_ORIGINAL

    @property
    def is_forward(self) -> bool:
        """是否为转发"""
        return self.post_type in [PostType.FORWARD, PostType.FORWARD_COMMENT]

    def to_str(self) -> str:
        """格式化帖子内容为可读字符串"""
        type_prefix = {
            PostType.BRAND_ORIGINAL: "品牌官方",
            PostType.FORWARD: f"{self.author_id} 转发",
            PostType.FORWARD_COMMENT: f"{self.author_id} 转发并评论",
        }
        prefix = type_prefix.get(self.post_type, self.author_id)
        return f"【{prefix}】{self.content}"

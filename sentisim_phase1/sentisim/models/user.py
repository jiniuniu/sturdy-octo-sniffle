"""
用户数据模型
"""

from enum import Enum

from pydantic import BaseModel, Field


class InfluenceLevel(str, Enum):
    """影响力层级"""

    KOL = "kol"
    ACTIVE = "active"
    NORMAL = "normal"
    SILENT = "silent"


class User(BaseModel):
    """用户"""

    user_id: str = Field(description="用户ID")

    persona_type: str = Field(description="所属人群类型名称")

    profile: str = Field(description="用户画像描述")

    influence_level: InfluenceLevel = Field(
        default=InfluenceLevel.NORMAL,
        description="影响力层级",
    )

    memory: list[str] = Field(default_factory=list, description="记忆列表")

    follower_ids: list[str] = Field(default_factory=list, description="粉丝ID列表")

    def add_memory(self, memory_item: str, max_size: int = 10) -> None:
        """添加记忆，保持长度限制"""
        if memory_item and memory_item.strip():
            self.memory.append(memory_item.strip())
            if len(self.memory) > max_size:
                self.memory = self.memory[-max_size:]

    @property
    def follower_count(self) -> int:
        """粉丝数量"""
        return len(self.follower_ids)


class UserProfile(BaseModel):
    """用户画像（用于 LLM 结构化输出）"""

    profile: str = Field(description="用户画像描述，80-120字")


class BatchUserProfiles(BaseModel):
    """批量用户画像（用于 LLM 结构化输出）- 已废弃，使用 BatchUsersWithMemories"""

    profiles: list[str] = Field(
        description="用户画像列表，每个元素是一个用户的画像描述（80-120字）"
    )


class UserWithMemory(BaseModel):
    """单个用户的画像和初始记忆"""

    profile: str = Field(description="用户画像描述（80-120字）")
    initial_memory: str = Field(description="用户对品牌的初始记忆或印象（一句话）")


class BatchUsersWithMemories(BaseModel):
    """批量用户画像和初始记忆（用于 LLM 结构化输出）"""

    users: list[UserWithMemory] = Field(
        description="用户列表，每个元素包含画像和初始记忆"
    )


class UserMemoryList(BaseModel):
    """用户初始记忆（用于 LLM 结构化输出）"""

    memories: list[str] = Field(
        default_factory=list, description="记忆列表，每条用一句话表达"
    )

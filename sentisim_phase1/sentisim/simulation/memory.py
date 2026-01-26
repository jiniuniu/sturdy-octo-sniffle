"""
记忆更新机制
"""

from config import settings
from sentisim.models import User, UserResponse


class MemoryManager:
    """记忆管理器"""

    def __init__(self, max_memory_size: int = None):
        """
        Args:
            max_memory_size: 最大记忆条数
        """
        self.max_memory_size = max_memory_size or settings.max_memory_size

    def update(self, user: User, response: UserResponse) -> bool:
        """
        根据用户反应更新记忆（原地修改）

        只有当用户的印象发生变化时才记录。
        记录的是印象变化本身，而不是看到的内容。

        Args:
            user: 用户
            response: 用户反应

        Returns:
            是否添加了新记忆
        """
        # 只有印象发生变化时才记录
        if not response.has_impression_change:
            return False

        # 添加印象变化到记忆
        user.add_memory(response.impression_change, self.max_memory_size)
        return True

    def batch_update(self, user_response_pairs: list[tuple[User, UserResponse]]) -> int:
        """
        批量更新记忆

        Args:
            user_response_pairs: [(用户, 反应), ...]

        Returns:
            更新的记忆数量
        """
        updated_count = 0
        for user, response in user_response_pairs:
            if self.update(user, response):
                updated_count += 1
        return updated_count

    def get_memory_stats(self, users: list[User]) -> dict:
        """
        获取记忆统计信息

        Args:
            users: 用户列表

        Returns:
            统计数据
        """
        total_memories = sum(len(u.memory) for u in users)
        users_with_memory = sum(1 for u in users if u.memory)

        return {
            "total_users": len(users),
            "users_with_memory": users_with_memory,
            "total_memories": total_memories,
            "avg_memories_per_user": (
                round(total_memories / len(users), 2) if users else 0
            ),
        }

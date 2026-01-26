"""
用户记忆初始化生成器
"""

from typing import Optional

from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext, User
from sentisim.models.user import UserMemoryList


class MemoryGenerator:
    """用户记忆初始化生成器"""

    def __init__(self, llm: SentiSimLLM):
        self.llm = llm

    async def generate_memory(
        self,
        user: User,
        brand_context: BrandContext,
    ) -> list[str]:
        """
        为单个用户生成初始记忆

        Args:
            user: 用户
            brand_context: 品牌上下文

        Returns:
            记忆列表
        """
        prompt = f"""
【用户画像】
{user.profile}

【品牌背景】
品牌名称：{brand_context.brand_name}
{brand_context.description}

基于该用户的特点和品牌的历史，这个用户对该品牌可能有什么记忆或印象？

请生成 1-3 条记忆，每条用一句话表达。
- 记忆应该与用户画像匹配（如KOL可能有合作经历，普通用户可能有消费体验）
- 如果这类用户可能没有特别的记忆，可以只返回 1 条或返回空列表
- 记忆可以是正面、负面或中性的
"""

        result = await self.llm.generate_structured(prompt, UserMemoryList)
        return result.memories

    async def initialize_memories(
        self,
        users: list[User],
        brand_context: BrandContext,
        max_concurrent: Optional[int] = None,
    ) -> None:
        """
        批量初始化用户记忆（原地修改）

        Args:
            users: 用户列表
            brand_context: 品牌上下文
            max_concurrent: 最大并发数
        """
        # 构建所有 prompt
        prompts = []
        for user in users:
            prompt = f"""
【用户画像】
{user.profile}

【品牌背景】
品牌名称：{brand_context.brand_name}
{brand_context.description}

基于该用户的特点和品牌的历史，这个用户对该品牌可能有什么记忆或印象？

请生成 1-3 条记忆，每条用一句话表达。
- 记忆应该与用户画像匹配（如KOL可能有合作经历，普通用户可能有消费体验）
- 如果这类用户可能没有特别的记忆，可以只返回 1 条或返回空列表
- 记忆可以是正面、负面或中性的
"""
            prompts.append(prompt)

        # 批量调用 LLM
        results = await self.llm.generate_structured_batch(
            prompts,
            UserMemoryList,
            max_concurrent=max_concurrent,
        )

        # 更新用户记忆
        for i, user in enumerate(users):
            result = results[i]
            if isinstance(result, Exception):
                user.memory = []
            else:
                user.memory = result.memories

"""
用户反应模拟 - 核心 LLM 调用
"""

from typing import Optional

from sentisim.llm import SentiSimLLM
from sentisim.models import Post, User, UserResponse


class ResponseSimulator:
    """用户反应模拟器"""

    def __init__(self, llm: SentiSimLLM):
        self.llm = llm

    def _format_memory(self, user: User) -> str:
        """格式化用户记忆"""
        if not user.memory:
            return "（暂无特别记忆）"
        return "\n".join(f"- {m}" for m in user.memory)

    def _format_author(self, author: Optional[User], is_brand: bool) -> str:
        """格式化发布者描述"""
        if is_brand:
            return "品牌官方账号"
        if author:
            # 截取画像的前100字
            profile_preview = author.profile[:100]
            if len(author.profile) > 100:
                profile_preview += "..."
            return f"一位用户：{profile_preview}"
        return "某位用户"

    async def simulate(
        self,
        user: User,
        post: Post,
        author: Optional[User] = None,
        is_brand_post: bool = False,
    ) -> UserResponse:
        """
        模拟用户对帖子的反应

        Args:
            user: 看到帖子的用户
            post: 帖子内容
            author: 帖子作者（可选）
            is_brand_post: 是否为品牌帖子

        Returns:
            用户反应
        """
        memory_text = self._format_memory(user)
        author_desc = self._format_author(author, is_brand_post)

        prompt = f"""
【你的身份】
{user.profile}

【你对该品牌的记忆】
{memory_text}

【你刚刚看到的内容】
发布者：{author_desc}
内容："{post.content}"

---

请完全代入这个用户的身份，回答以下问题：

1. **第一反应**：看到这条内容，你的第一反应是什么？（一句话）

2. **情绪感受**：你现在的情绪是怎样的？（如：开心、无感、反感、愤怒、担忧、想吐槽等）

3. **印象变化**：这条内容会改变你对该品牌的印象吗？如果会，怎么变？（一句话描述变化，如果没变化就回答"没有变化"）

4. **行动决定**：你会怎么做？选择一个：
   - ignore: 划走，不互动
   - like: 点赞
   - forward: 直接转发
   - forward_comment: 转发并写一句评论

5. **发布内容**：如果选择了 forward_comment，你会写什么评论？（否则留空或填 null）

注意：
- 请真实模拟这个用户的反应，不要过于戏剧化
- 大多数普通用户看到内容后会选择 ignore 或 like
- 只有内容真正触动用户时才会转发
"""

        response = await self.llm.generate_structured(prompt, UserResponse)
        return response

    async def simulate_batch(
        self,
        user_post_pairs: list[tuple[User, Post, Optional[User], bool]],
        max_concurrent: Optional[int] = None,
    ) -> list[UserResponse | Exception]:
        """
        批量模拟用户反应

        Args:
            user_post_pairs: [(用户, 帖子, 作者, 是否品牌帖子), ...]
            max_concurrent: 最大并发数

        Returns:
            反应列表（可能包含异常）
        """
        prompts = []

        for user, post, author, is_brand_post in user_post_pairs:
            memory_text = self._format_memory(user)
            author_desc = self._format_author(author, is_brand_post)

            prompt = f"""
【你的身份】
{user.profile}

【你对该品牌的记忆】
{memory_text}

【你刚刚看到的内容】
发布者：{author_desc}
内容："{post.content}"

---

请完全代入这个用户的身份，回答以下问题：

1. **第一反应**：看到这条内容，你的第一反应是什么？（一句话）

2. **情绪感受**：你现在的情绪是怎样的？（如：开心、无感、反感、愤怒、担忧、想吐槽等）

3. **印象变化**：这条内容会改变你对该品牌的印象吗？如果会，怎么变？（一句话描述变化，如果没变化就回答"没有变化"）

4. **行动决定**：你会怎么做？选择一个：
   - ignore: 划走，不互动
   - like: 点赞
   - forward: 直接转发
   - forward_comment: 转发并写一句评论

5. **发布内容**：如果选择了 forward_comment，你会写什么评论？（否则留空或填 null）

注意：
- 请真实模拟这个用户的反应，不要过于戏剧化
- 大多数普通用户看到内容后会选择 ignore 或 like
- 只有内容真正触动用户时才会转发
"""
            prompts.append(prompt)

        results = await self.llm.generate_structured_batch(
            prompts,
            UserResponse,
            max_concurrent=max_concurrent,
        )

        return results

"""
用户画像生成器
"""

import random
from typing import Optional

from sentisim.llm import SentiSimLLM
from sentisim.models import InfluenceLevel, PersonaType, User
from sentisim.models.user import UserProfile


class UserGenerator:
    """用户画像生成器"""

    def __init__(self, llm: SentiSimLLM):
        self.llm = llm

    def _select_persona(self, persona_types: list[PersonaType]) -> PersonaType:
        """按比例随机选择人群类型"""
        weights = [p.proportion for p in persona_types]
        return random.choices(persona_types, weights=weights, k=1)[0]

    async def generate_profile(
        self,
        persona: PersonaType,
        influence_level: InfluenceLevel,
        follower_count: int,
    ) -> str:
        """
        生成单个用户画像

        Args:
            persona: 人群类型
            influence_level: 影响力层级
            follower_count: 粉丝数量

        Returns:
            用户画像描述
        """
        influence_desc = {
            InfluenceLevel.KOL: "KOL/意见领袖，在某个领域有较大影响力",
            InfluenceLevel.ACTIVE: "活跃用户，经常发言互动，有一定影响力",
            InfluenceLevel.NORMAL: "普通用户，偶尔互动",
            InfluenceLevel.SILENT: "沉默用户，主要浏览很少发言",
        }

        prompt = f"""
【人群类型】
{persona.type_name}

【人群特征】
{persona.description}

【该用户的影响力】
影响力层级：{influence_desc[influence_level]}
粉丝数量：约 {follower_count} 人

请生成一个具体用户的画像描述（80-120字），包含：
- 年龄、性别、职业、所在城市
- 性格特点和表达风格
- 与该品牌的具体故事或关系
- 如果是KOL，说明其影响力领域和风格

要求画像具体生动，有个人特色，像是描述一个真实的人。
"""

        result = await self.llm.generate_structured(prompt, UserProfile)
        return result.profile

    async def generate_users(
        self,
        user_ids: list[str],
        persona_types: list[PersonaType],
        influence_levels: dict[str, InfluenceLevel],
        follower_counts: dict[str, int],
        follower_map: dict[str, list[str]],
        max_concurrent: Optional[int] = None,
    ) -> list[User]:
        """
        批量生成用户

        Args:
            user_ids: 用户ID列表
            persona_types: 人群类型列表
            influence_levels: 用户ID -> 影响力层级
            follower_counts: 用户ID -> 粉丝数
            follower_map: 用户ID -> 粉丝ID列表
            max_concurrent: 最大并发数

        Returns:
            用户列表
        """
        # 为每个用户分配人群类型
        user_personas = {uid: self._select_persona(persona_types) for uid in user_ids}

        # 构建所有 prompt
        prompts = []
        for uid in user_ids:
            persona = user_personas[uid]
            level = influence_levels[uid]
            count = follower_counts[uid]

            influence_desc = {
                InfluenceLevel.KOL: "KOL/意见领袖，在某个领域有较大影响力",
                InfluenceLevel.ACTIVE: "活跃用户，经常发言互动，有一定影响力",
                InfluenceLevel.NORMAL: "普通用户，偶尔互动",
                InfluenceLevel.SILENT: "沉默用户，主要浏览很少发言",
            }

            prompt = f"""
【人群类型】
{persona.type_name}

【人群特征】
{persona.description}

【该用户的影响力】
影响力层级：{influence_desc[level]}
粉丝数量：约 {count} 人

请生成一个具体用户的画像描述（80-120字），包含：
- 年龄、性别、职业、所在城市
- 性格特点和表达风格
- 与该品牌的具体故事或关系
- 如果是KOL，说明其影响力领域和风格

要求画像具体生动，有个人特色，像是描述一个真实的人。
"""
            prompts.append(prompt)

        # 批量调用 LLM
        results = await self.llm.generate_structured_batch(
            prompts,
            UserProfile,
            max_concurrent=max_concurrent,
        )

        # 构建用户对象
        users = []
        for i, uid in enumerate(user_ids):
            result = results[i]

            # 处理可能的异常
            if isinstance(result, Exception):
                profile = f"[生成失败] {user_personas[uid].type_name}类型用户"
            else:
                profile = result.profile

            user = User(
                user_id=uid,
                persona_type=user_personas[uid].type_name,
                profile=profile,
                influence_level=influence_levels[uid],
                memory=[],
                follower_ids=follower_map.get(uid, []),
            )
            users.append(user)

        return users

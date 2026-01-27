"""
用户画像生成器 - 批量生成优化版本
"""

import random
from collections import defaultdict
from typing import Optional

from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext, InfluenceLevel, PersonaType, User
from sentisim.models.user import BatchUsersWithMemories, UserProfile


class UserGenerator:
    """用户画像生成器"""

    def __init__(self, llm: SentiSimLLM):
        self.llm = llm

    def _select_persona(self, persona_types: list[PersonaType]) -> PersonaType:
        """按比例随机选择人群类型"""
        weights = [p.proportion for p in persona_types]
        return random.choices(persona_types, weights=weights, k=1)[0]

    def _get_influence_desc(self, level: InfluenceLevel) -> str:
        """获取影响力层级的描述"""
        return {
            InfluenceLevel.KOL: "KOL/意见领袖，在某个领域有较大影响力，粉丝众多",
            InfluenceLevel.ACTIVE: "活跃用户，经常发言互动，有一定影响力",
            InfluenceLevel.NORMAL: "普通用户，偶尔互动，影响力一般",
            InfluenceLevel.SILENT: "沉默用户，主要浏览很少发言，影响力较小",
        }[level]

    def _build_batch_prompt(
        self,
        persona: PersonaType,
        influence_level: InfluenceLevel,
        count: int,
        brand_context: BrandContext,
    ) -> str:
        """构建批量生成用户画像和初始记忆的 prompt"""
        influence_desc = self._get_influence_desc(influence_level)

        return f"""
【品牌背景】
品牌名称：{brand_context.brand_name}
{brand_context.description}

【人群类型】
{persona.type_name}

【人群特征】
{persona.description}

【社交网络中影响力层级】
{influence_desc}

【需要生成的用户数量】
{count} 个

请生成 {count} 个不同的用户，每个用户包含画像和初始记忆。

要求：
1. 所有用户都属于"{persona.type_name}"人群类型
2. 所有用户都是"{influence_level.value}"影响力层级
3. 每个用户要有独特的个人特色，避免雷同
4. 用户之间要有年龄、性别、职业、城市等方面的多样性
5. 描述中需要体现在社交网络中的影响力层级

每个用户包含两个字段：
1. profile（画像，80-120字）：
   - 年龄、性别、职业、所在城市
   - 性格特点和表达风格
   - 与该品牌的具体故事或关系
   {"- 作为KOL的影响力领域和风格特点" if influence_level == InfluenceLevel.KOL else ""}

2. initial_memory（初始记忆，一句话）：
   - 基于用户画像，这个用户对该品牌的一条记忆或印象
   - 可以是正面、负面或中性的，需要符合这个人群的特征
   - 例如："去年夏天第一次喝XX饮料，觉得味道清爽"

请确保返回的 users 列表包含恰好 {count} 个用户。
"""

    async def generate_profile(
        self,
        persona: PersonaType,
        influence_level: InfluenceLevel,
        follower_count: int,
    ) -> str:
        """
        生成单个用户画像（保留兼容性）

        Args:
            persona: 人群类型
            influence_level: 影响力层级
            follower_count: 粉丝数量

        Returns:
            用户画像描述
        """
        influence_desc = self._get_influence_desc(influence_level)

        prompt = f"""
【人群类型】
{persona.type_name}

【人群特征】
{persona.description}

【该用户的影响力】
影响力层级：{influence_desc}
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
        brand_context: BrandContext,
        max_concurrent: Optional[int] = None,
    ) -> list[User]:
        """
        批量生成用户 - 按 (persona, influence_level) 分组优化，同时生成初始记忆

        Args:
            user_ids: 用户ID列表
            persona_types: 人群类型列表
            influence_levels: 用户ID -> 影响力层级
            follower_counts: 用户ID -> 粉丝数
            follower_map: 用户ID -> 粉丝ID列表
            brand_context: 品牌上下文
            max_concurrent: 最大并发数

        Returns:
            用户列表（已包含初始记忆）
        """
        # 1. 为每个用户分配人群类型
        user_personas: dict[str, PersonaType] = {
            uid: self._select_persona(persona_types) for uid in user_ids
        }

        # 2. 按 (persona_name, influence_level) 分组
        groups: dict[tuple[str, InfluenceLevel], list[str]] = defaultdict(list)
        persona_map: dict[str, PersonaType] = {}  # persona_name -> PersonaType

        for uid in user_ids:
            persona = user_personas[uid]
            level = influence_levels[uid]
            key = (persona.type_name, level)
            groups[key].append(uid)
            persona_map[persona.type_name] = persona

        # 3. 为每个分组构建 prompt
        group_keys = list(groups.keys())
        prompts = []

        for persona_name, level in group_keys:
            uids = groups[(persona_name, level)]
            persona = persona_map[persona_name]
            prompt = self._build_batch_prompt(persona, level, len(uids), brand_context)
            prompts.append(prompt)

        # 4. 批量调用 LLM
        results = await self.llm.generate_structured_batch(
            prompts,
            BatchUsersWithMemories,
            max_concurrent=max_concurrent,
        )

        # 5. 分配画像和记忆到具体用户
        user_profiles: dict[str, str] = {}
        user_memories: dict[str, str] = {}

        for i, (persona_name, level) in enumerate(group_keys):
            uids = groups[(persona_name, level)]
            result = results[i]

            if isinstance(result, Exception):
                # 生成失败，使用默认值
                for uid in uids:
                    user_profiles[uid] = (
                        f"[生成失败] {persona_name}类型的{level.value}用户"
                    )
                    user_memories[uid] = ""
            else:
                user_data = result.users
                # 随机打乱顺序，然后分配给用户
                random.shuffle(user_data)

                for j, uid in enumerate(uids):
                    if j < len(user_data):
                        user_profiles[uid] = user_data[j].profile
                        user_memories[uid] = user_data[j].initial_memory
                    else:
                        # 如果数量不足，复用已有数据
                        idx = j % len(user_data) if user_data else 0
                        if user_data:
                            user_profiles[uid] = user_data[idx].profile
                            user_memories[uid] = user_data[idx].initial_memory
                        else:
                            user_profiles[uid] = f"[画像不足] {persona_name}类型用户"
                            user_memories[uid] = ""

        # 6. 构建用户对象（已包含初始记忆）
        users = []
        for uid in user_ids:
            initial_memory = user_memories.get(uid, "")
            user = User(
                user_id=uid,
                persona_type=user_personas[uid].type_name,
                profile=user_profiles.get(uid, "[未生成画像]"),
                influence_level=influence_levels[uid],
                memory=[initial_memory] if initial_memory else [],
                follower_ids=follower_map.get(uid, []),
            )
            users.append(user)

        return users

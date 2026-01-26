"""
World - 虚拟世界管理

负责：
- 创建虚拟世界（网络拓扑、用户画像、记忆）
- 持久化存储和加载
- 运行模拟
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sentisim.generators import MemoryGenerator, PersonaGenerator, UserGenerator
from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext, InfluenceLevel, PersonaType, User
from sentisim.network import InfluenceAssigner, NetworkBuilder
from sentisim.simulation import SimulationEngine, SimulationResult


@dataclass
class WorldMeta:
    """世界元信息"""

    world_id: str
    created_at: str
    user_count: int
    persona_count: int
    brand_context: BrandContext

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "created_at": self.created_at,
            "user_count": self.user_count,
            "persona_count": self.persona_count,
            "brand_context": self.brand_context.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldMeta":
        return cls(
            world_id=data["world_id"],
            created_at=data["created_at"],
            user_count=data["user_count"],
            persona_count=data["persona_count"],
            brand_context=BrandContext(**data["brand_context"]),
        )


class World:
    """虚拟世界"""

    def __init__(
        self,
        meta: WorldMeta,
        personas: list[PersonaType],
        users: list[User],
        network_data: dict,
        brand_account_id: str,
    ):
        self.meta = meta
        self.personas = personas
        self.users = users
        self.users_dict = {u.user_id: u for u in users}
        self.network_data = network_data
        self.brand_account_id = brand_account_id

        # 保存初始记忆用于重置
        self._initial_memories = {u.user_id: u.memory.copy() for u in users}

    def reset_memories(self):
        """重置所有用户的记忆到初始状态"""
        for user in self.users:
            user.memory = self._initial_memories[user.user_id].copy()
        logger.info("已重置所有用户记忆到初始状态")

    def get_user(self, user_id: str) -> Optional[User]:
        return self.users_dict.get(user_id)

    # ========================================
    # 创建
    # ========================================

    @classmethod
    async def create(
        cls,
        brand_context: BrandContext,
        user_count: int = 100,
        persona_count: int = 6,
        max_concurrent: int = 10,
        llm: Optional[SentiSimLLM] = None,
        world_id: Optional[str] = None,
    ) -> "World":
        """
        创建新的虚拟世界

        Args:
            brand_context: 品牌上下文
            user_count: 用户数量
            persona_count: 人群类型数量
            max_concurrent: LLM 并发数
            llm: LLM 客户端（可选，不传则自动创建）
            world_id: 世界ID（可选，不传则自动生成）

        Returns:
            World 实例
        """
        if llm is None:
            llm = SentiSimLLM()

        if world_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = brand_context.brand_name.replace(" ", "_")[:20]
            world_id = f"{safe_name}_{user_count}users_{timestamp}"

        logger.info(f"创建虚拟世界: {world_id}")

        # 1. 构建网络拓扑
        logger.info(f"[1/5] 构建网络拓扑 (用户数: {user_count})")
        builder = NetworkBuilder()
        topology = builder.build(user_count=user_count)
        logger.info(
            f"  节点: {len(topology.user_ids)}, 边: {topology.graph.number_of_edges()}"
        )

        # 2. 分配影响力
        logger.info("[2/5] 分配影响力角色")
        assigner = InfluenceAssigner()
        influence_levels = assigner.assign(topology)
        stats = assigner.get_stats(influence_levels)
        logger.info(f"  分布: {stats['counts']}")

        # 3. 生成人群类型
        logger.info(f"[3/5] 生成人群类型 (数量: {persona_count})")
        persona_gen = PersonaGenerator(llm)
        personas = await persona_gen.generate(brand_context, count=persona_count)
        for p in personas:
            logger.info(f"  [{p.proportion:5.1f}%] {p.type_name}")

        # 4. 生成用户画像
        logger.info(f"[4/5] 生成用户画像 (并发: {max_concurrent})")
        user_gen = UserGenerator(llm)
        users = await user_gen.generate_users(
            user_ids=topology.user_ids,
            persona_types=personas,
            influence_levels=influence_levels,
            follower_counts=topology.follower_counts,
            follower_map=topology.follower_map,
            max_concurrent=max_concurrent,
        )
        logger.info(f"  生成了 {len(users)} 个用户")

        # 5. 初始化记忆
        logger.info(f"[5/5] 初始化用户记忆 (并发: {max_concurrent})")
        memory_gen = MemoryGenerator(llm)
        await memory_gen.initialize_memories(
            users, brand_context, max_concurrent=max_concurrent
        )
        users_with_memory = sum(1 for u in users if u.memory)
        logger.info(f"  有记忆的用户: {users_with_memory}/{len(users)}")

        # 构建网络数据（用于持久化）
        network_data = {
            "edges": list(topology.graph.edges()),
            "user_ids": topology.user_ids,
            "brand_account_id": topology.brand_account_id,
            "brand_followers": topology.get_followers(topology.brand_account_id),
        }

        meta = WorldMeta(
            world_id=world_id,
            created_at=datetime.now().isoformat(),
            user_count=user_count,
            persona_count=persona_count,
            brand_context=brand_context,
        )

        logger.info(f"虚拟世界创建完成，LLM 调用次数: {llm.call_count}")

        return cls(
            meta=meta,
            personas=personas,
            users=users,
            network_data=network_data,
            brand_account_id=topology.brand_account_id,
        )

    # ========================================
    # 持久化
    # ========================================

    def save(self, base_path: str = "data/worlds") -> Path:
        """
        保存世界到磁盘

        Args:
            base_path: 基础路径

        Returns:
            保存的目录路径
        """
        world_dir = Path(base_path) / self.meta.world_id
        world_dir.mkdir(parents=True, exist_ok=True)

        # 1. 元信息
        meta_file = world_dir / "meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta.to_dict(), f, ensure_ascii=False, indent=2)

        # 2. 网络拓扑
        network_file = world_dir / "network.json"
        with open(network_file, "w", encoding="utf-8") as f:
            json.dump(self.network_data, f, ensure_ascii=False, indent=2)

        # 3. 人群类型
        personas_file = world_dir / "personas.json"
        personas_data = [p.model_dump() for p in self.personas]
        with open(personas_file, "w", encoding="utf-8") as f:
            json.dump(personas_data, f, ensure_ascii=False, indent=2)

        # 4. 用户（包含初始记忆）
        users_file = world_dir / "users.json"
        users_data = []
        for user in self.users:
            user_dict = user.model_dump()
            user_dict["influence_level"] = user.influence_level.value
            # 保存初始记忆
            user_dict["initial_memory"] = self._initial_memories[user.user_id]
            users_data.append(user_dict)
        with open(users_file, "w", encoding="utf-8") as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)

        # 5. 创建 simulations 目录
        sim_dir = world_dir / "simulations"
        sim_dir.mkdir(exist_ok=True)

        logger.info(f"世界已保存到: {world_dir}")
        return world_dir

    @classmethod
    def load(cls, world_path: str) -> "World":
        """
        从磁盘加载世界

        Args:
            world_path: 世界目录路径

        Returns:
            World 实例
        """
        world_dir = Path(world_path)

        if not world_dir.exists():
            raise FileNotFoundError(f"世界目录不存在: {world_dir}")

        logger.info(f"加载世界: {world_dir}")

        # 1. 元信息
        with open(world_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = WorldMeta.from_dict(json.load(f))

        # 2. 网络拓扑
        with open(world_dir / "network.json", "r", encoding="utf-8") as f:
            network_data = json.load(f)

        # 3. 人群类型
        with open(world_dir / "personas.json", "r", encoding="utf-8") as f:
            personas = [PersonaType(**p) for p in json.load(f)]

        # 4. 用户
        with open(world_dir / "users.json", "r", encoding="utf-8") as f:
            users_data = json.load(f)

        users = []
        initial_memories = {}
        for ud in users_data:
            initial_memories[ud["user_id"]] = ud.pop(
                "initial_memory", ud.get("memory", [])
            )
            ud["influence_level"] = InfluenceLevel(ud["influence_level"])
            users.append(User(**ud))

        world = cls(
            meta=meta,
            personas=personas,
            users=users,
            network_data=network_data,
            brand_account_id=network_data["brand_account_id"],
        )
        world._initial_memories = initial_memories

        logger.info(f"世界加载完成: {meta.world_id}")
        logger.info(f"  用户数: {len(users)}, 人群类型: {len(personas)}")

        return world

    # ========================================
    # 模拟
    # ========================================

    async def simulate(
        self,
        test_content: str,
        max_steps: int = 10,
        max_concurrent: int = 10,
        reset_memory: bool = True,
        save_result: bool = True,
        llm: Optional[SentiSimLLM] = None,
        on_step_complete: Optional[callable] = None,
    ) -> SimulationResult:
        """
        运行模拟

        Args:
            test_content: 待测试内容
            max_steps: 最大步数
            max_concurrent: LLM 并发数
            reset_memory: 是否重置记忆到初始状态
            save_result: 是否保存结果
            llm: LLM 客户端
            on_step_complete: 每步回调

        Returns:
            模拟结果
        """
        if llm is None:
            llm = SentiSimLLM()

        # 重置记忆
        if reset_memory:
            self.reset_memories()

        logger.info(f"开始模拟: {test_content[:50]}...")

        # 获取关注品牌的用户列表
        brand_followers = self.network_data.get("brand_followers", [])

        engine = SimulationEngine(
            llm=llm,
            users=self.users_dict,
            brand_account_id=self.brand_account_id,
            brand_followers=brand_followers,
            max_concurrent=max_concurrent,
        )

        result = await engine.run(
            test_content=test_content,
            max_steps=max_steps,
            on_step_complete=on_step_complete,
        )

        logger.info(f"模拟完成: 触达 {result.total_reach}, 步数 {result.steps_run}")

        # 保存结果
        if save_result:
            self._save_simulation_result(test_content, max_steps, result)

        return result

    def _save_simulation_result(
        self,
        test_content: str,
        max_steps: int,
        result: SimulationResult,
    ) -> Path:
        """保存模拟结果"""
        # 找到 world 目录
        world_dir = Path("data/worlds") / self.meta.world_id
        sim_dir = world_dir / "simulations"
        sim_dir.mkdir(parents=True, exist_ok=True)

        # 生成模拟ID
        existing = list(sim_dir.glob("sim_*"))
        sim_id = f"sim_{len(existing) + 1:03d}"
        sim_path = sim_dir / sim_id
        sim_path.mkdir()

        # 1. 配置
        config = {
            "sim_id": sim_id,
            "created_at": datetime.now().isoformat(),
            "test_content": test_content,
            "max_steps": max_steps,
        }
        with open(sim_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 2. 结果统计
        stats = {
            "total_reach": result.total_reach,
            "steps_run": result.steps_run,
            "total_posts": len(result.posts),
            "spreading_posts": len(result.get_spreading_posts()),
            "action_counts": result.get_action_counts(),
            "emotion_counts": result.get_emotion_counts(),
            "negative_count": len(result.get_negative_responses()),
        }
        with open(sim_path / "result.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 3. 详细响应 (JSONL)
        with open(sim_path / "responses.jsonl", "w", encoding="utf-8") as f:
            for record in result.responses:
                user = self.users_dict[record.user_id]
                data = {
                    "step": record.step,
                    "post_id": record.post_id,
                    "user_id": record.user_id,
                    "user_persona": record.user_persona,
                    "influence_level": user.influence_level.value,
                    "response": {
                        "first_reaction": record.response.first_reaction,
                        "emotion": record.response.emotion,
                        "impression_change": record.response.impression_change,
                        "action": record.response.action.value,
                        "content": record.response.content,
                        "will_spread": record.response.will_spread,
                        "is_negative": record.response.is_negative,
                    },
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # 4. 帖子列表
        posts_data = []
        for post in result.posts:
            posts_data.append(
                {
                    "post_id": post.post_id,
                    "author_id": post.author_id,
                    "content": post.content,
                    "post_type": post.post_type.value,
                    "original_post_id": post.original_post_id,
                    "timestamp": post.timestamp,
                }
            )
        with open(sim_path / "posts.json", "w", encoding="utf-8") as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)

        logger.info(f"模拟结果已保存到: {sim_path}")
        return sim_path

    def list_simulations(self) -> list[dict]:
        """列出所有模拟"""
        world_dir = Path("data/worlds") / self.meta.world_id
        sim_dir = world_dir / "simulations"

        if not sim_dir.exists():
            return []

        simulations = []
        for sim_path in sorted(sim_dir.glob("sim_*")):
            config_file = sim_path / "config.json"
            result_file = sim_path / "result.json"

            if config_file.exists() and result_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                with open(result_file, "r", encoding="utf-8") as f:
                    result = json.load(f)

                simulations.append(
                    {
                        "sim_id": config["sim_id"],
                        "created_at": config["created_at"],
                        "test_content": config["test_content"][:50] + "...",
                        "total_reach": result["total_reach"],
                        "negative_count": result["negative_count"],
                    }
                )

        return simulations

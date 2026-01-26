"""
传播模拟引擎 - 主循环
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from sentisim.llm import SentiSimLLM
from sentisim.models import ActionType, Post, PostType, User, UserResponse
from sentisim.simulation.memory import MemoryManager
from sentisim.simulation.response import ResponseSimulator


@dataclass
class ResponseRecord:
    """单条反应记录"""

    user_id: str
    user_persona: str
    post_id: str
    step: int
    response: UserResponse


@dataclass
class SimulationResult:
    """模拟结果"""

    posts: list[Post] = field(default_factory=list)
    responses: list[ResponseRecord] = field(default_factory=list)

    # 统计数据
    total_reach: int = 0
    steps_run: int = 0

    def get_action_counts(self) -> dict[str, int]:
        """获取行动分布"""
        counts = {action.value: 0 for action in ActionType}
        for record in self.responses:
            counts[record.response.action.value] += 1
        return counts

    def get_emotion_counts(self) -> dict[str, int]:
        """获取情绪分布"""
        counts = {}
        for record in self.responses:
            emotion = record.response.emotion
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts

    def get_negative_responses(self) -> list[ResponseRecord]:
        """获取负面反应"""
        return [r for r in self.responses if r.response.is_negative]

    def get_spreading_posts(self) -> list[Post]:
        """获取产生传播的帖子（非品牌原创）"""
        return [p for p in self.posts if p.post_type != PostType.BRAND_ORIGINAL]

    def get_responses_by_persona(self) -> dict[str, list[ResponseRecord]]:
        """按人群类型分组反应"""
        by_persona = {}
        for record in self.responses:
            persona = record.user_persona
            if persona not in by_persona:
                by_persona[persona] = []
            by_persona[persona].append(record)
        return by_persona


class SimulationEngine:
    """传播模拟引擎"""

    def __init__(
        self,
        llm: SentiSimLLM,
        users: dict[str, User],
        brand_account_id: str,
        brand_followers: list[str] = None,
        max_concurrent: Optional[int] = None,
    ):
        """
        Args:
            llm: LLM 客户端
            users: 用户字典 {user_id: User}
            brand_account_id: 品牌账号ID
            brand_followers: 关注品牌的用户ID列表
            max_concurrent: 最大并发数
        """
        self.llm = llm
        self.users = users
        self.brand_account_id = brand_account_id
        self.brand_followers = set(brand_followers) if brand_followers else set()
        self.max_concurrent = max_concurrent

        self.response_simulator = ResponseSimulator(llm)
        self.memory_manager = MemoryManager()

        # 追踪每个帖子的根帖子ID {post_id: root_post_id}
        self._post_roots: dict[str, str] = {}
        # 追踪已触达用户 {root_post_id: set(user_ids)}
        self._seen_users: dict[str, set[str]] = {}

    def _generate_post_id(self) -> str:
        """生成帖子ID"""
        return f"post_{uuid.uuid4().hex[:8]}"

    def _get_root_post_id(self, post: Post) -> str:
        """获取帖子的根帖子ID（用于去重）"""
        if post.post_id in self._post_roots:
            return self._post_roots[post.post_id]
        # 品牌原帖的根是自己
        return post.post_id

    def _has_seen_root(self, user_id: str, root_post_id: str) -> bool:
        """检查用户是否已经看过这个根帖子"""
        return user_id in self._seen_users.get(root_post_id, set())

    def _mark_as_seen(self, user_id: str, root_post_id: str) -> None:
        """标记用户已经看过这个根帖子"""
        if root_post_id not in self._seen_users:
            self._seen_users[root_post_id] = set()
        self._seen_users[root_post_id].add(user_id)

    def _get_audience(self, author_id: str) -> list[str]:
        """获取能看到帖子的用户（作者的粉丝）"""
        if author_id == self.brand_account_id:
            # 品牌帖子：只有关注品牌的用户能看到
            return [uid for uid in self.brand_followers if uid in self.users]

        author = self.users.get(author_id)
        if author:
            return author.follower_ids
        return []

    def _create_new_post(
        self,
        author_id: str,
        response: UserResponse,
        original_post: Post,
        timestamp: int,
    ) -> Optional[Post]:
        """根据用户反应创建新帖子"""
        if not response.will_spread:
            return None

        # 确定帖子类型
        if response.action == ActionType.FORWARD:
            post_type = PostType.FORWARD
            content = original_post.content  # 直接转发，内容不变
        elif response.action == ActionType.FORWARD_COMMENT:
            post_type = PostType.FORWARD_COMMENT
            content = response.content or f"转发：{original_post.content[:50]}..."
        else:  # CREATE
            post_type = PostType.CREATION
            content = response.content or "[用户原创内容]"

        return Post(
            post_id=self._generate_post_id(),
            author_id=author_id,
            content=content,
            post_type=post_type,
            original_post_id=original_post.post_id,
            timestamp=timestamp,
        )

    async def run(
        self,
        test_content: str,
        max_steps: int = 10,
        on_step_complete: Optional[callable] = None,
        verbose: bool = False,
    ) -> SimulationResult:
        """
        运行传播模拟

        Args:
            test_content: 待测试的品牌内容
            max_steps: 最大模拟步数
            on_step_complete: 每步完成后的回调 (step, wave_size, new_posts_count)
            verbose: 是否打印详细日志

        Returns:
            模拟结果
        """
        result = SimulationResult()

        # 重置追踪状态
        self._post_roots = {}
        self._seen_users = {}

        # 创建品牌初始帖子
        brand_post = Post(
            post_id=self._generate_post_id(),
            author_id=self.brand_account_id,
            content=test_content,
            post_type=PostType.BRAND_ORIGINAL,
            timestamp=0,
        )
        result.posts.append(brand_post)

        # 品牌原帖的根是自己
        self._post_roots[brand_post.post_id] = brand_post.post_id

        # 当前传播队列
        current_wave = [brand_post]

        if verbose:
            print(f"\n[Verbose] 开始模拟，最大步数: {max_steps}")
            print(f"[Verbose] 品牌帖子: {brand_post.post_id}")

        for step in range(max_steps):
            if not current_wave:
                if verbose:
                    print(f"\n[Step {step}] 当前波次为空，结束模拟")
                break

            next_wave = []

            # 收集本轮所有需要模拟的 (用户, 帖子, 作者, 是否品牌帖子)
            simulation_tasks = []
            task_metadata = []  # 记录元数据用于后续处理

            if verbose:
                print(f"\n[Step {step}] 当前波次帖子数: {len(current_wave)}")

            skipped_count = 0  # 记录跳过的用户数

            for post in current_wave:
                is_brand_post = post.author_id == self.brand_account_id
                author = None if is_brand_post else self.users.get(post.author_id)

                # 获取这个帖子的根帖子ID
                root_post_id = self._get_root_post_id(post)

                # 获取能看到这条帖子的用户
                audience_ids = self._get_audience(post.author_id)

                post_skipped = 0
                post_reached = 0

                for user_id in audience_ids:
                    # 跳过已经看过这个根帖子的用户
                    if self._has_seen_root(user_id, root_post_id):
                        skipped_count += 1
                        post_skipped += 1
                        continue

                    user = self.users.get(user_id)
                    if user:
                        # 标记为已触达
                        self._mark_as_seen(user_id, root_post_id)
                        post_reached += 1

                        simulation_tasks.append((user, post, author, is_brand_post))
                        task_metadata.append(
                            {
                                "user_id": user_id,
                                "user_persona": user.persona_type,
                                "post_id": post.post_id,
                                "post": post,
                                "step": step,
                                "root_post_id": root_post_id,
                            }
                        )

                if verbose:
                    post_type_str = "品牌帖" if is_brand_post else post.post_type.value
                    print(f"  - 帖子 {post.post_id[:12]}... ({post_type_str}): "
                          f"受众={len(audience_ids)}, 新触达={post_reached}, 跳过={post_skipped}")

            if verbose:
                print(f"[Step {step}] 总计: 新触达用户={len(simulation_tasks)}, 跳过已触达={skipped_count}")

            if not simulation_tasks:
                if verbose:
                    print(f"[Step {step}] 没有新用户需要处理，结束模拟")
                break

            # 批量模拟用户反应
            if verbose:
                print(f"[Step {step}] 正在模拟 {len(simulation_tasks)} 个用户的反应...")

            responses = await self.response_simulator.simulate_batch(
                simulation_tasks,
                max_concurrent=self.max_concurrent,
            )

            # 处理结果
            step_action_counts: dict[str, int] = {}
            error_count = 0

            for i, response in enumerate(responses):
                meta = task_metadata[i]
                user = self.users[meta["user_id"]]

                if isinstance(response, Exception):
                    # 跳过失败的模拟
                    error_count += 1
                    continue

                # 统计 action
                action_name = response.action.value
                step_action_counts[action_name] = step_action_counts.get(action_name, 0) + 1

                # 记录反应
                record = ResponseRecord(
                    user_id=meta["user_id"],
                    user_persona=meta["user_persona"],
                    post_id=meta["post_id"],
                    step=meta["step"],
                    response=response,
                )
                result.responses.append(record)
                result.total_reach += 1

                # 更新用户记忆
                self.memory_manager.update(user, response)

                # 如果产生新内容，加入下一轮传播
                new_post = self._create_new_post(
                    author_id=meta["user_id"],
                    response=response,
                    original_post=meta["post"],
                    timestamp=step + 1,
                )
                if new_post:
                    # 设置新帖子的根帖子ID
                    if new_post.post_type == PostType.CREATION:
                        # 二创内容：root 是自己，形成新的传播链
                        self._post_roots[new_post.post_id] = new_post.post_id
                    else:
                        # 转发类内容：继承原帖的根帖子ID
                        self._post_roots[new_post.post_id] = meta["root_post_id"]
                    next_wave.append(new_post)
                    result.posts.append(new_post)

                    if verbose:
                        root_info = "新传播链" if new_post.post_type == PostType.CREATION else f"继承自 {meta['root_post_id'][:12]}..."
                        content_preview = new_post.content[:50] + "..." if len(new_post.content) > 50 else new_post.content
                        print(f"    [新帖子] {new_post.post_id[:12]}... ({new_post.post_type.value}, {root_info}) "
                              f"by {meta['user_persona']}: \"{content_preview}\"")

            result.steps_run = step + 1

            if verbose:
                action_summary = ", ".join(f"{k}={v}" for k, v in sorted(step_action_counts.items()))
                print(f"[Step {step}] 反应统计: {action_summary}")
                if error_count > 0:
                    print(f"[Step {step}] 错误数: {error_count}")
                print(f"[Step {step}] 产生新帖子: {len(next_wave)} 条")
                print(f"[Step {step}] 累计触达: {result.total_reach}")

            # 回调
            if on_step_complete:
                on_step_complete(step, len(simulation_tasks), len(next_wave))

            current_wave = next_wave

        if verbose:
            print(f"\n[Verbose] 模拟结束，共运行 {result.steps_run} 步")

        return result

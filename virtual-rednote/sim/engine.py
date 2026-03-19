"""
事件驱动仿真引擎
"""
import heapq
import uuid
import math
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import AsyncIterator

from .models import Event, EventType, ActionType, Content, ContentType, Agent, AgentTier, BrandAgent
from .rules import compute_action, cosine_similarity
from llm.comment import generate_comment
from llm.brand import brand_review_comments


@dataclass
class SimulationConfig:
    time_window_days: float = 7.0
    max_events: int = 10000
    brand_check_interval: int = 30   # 每 N 个仿真事件触发一次品牌巡查


class EventQueue:
    def __init__(self):
        self._heap: list[Event] = []

    def push(self, event: Event):
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def __len__(self):
        return len(self._heap)


class SimulationEngine:
    def __init__(
        self,
        sim_id: str,
        world_id: str,
        agents: dict[str, Agent],
        graph,                        # networkx DiGraph
        seed_content: Content,
        config: SimulationConfig | None = None,
        brand_agent: BrandAgent | None = None,
    ):
        self.sim_id = sim_id
        self.world_id = world_id
        self.agents = agents
        self.graph = graph
        self.seed_content = seed_content
        self.config = config or SimulationConfig()
        self.brand_agent = brand_agent

        self.queue = EventQueue()
        self.event_count = 0
        self.metrics = {"reach": 0, "likes": 0, "comments": 0, "reposts": 0}

        # content_id -> Content（内存缓存）
        self._cache: dict[str, Content] = {seed_content.id: seed_content}
        # content_id -> set of agent_ids（已加入曝光队列，避免同一内容重复推给同一人）
        self._exposed: dict[str, set[str]] = {}
        # 全局已处理过的 agent（不管哪条内容），避免同一传播链重复触达
        self._reached: set[str] = set()
        # content_id -> set of agent_ids（已转发，用于 social_pressure）
        self._reposted_by: dict[str, set[str]] = {}
        # LLM 评论生成结果队列
        self._comment_queue: asyncio.Queue = asyncio.Queue()
        self._pending_comments: int = 0
        # 品牌回复队列
        self._brand_queue: asyncio.Queue = asyncio.Queue()
        self._pending_brand: int = 0
        # 已回复过的 agent（不重复回复）
        self._replied_to: set[str] = set()
        # 待巡查的新评论列表 [{agent_id, tier, text}, ...]
        self._new_comments_buf: list[dict] = []
        self._last_brand_check: int = 0

    async def run(self) -> AsyncIterator[dict]:
        self._bootstrap()
        yield self._graph_init_event()
        async for sse in self._loop():
            # 每个仿真事件前，先 flush 已完成的 LLM 评论和品牌回复
            async for comment_sse in self._flush_comments():
                yield comment_sse
            async for brand_sse in self._flush_brand():
                yield brand_sse
            yield sse
        # 仿真结束后，等待所有在途 LLM 任务完成（最多 30 秒）
        try:
            await asyncio.wait_for(self._drain_all(), timeout=30)
        except asyncio.TimeoutError:
            pass
        async for comment_sse in self._flush_comments():
            yield comment_sse
        async for brand_sse in self._flush_brand():
            yield brand_sse
        yield self._done_event()

    async def _flush_comments(self) -> AsyncIterator[dict]:
        """把 comment_queue 里已就绪的评论全部 yield 出去（非阻塞）"""
        while not self._comment_queue.empty():
            yield self._comment_queue.get_nowait()

    async def _flush_brand(self) -> AsyncIterator[dict]:
        """把 brand_queue 里已就绪的品牌回复全部 yield 出去（非阻塞）"""
        while not self._brand_queue.empty():
            yield self._brand_queue.get_nowait()

    async def _drain_all(self):
        """等待所有在途 LLM 任务完成"""
        while self._pending_comments > 0 or self._pending_brand > 0:
            await asyncio.sleep(0.2)

    # ── 启动 ──────────────────────────────────────────────

    def _bootstrap(self):
        """向品牌的直接粉丝推送第一批曝光；若品牌不在图中则给所有 KOL/KOC"""
        self._exposed[self.seed_content.id] = set()
        self._reposted_by[self.seed_content.id] = set()

        brand_id = self.seed_content.author_id
        followers = list(self.graph.predecessors(brand_id)) if brand_id in self.graph else []
        if not followers:
            followers = [a_id for a_id, a in self.agents.items() if a.tier in (AgentTier.KOL, AgentTier.KOC)]

        for agent_id in followers:
            self._exposed[self.seed_content.id].add(agent_id)
            self.queue.push(Event(
                sim_time=0.0,
                type=EventType.CONTENT_EXPOSED,
                payload={"agent_id": agent_id, "content_id": self.seed_content.id},
            ))

    # ── 主循环 ────────────────────────────────────────────

    async def _loop(self) -> AsyncIterator[dict]:
        while self.queue and self.event_count < self.config.max_events:
            event = self.queue.pop()
            if event.sim_time > self.config.time_window_days:
                break

            self.event_count += 1

            if event.type == EventType.CONTENT_EXPOSED:
                sse = self._handle_exposed(event)
                if sse:
                    yield sse

            elif event.type == EventType.CONTENT_PUBLISHED:
                self._handle_published(event)

            # 每 brand_check_interval 个事件触发一次品牌巡查
            interval = self.config.brand_check_interval
            if (self.brand_agent and self._new_comments_buf and
                    self.event_count - self._last_brand_check >= interval):
                self._last_brand_check = self.event_count
                pending, self._new_comments_buf = self._new_comments_buf, []
                self._pending_brand += 1
                asyncio.create_task(self._run_brand_check(pending, event.sim_time))

            await asyncio.sleep(0.5)

    # ── 处理曝光 ──────────────────────────────────────────

    def _handle_exposed(self, event: Event) -> dict | None:
        agent_id   = event.payload["agent_id"]
        content_id = event.payload["content_id"]

        agent   = self.agents.get(agent_id)
        content = self._cache.get(content_id)
        if agent is None or content is None:
            return None

        if agent_id in self._reached:
            return None

        social_pressure = self._social_pressure(agent, content_id)
        action = compute_action(agent, content, social_pressure)
        if action == ActionType.IGNORE:
            return None

        self._reached.add(agent_id)
        self.metrics["reach"] += 1

        if action == ActionType.LIKE:
            self.metrics["likes"] += 1
            content.likes += 1

        elif action == ActionType.COMMENT:
            self.metrics["comments"] += 1
            content.comments += 1
            # 品牌巡查缓冲条目（text 初始为空，_gen_comment 完成后会填充）
            brand_buf_entry = None
            if self.brand_agent and agent_id not in self._replied_to:
                brand_buf_entry = {"agent_id": agent_id, "tier": agent.tier.value, "text": ""}
                self._new_comments_buf.append(brand_buf_entry)
            if agent.persona:
                self._pending_comments += 1
                asyncio.create_task(self._gen_comment(agent, content, event.sim_time, brand_buf_entry))

        elif action == ActionType.REPOST:
            self.metrics["reposts"] += 1
            content.reposts += 1
            self._reposted_by.setdefault(content_id, set()).add(agent_id)
            derived = self._make_derived(agent, content, ContentType.REPOST, event.sim_time)
            self.queue.push(Event(
                sim_time=event.sim_time,
                type=EventType.CONTENT_PUBLISHED,
                payload={"content_id": derived.id, "viral_boost": 1.0},
            ))
            # KOL 转发立即触发一次小范围病毒扩散
            if agent.tier == AgentTier.KOL:
                self._kol_viral_push(derived, event.sim_time)

        seed    = self.seed_content
        new_cid = derived.id if action == ActionType.REPOST else None
        # 找到内容的原始作者，用于高亮传播路径上的边
        source_id   = content.author_id
        return {
            "type": "agent_reacted",
            "sim_time": event.sim_time,
            "agent_id": agent_id,
            "agent_tier": agent.tier.value,
            "action": action.value,
            "content_id": content_id,
            "new_content_id": new_cid,
            "highlight_edge": {"source": source_id, "target": agent_id},
            "metrics": dict(self.metrics),
            "seed_stats": {"likes": self.metrics["likes"], "comments": self.metrics["comments"], "reposts": self.metrics["reposts"]},
        }

    # ── 处理发布（传播扩散）─────────────────────────────────

    def _handle_published(self, event: Event):
        content_id  = event.payload["content_id"]
        viral_boost = event.payload.get("viral_boost", 1.0)
        content = self._cache.get(content_id)
        if content is None:
            return

        author_id = content.author_id
        followers = list(self.graph.predecessors(author_id)) if author_id in self.graph else []
        exposed_set = self._exposed.setdefault(content_id, set())
        new_time = content.sim_time + self._time_delta(len(followers))

        for agent_id in followers:
            if agent_id not in exposed_set:
                exposed_set.add(agent_id)
                self.queue.push(Event(
                    sim_time=new_time,
                    type=EventType.CONTENT_EXPOSED,
                    payload={"agent_id": agent_id, "content_id": content_id},
                ))

        # 出圈扩散：repost 数 >= 3 且 viral_boost >= 1.0
        repost_count = len(self._reposted_by.get(content_id, set()))
        if repost_count >= 3 and viral_boost >= 1.0:
            self._viral_spread(content, exposed_set, repost_count, new_time)

    # ── LLM 评论生成 ──────────────────────────────────────

    async def _gen_comment(self, agent: Agent, content: Content, sim_time: float, brand_entry: dict | None = None):
        try:
            text = await generate_comment(agent.persona, self.seed_content.text)
            if brand_entry is not None:
                brand_entry["text"] = text
            await self._comment_queue.put({
                "type":       "comment_generated",
                "sim_time":   sim_time,
                "agent_id":   agent.id,
                "agent_tier": agent.tier.value,
                "text":       text,
            })
        except Exception:
            pass  # LLM 失败静默跳过，不影响仿真
        finally:
            self._pending_comments -= 1

    # ── 品牌回复 ──────────────────────────────────────────

    async def _run_brand_check(self, comments: list[dict], sim_time: float):
        """从 comments 中过滤掉已回复过的，调用 LLM 决定是否回复"""
        try:
            # 等待一小段时间，让 _gen_comment 任务有机会填充 brand_entry["text"]
            await asyncio.sleep(2.0)
            candidates = [c for c in comments if c["agent_id"] not in self._replied_to and c.get("text")]
            if not candidates:
                return
            replies = await brand_review_comments(
                brand_persona=self.brand_agent.persona(),
                post_text=self.seed_content.text,
                new_comments=candidates,
            )
            for r in replies:
                agent_id = r["agent_id"]
                if agent_id in self._replied_to:
                    continue
                self._replied_to.add(agent_id)
                await self._brand_queue.put({
                    "type":       "brand_reply",
                    "sim_time":   sim_time,
                    "agent_id":   agent_id,
                    "reply":      r["reply"],
                })
        except Exception:
            pass
        finally:
            self._pending_brand -= 1

    # ── 辅助方法 ──────────────────────────────────────────

    def _social_pressure(self, agent: Agent, content_id: str) -> float:
        """
        agent 关注列表中已转发此内容的加权比例。
        KOL 转发权重 = 3，其余 = 1。
        """
        following = list(self.graph.successors(agent.id)) if agent.id in self.graph else []
        if not following:
            return 0.0
        reposted = self._reposted_by.get(content_id, set())
        weighted_reposts = sum(
            3.0 if self.agents[fid].tier == AgentTier.KOL else 1.0
            for fid in following
            if fid in reposted and fid in self.agents
        )
        total_weight = sum(
            3.0 if self.agents[fid].tier == AgentTier.KOL else 1.0
            for fid in following
            if fid in self.agents
        )
        return weighted_reposts / total_weight if total_weight > 0 else 0.0

    def _make_derived(self, agent: Agent, parent: Content, ctype: ContentType, sim_time: float) -> Content:
        noise = np.random.normal(0, 0.05, len(parent.vector))
        new_vector = list(np.clip(np.array(parent.vector) + noise, 0, 1))
        content = Content(
            id=f"post_{uuid.uuid4().hex[:12]}",
            sim_id=self.sim_id,
            author_id=agent.id,
            type=ctype,
            vector=new_vector,
            parent_id=parent.id,
            sim_time=sim_time,
        )
        self._cache[content.id] = content
        return content

    def _kol_viral_push(self, content: Content, sim_time: float, n: int = 5):
        """KOL 转发后立即向最相似的 n 个非曝光 agent 推送"""
        exposed_set = self._exposed.setdefault(content.id, set())
        candidates = [a for a_id, a in self.agents.items() if a_id not in exposed_set]
        if not candidates:
            return
        scores = np.array([cosine_similarity(a.vector, content.vector) for a in candidates])
        scores = np.clip(scores, 0, None)
        total = scores.sum()
        if total == 0:
            return
        chosen = np.random.choice(
            candidates,
            size=min(n, len(candidates)),
            replace=False,
            p=scores / total,
        )
        for a in chosen:
            exposed_set.add(a.id)
            self.queue.push(Event(
                sim_time=sim_time + 0.1,
                type=EventType.CONTENT_EXPOSED,
                payload={"agent_id": a.id, "content_id": content.id},
            ))

    def _viral_spread(self, content: Content, exposed_set: set, repost_count: int, base_time: float):
        """repost 数超阈值后向相似 agent 出圈扩散"""
        n_extra = min(repost_count * 2, 20)
        candidates = [a for a_id, a in self.agents.items() if a_id not in exposed_set]
        if not candidates:
            return
        scores = np.array([cosine_similarity(a.vector, content.vector) for a in candidates])
        scores = np.clip(scores, 0, None)
        total = scores.sum()
        if total == 0:
            return
        chosen = np.random.choice(
            candidates,
            size=min(n_extra, len(candidates)),
            replace=False,
            p=scores / total,
        )
        for a in chosen:
            exposed_set.add(a.id)
            self.queue.push(Event(
                sim_time=base_time + self._time_delta(repost_count),
                type=EventType.CONTENT_EXPOSED,
                payload={"agent_id": a.id, "content_id": content.id},
            ))

    @staticmethod
    def _time_delta(n_followers: int) -> float:
        """粉丝越多传播越快（单位：天）"""
        return max(0.01, 0.5 / math.log(n_followers + 2))

    def _graph_init_event(self) -> dict:
        """推送完整的节点列表和边列表，供前端初始化静态图"""
        nodes = [{"id": a_id, "tier": a.tier.value, "persona": a.persona} for a_id, a in self.agents.items()]
        nodes.append({"id": "brand_001", "tier": "brand", "persona": ""})
        edges = [
            {"source": u, "target": v}
            for u, v in self.graph.edges()
        ]
        return {"type": "graph_init", "nodes": nodes, "edges": edges}

    def _done_event(self) -> dict:
        reach = max(self.metrics["reach"], 1)
        return {
            "type": "simulation_done",
            "total_events": self.event_count,
            "metrics": dict(self.metrics),
            "repost_rate":  round(self.metrics["reposts"]  / reach, 4),
            "comment_rate": round(self.metrics["comments"] / reach, 4),
        }

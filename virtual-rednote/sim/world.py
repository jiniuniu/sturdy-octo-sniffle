"""
世界生成：Agent 生成 + 社交图生成
向量采样使用 Sobol 低差异序列 + Beta 分位映射，保证多维空间均匀覆盖。
"""

import uuid

import networkx as nx
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats.qmc import Sobol

from .models import Agent, AgentTier, Dimension

# 各 tier 的行为参数范围 (min, max) 和向量采样 Beta 参数
# KOL：在某些维度上偏高（专业人设）；普通用户偏均匀
TIER_CONFIG = {
    AgentTier.KOL: {
        "activity": (0.7, 1.0),
        "expressiveness": (0.7, 1.0),
        "sharing": (0.6, 1.0),
        "vec_alpha": 5.0,
        "vec_beta": 2.0,  # 偏高端，体现专业偏好
    },
    AgentTier.KOC: {
        "activity": (0.4, 0.8),
        "expressiveness": (0.4, 0.8),
        "sharing": (0.3, 0.7),
        "vec_alpha": 3.0,
        "vec_beta": 2.0,  # 有偏好但不极端
    },
    AgentTier.NORMAL: {
        "activity": (0.1, 0.5),
        "expressiveness": (0.1, 0.4),
        "sharing": (0.05, 0.3),
        "vec_alpha": 2.0,
        "vec_beta": 2.0,  # 均匀分布，无强烈偏好
    },
}


def generate_agents(
    world_id: str,
    dimensions: list[Dimension],
    n_total: int = 200,
    seed: int = 42,
) -> list[Agent]:
    """按比例生成 agents：5% KOL, 15% KOC, 80% normal"""
    rng = np.random.default_rng(seed)
    n_kol = max(1, int(n_total * 0.05))
    n_koc = max(1, int(n_total * 0.15))
    n_normal = n_total - n_kol - n_koc

    n_dims = len(dimensions)
    agents = []
    idx = 0

    for tier, count in [
        (AgentTier.KOL, n_kol),
        (AgentTier.KOC, n_koc),
        (AgentTier.NORMAL, n_normal),
    ]:
        cfg = TIER_CONFIG[tier]

        # Sobol 采样 [0,1]^n_dims，再用 Beta 分位函数映射，保证低差异均匀覆盖
        sobol = Sobol(d=n_dims, scramble=True, seed=seed + idx)
        u = sobol.random(count)  # shape (count, n_dims)
        vectors = beta_dist.ppf(
            np.clip(u, 1e-6, 1 - 1e-6),  # 避免 ppf(0)/ppf(1) 溢出
            a=cfg["vec_alpha"],
            b=cfg["vec_beta"],
        )  # shape (count, n_dims)

        # 行为参数仍用普通随机（标量，不需要低差异覆盖）
        activities = rng.uniform(*cfg["activity"], size=count)
        expressivenesses = rng.uniform(*cfg["expressiveness"], size=count)
        sharings = rng.uniform(*cfg["sharing"], size=count)

        for k in range(count):
            agents.append(
                Agent(
                    id=f"agent_{idx:04d}",
                    world_id=world_id,
                    tier=tier,
                    vector=[float(v) for v in vectors[k]],
                    activity=float(activities[k]),
                    expressiveness=float(expressivenesses[k]),
                    sharing=float(sharings[k]),
                )
            )
            idx += 1

    return agents


def generate_social_graph(agents: list[Agent], seed: int = 42) -> nx.DiGraph:
    """
    幂律分布社交图：
    - 所有 KOC 关注所有 KOL
    - 普通用户随机关注少量 KOL/KOC + 少量普通用户
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    for agent in agents:
        G.add_node(agent.id, tier=agent.tier.value)

    kols = [a for a in agents if a.tier == AgentTier.KOL]
    kocs = [a for a in agents if a.tier == AgentTier.KOC]
    normals = [a for a in agents if a.tier == AgentTier.NORMAL]

    def add_follow(follower: Agent, followee: Agent):
        if follower.id == followee.id:
            return
        G.add_edge(follower.id, followee.id)
        follower.following.append(followee.id)
        followee.followers.append(follower.id)

    # KOC 全部关注 KOL
    for koc in kocs:
        for kol in kols:
            add_follow(koc, kol)

    # 普通用户：随机关注 1~3 个 KOL，2~5 个 KOC，0~3 个普通用户
    for normal in normals:
        for kol in rng.choice(
            kols, size=min(rng.integers(1, 4), len(kols)), replace=False
        ):
            add_follow(normal, kol)
        for koc in rng.choice(
            kocs, size=min(rng.integers(2, 6), len(kocs)), replace=False
        ):
            add_follow(normal, koc)
        peers = [a for a in normals if a.id != normal.id]
        if peers:
            for peer in rng.choice(
                peers, size=min(rng.integers(0, 4), len(peers)), replace=False
            ):
                add_follow(normal, peer)

    return G


def create_world(
    dimensions: list[Dimension],
    n_agents: int = 200,
    seed: int = 42,
) -> tuple[str, list[Agent], nx.DiGraph]:
    """给定维度，生成 world_id + agents + 社交图"""
    world_id = f"world_{uuid.uuid4().hex[:8]}"
    agents = generate_agents(world_id, dimensions, n_total=n_agents, seed=seed)
    graph = generate_social_graph(agents, seed=seed)
    return world_id, agents, graph

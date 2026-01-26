"""
社交网络拓扑构建 - 基于 BA 无标度网络模型
"""

import random
from dataclasses import dataclass, field

import networkx as nx
from config import settings


@dataclass
class NetworkTopology:
    """网络拓扑结构"""

    graph: nx.DiGraph
    user_ids: list[str]
    brand_account_id: str

    # 缓存的统计数据
    _follower_counts: dict[str, int] = field(default_factory=dict)
    _follower_map: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后计算统计数据"""
        self._compute_stats()

    def _compute_stats(self):
        """计算粉丝数和粉丝列表"""
        for node in self.graph.nodes():
            # in_edges: 谁关注了这个节点（即粉丝）
            followers = [e[0] for e in self.graph.in_edges(node)]
            self._follower_map[node] = followers
            self._follower_counts[node] = len(followers)

    def get_follower_count(self, user_id: str) -> int:
        """获取用户的粉丝数"""
        return self._follower_counts.get(user_id, 0)

    def get_followers(self, user_id: str) -> list[str]:
        """获取用户的粉丝列表"""
        return self._follower_map.get(user_id, [])

    def get_following(self, user_id: str) -> list[str]:
        """获取用户关注的人"""
        return [e[1] for e in self.graph.out_edges(user_id)]

    @property
    def follower_counts(self) -> dict[str, int]:
        """所有用户的粉丝数"""
        return self._follower_counts.copy()

    @property
    def follower_map(self) -> dict[str, list[str]]:
        """所有用户的粉丝列表"""
        return {k: v.copy() for k, v in self._follower_map.items()}


class NetworkBuilder:
    """网络拓扑构建器"""

    def __init__(
        self,
        ba_m: int = None,
        brand_follow_ratio: float = None,
    ):
        """
        Args:
            ba_m: BA模型参数，每个新节点连接的边数
            brand_follow_ratio: 关注品牌账号的用户比例
        """
        self.ba_m = ba_m or settings.ba_model_m
        self.brand_follow_ratio = brand_follow_ratio or settings.brand_follow_ratio

    def build(self, user_count: int) -> NetworkTopology:
        """
        构建社交网络拓扑

        Args:
            user_count: 用户数量（不含品牌账号）

        Returns:
            NetworkTopology 对象
        """
        # 1. 使用 BA 模型生成无标度网络
        ba_graph = nx.barabasi_albert_graph(n=user_count, m=self.ba_m)

        # 2. 转换为有向图（关注关系）
        # 无向边 (a, b) 转换为双向关注或单向关注
        directed = nx.DiGraph()

        for u, v in ba_graph.edges():
            # 随机决定关注方向（60% 双向，40% 单向）
            if random.random() < 0.6:
                directed.add_edge(u, v)
                directed.add_edge(v, u)
            else:
                # 单向：度数高的更可能被关注
                if ba_graph.degree(u) > ba_graph.degree(v):
                    directed.add_edge(v, u)
                else:
                    directed.add_edge(u, v)

        # 确保所有节点都在图中
        for i in range(user_count):
            if i not in directed:
                directed.add_node(i)

        # 3. 生成用户ID
        user_ids = [f"user_{i:04d}" for i in range(user_count)]

        # 4. 重新标记节点
        mapping = {i: user_ids[i] for i in range(user_count)}
        directed = nx.relabel_nodes(directed, mapping)

        # 5. 添加品牌账号
        brand_id = "brand_official"
        directed.add_node(brand_id)

        # 6. 部分用户关注品牌账号
        follow_count = int(user_count * self.brand_follow_ratio)
        followers = random.sample(user_ids, follow_count)
        for uid in followers:
            directed.add_edge(uid, brand_id)  # uid 关注 brand

        return NetworkTopology(
            graph=directed,
            user_ids=user_ids,
            brand_account_id=brand_id,
        )

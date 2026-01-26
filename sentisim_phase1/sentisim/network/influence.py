"""
影响力角色分配 - 基于网络结构
"""

from config import settings
from sentisim.models import InfluenceLevel
from sentisim.network.topology import NetworkTopology


class InfluenceAssigner:
    """影响力角色分配器"""

    def __init__(
        self,
        kol_percentile: float = None,
        active_percentile: float = None,
        normal_percentile: float = None,
    ):
        """
        Args:
            kol_percentile: KOL 占比阈值（前 x%）
            active_percentile: 活跃用户占比阈值
            normal_percentile: 普通用户占比阈值
        """
        self.kol_percentile = kol_percentile or settings.kol_percentile
        self.active_percentile = active_percentile or settings.active_percentile
        self.normal_percentile = normal_percentile or settings.normal_percentile

    def assign(self, topology: NetworkTopology) -> dict[str, InfluenceLevel]:
        """
        根据网络拓扑分配影响力角色

        Args:
            topology: 网络拓扑

        Returns:
            用户ID -> 影响力层级 的映射
        """
        # 获取所有用户的粉丝数（不含品牌账号）
        follower_counts = {
            uid: topology.get_follower_count(uid) for uid in topology.user_ids
        }

        # 按粉丝数降序排列
        sorted_users = sorted(follower_counts.items(), key=lambda x: x[1], reverse=True)

        total = len(sorted_users)
        influence_levels = {}

        for rank, (user_id, _) in enumerate(sorted_users):
            percentile = rank / total

            if percentile < self.kol_percentile:
                level = InfluenceLevel.KOL
            elif percentile < self.active_percentile:
                level = InfluenceLevel.ACTIVE
            elif percentile < self.normal_percentile:
                level = InfluenceLevel.NORMAL
            else:
                level = InfluenceLevel.SILENT

            influence_levels[user_id] = level

        return influence_levels

    def get_stats(self, influence_levels: dict[str, InfluenceLevel]) -> dict:
        """
        获取影响力分布统计

        Args:
            influence_levels: 影响力分配结果

        Returns:
            统计数据
        """
        total = len(influence_levels)
        counts = {level: 0 for level in InfluenceLevel}

        for level in influence_levels.values():
            counts[level] += 1

        return {
            "total": total,
            "counts": {level.value: count for level, count in counts.items()},
            "percentages": {
                level.value: round(count / total * 100, 1)
                for level, count in counts.items()
            },
        }

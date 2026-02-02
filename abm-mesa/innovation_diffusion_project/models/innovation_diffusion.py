"""
消费者从众行为模型 - 基于创新扩散理论
使用小世界网络模拟社交网络中的消费行为传播

创新扩散理论的五类消费者：
1. Innovators (创新者) - 2.5%
2. Early Adopters (早期采纳者) - 13.5%
3. Early Majority (早期大众) - 34%
4. Late Majority (晚期大众) - 34%
5. Laggards (落后者) - 16%

每类消费者有不同的从众阈值分布
"""

from enum import Enum

import mesa
import networkx as nx
import numpy as np


class BuyingState(Enum):
    """购买状态枚举"""

    NOT_BOUGHT = 0  # 未购买
    BOUGHT = 1  # 已购买


class ConsumerType(Enum):
    """消费者类型（基于创新扩散理论）"""

    INNOVATOR = 0  # 创新者
    EARLY_ADOPTER = 1  # 早期采纳者
    EARLY_MAJORITY = 2  # 早期大众
    LATE_MAJORITY = 3  # 晚期大众
    LAGGARD = 4  # 落后者


# 创新扩散理论的配置
DIFFUSION_CONFIG = {
    ConsumerType.INNOVATOR: {
        "proportion": 0.025,  # 2.5%
        "threshold_mean": 0.1,  # 低阈值：几乎不需要他人影响
        "threshold_std": 0.05,
        "color": "#FF0000",  # 红色
        "label": "创新者",
    },
    ConsumerType.EARLY_ADOPTER: {
        "proportion": 0.135,  # 13.5%
        "threshold_mean": 0.25,  # 较低阈值
        "threshold_std": 0.08,
        "color": "#FF6B00",  # 橙红色
        "label": "早期采纳者",
    },
    ConsumerType.EARLY_MAJORITY: {
        "proportion": 0.34,  # 34%
        "threshold_mean": 0.45,  # 中等阈值
        "threshold_std": 0.1,
        "color": "#FFD700",  # 金色
        "label": "早期大众",
    },
    ConsumerType.LATE_MAJORITY: {
        "proportion": 0.34,  # 34%
        "threshold_mean": 0.65,  # 较高阈值
        "threshold_std": 0.1,
        "color": "#4ECDC4",  # 青色
        "label": "晚期大众",
    },
    ConsumerType.LAGGARD: {
        "proportion": 0.16,  # 16%
        "threshold_mean": 0.85,  # 高阈值：需要大多数人购买才会跟随
        "threshold_std": 0.08,
        "color": "#95A5A6",  # 灰色
        "label": "落后者",
    },
}


class ConsumerAgent(mesa.Agent):
    """消费者 Agent - 基于创新扩散理论"""

    def __init__(self, model, consumer_type):
        super().__init__(model)

        # 消费者类型
        self.consumer_type = consumer_type

        # 初始状态：未购买
        self.state = BuyingState.NOT_BOUGHT

        # 根据消费者类型设置从众阈值（正态分布采样）
        config = DIFFUSION_CONFIG[consumer_type]
        self.conformity_threshold = np.clip(
            self.random.gauss(config["threshold_mean"], config["threshold_std"]),
            0.0,  # 最小值
            1.0,  # 最大值
        )

    def step(self):
        """每一步的行为"""
        # 如果已经购买了，就不再改变
        if self.state == BuyingState.BOUGHT:
            return

        # 获取邻居 Agents
        neighbor_agents = self.model.grid.get_neighbors(self.pos, include_center=False)

        if len(neighbor_agents) > 0:
            # 统计邻居中已购买的数量
            bought_neighbors = sum(
                1 for agent in neighbor_agents if agent.state == BuyingState.BOUGHT
            )

            # 计算购买比例
            buying_ratio = bought_neighbors / len(neighbor_agents)

            # 从众决策：如果邻居中购买的比例超过阈值，则购买
            if buying_ratio >= self.conformity_threshold:
                self.state = BuyingState.BOUGHT


class InnovationDiffusionModel(mesa.Model):
    """基于创新扩散理论的消费者从众行为模型"""

    def __init__(
        self,
        n_consumers=100,
        k_neighbors=4,
        p_rewire=0.3,
        initial_buyers_pct=0.05,  # 初始购买者比例（仅限早期采纳者）
        seed=None,
    ):
        """
        初始化模型

        参数:
            n_consumers: 消费者总数
            k_neighbors: 小世界网络中每个节点的邻居数
            p_rewire: 重连概率
            initial_buyers_pct: 初始购买者比例（仅从早期采纳者中选择）
            seed: 随机种子
        """
        super().__init__(seed=seed)

        self.n_consumers = n_consumers
        self.k_neighbors = k_neighbors
        self.p_rewire = p_rewire
        self.initial_buyers_pct = initial_buyers_pct

        # 创建小世界网络
        self.G = nx.watts_strogatz_graph(
            n=self.n_consumers,
            k=self.k_neighbors,
            p=self.p_rewire,
            seed=self.random.randint(0, 10000),
        )

        # 创建网络空间
        self.grid = mesa.space.NetworkGrid(self.G)

        # 计算每种类型的消费者数量
        type_counts = {}
        for consumer_type in ConsumerType:
            count = int(n_consumers * DIFFUSION_CONFIG[consumer_type]["proportion"])
            type_counts[consumer_type] = count

        # 调整以确保总数正确
        total_assigned = sum(type_counts.values())
        if total_assigned < n_consumers:
            # 将剩余的分配给早期大众
            type_counts[ConsumerType.EARLY_MAJORITY] += n_consumers - total_assigned

        # 创建消费者类型列表
        consumer_types = []
        for consumer_type, count in type_counts.items():
            consumer_types.extend([consumer_type] * count)

        # 打乱顺序
        self.random.shuffle(consumer_types)

        # 创建消费者agents
        self.consumer_groups = {t: [] for t in ConsumerType}  # 改名避免冲突
        for i, node in enumerate(self.G.nodes()):
            agent = ConsumerAgent(self, consumer_types[i])
            self.grid.place_agent(agent, node)
            self.consumer_groups[agent.consumer_type].append(agent)

        # 设置初始购买者（仅从创新者和早期采纳者中选择）
        # 优先选择创新者，然后是早期采纳者
        potential_initial_buyers = (
            self.consumer_groups[ConsumerType.INNOVATOR]
            + self.consumer_groups[ConsumerType.EARLY_ADOPTER]
        )

        # 计算初始购买者数量，但不超过早期采纳者的总数
        max_initial_buyers = len(potential_initial_buyers)
        n_initial_buyers = min(
            int(self.n_consumers * self.initial_buyers_pct), max_initial_buyers
        )

        # 随机选择初始购买者
        if n_initial_buyers > 0:
            initial_buyers = self.random.sample(
                potential_initial_buyers, n_initial_buyers
            )
            for agent in initial_buyers:
                agent.state = BuyingState.BOUGHT

        # 创建数据收集器
        self.datacollector = mesa.DataCollector(
            model_reporters={
                # 总体统计
                "Buyers": lambda m: self.count_state(m, BuyingState.BOUGHT),
                "Non-Buyers": lambda m: self.count_state(m, BuyingState.NOT_BOUGHT),
                "Adoption_Rate": lambda m: self.count_state(m, BuyingState.BOUGHT)
                / m.n_consumers,
                # 各类型购买者数量（绝对值）
                "Innovators_Bought": lambda m: self.count_type_state(
                    m, ConsumerType.INNOVATOR, BuyingState.BOUGHT
                ),
                "Early_Adopters_Bought": lambda m: self.count_type_state(
                    m, ConsumerType.EARLY_ADOPTER, BuyingState.BOUGHT
                ),
                "Early_Majority_Bought": lambda m: self.count_type_state(
                    m, ConsumerType.EARLY_MAJORITY, BuyingState.BOUGHT
                ),
                "Late_Majority_Bought": lambda m: self.count_type_state(
                    m, ConsumerType.LATE_MAJORITY, BuyingState.BOUGHT
                ),
                "Laggards_Bought": lambda m: self.count_type_state(
                    m, ConsumerType.LAGGARD, BuyingState.BOUGHT
                ),
                # 各类型采纳率（比例）
                "Innovators_Rate": lambda m: (
                    self.count_type_state(m, ConsumerType.INNOVATOR, BuyingState.BOUGHT)
                    / len(m.consumer_groups[ConsumerType.INNOVATOR])
                    if m.consumer_groups[ConsumerType.INNOVATOR]
                    else 0
                ),
                "Early_Adopters_Rate": lambda m: (
                    self.count_type_state(
                        m, ConsumerType.EARLY_ADOPTER, BuyingState.BOUGHT
                    )
                    / len(m.consumer_groups[ConsumerType.EARLY_ADOPTER])
                    if m.consumer_groups[ConsumerType.EARLY_ADOPTER]
                    else 0
                ),
                "Early_Majority_Rate": lambda m: (
                    self.count_type_state(
                        m, ConsumerType.EARLY_MAJORITY, BuyingState.BOUGHT
                    )
                    / len(m.consumer_groups[ConsumerType.EARLY_MAJORITY])
                    if m.consumer_groups[ConsumerType.EARLY_MAJORITY]
                    else 0
                ),
                "Late_Majority_Rate": lambda m: (
                    self.count_type_state(
                        m, ConsumerType.LATE_MAJORITY, BuyingState.BOUGHT
                    )
                    / len(m.consumer_groups[ConsumerType.LATE_MAJORITY])
                    if m.consumer_groups[ConsumerType.LATE_MAJORITY]
                    else 0
                ),
                "Laggards_Rate": lambda m: (
                    self.count_type_state(m, ConsumerType.LAGGARD, BuyingState.BOUGHT)
                    / len(m.consumer_groups[ConsumerType.LAGGARD])
                    if m.consumer_groups[ConsumerType.LAGGARD]
                    else 0
                ),
            }
        )

        self.running = True
        self.datacollector.collect(self)

    @staticmethod
    def count_state(model, state):
        """统计特定状态的agent数量"""
        return sum(1 for agent in model.agents if agent.state == state)

    @staticmethod
    def count_type_state(model, consumer_type, state):
        """统计特定类型且特定状态的agent数量"""
        return sum(
            1
            for agent in model.agents
            if agent.consumer_type == consumer_type and agent.state == state
        )

    def step(self):
        """模型的一步"""
        # 随机激活所有agents
        self.agents.shuffle_do("step")

        # 收集数据
        self.datacollector.collect(self)

        # 检查是否所有人都购买了
        if self.count_state(self, BuyingState.BOUGHT) == self.n_consumers:
            self.running = False


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 测试模型
    model = InnovationDiffusionModel(n_consumers=50, seed=42)
    print(f"模型创建成功，消费者数量: {len(model.agents)}")

"""
场景2：高密度网络
"""
from .scenario_base import ScenarioConfig


high_density_scenario = ScenarioConfig(
    name="场景2：高密度网络",
    description="高重连概率和更多邻居，模拟信息传播快速的社交网络",
    params={
        "n_consumers": 100,
        "k_neighbors": 8,       # 更多邻居
        "p_rewire": 0.8,        # 高重连概率 → 接近随机网络
        "initial_buyers_pct": 0.05,
    },
    hypothesis="高密度网络将加速创新扩散，缩短达到临界质量的时间",
    expected_outcome="预期：扩散速度显著快于默认场景，S 曲线更陡峭"
)

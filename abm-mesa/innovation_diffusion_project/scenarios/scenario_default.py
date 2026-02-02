"""
场景1：默认参数（基准场景）
"""
from .scenario_base import ScenarioConfig


default_scenario = ScenarioConfig(
    name="场景1：默认参数",
    description="标准的小世界网络参数，作为对比基准",
    params={
        "n_consumers": 100,
        "k_neighbors": 4,
        "p_rewire": 0.3,
        "initial_buyers_pct": 0.05,
    },
    hypothesis="在标准小世界网络中，创新扩散将呈现经典的 S 曲线",
    expected_outcome="预期：创新者和早期采纳者最先采纳，随后是早期大众，形成完整的扩散过程"
)

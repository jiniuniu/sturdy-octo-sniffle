"""
场景3：低初始购买者
"""
from .scenario_base import ScenarioConfig


low_initial_scenario = ScenarioConfig(
    name="场景3：低初始购买者",
    description="极低的初始购买者比例，测试临界质量的重要性",
    params={
        "n_consumers": 100,
        "k_neighbors": 4,
        "p_rewire": 0.3,
        "initial_buyers_pct": 0.01,  # 仅 1% (只有1个人)
    },
    hypothesis="初始购买者过少可能无法形成临界质量，导致扩散停滞",
    expected_outcome="预期：扩散可能停滞或非常缓慢，难以达到早期大众"
)

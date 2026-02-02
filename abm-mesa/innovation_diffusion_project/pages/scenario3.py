"""
场景3页面：低初始购买者
"""
import solara
from scenarios import low_initial_scenario
from components import ScenarioInfo, ModelVisualization


@solara.component
def Page():
    """场景3页面组件"""
    
    with solara.Column(gap="20px"):
        # 场景信息
        ScenarioInfo(low_initial_scenario)
        
        # 模型可视化
        ModelVisualization(
            scenario_config=low_initial_scenario,
            show_network=True,
            show_type_rates=True,
            show_total_rate=True
        )

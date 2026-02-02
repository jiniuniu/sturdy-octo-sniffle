"""
场景1页面：默认参数
"""
import solara
from scenarios import default_scenario
from components import ScenarioInfo, ModelVisualization


@solara.component
def Page():
    """场景1页面组件"""
    
    with solara.Column(gap="20px"):
        # 场景信息
        ScenarioInfo(default_scenario)
        
        # 模型可视化
        ModelVisualization(
            scenario_config=default_scenario,
            show_network=True,
            show_type_rates=True,
            show_total_rate=True
        )

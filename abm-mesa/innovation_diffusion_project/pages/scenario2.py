"""
场景2页面：高密度网络
"""
import solara
from scenarios import high_density_scenario
from components import ScenarioInfo, ModelVisualization


@solara.component
def Page():
    """场景2页面组件"""
    
    with solara.Column(gap="20px"):
        # 场景信息
        ScenarioInfo(high_density_scenario)
        
        # 模型可视化
        ModelVisualization(
            scenario_config=high_density_scenario,
            show_network=True,
            show_type_rates=True,
            show_total_rate=True
        )

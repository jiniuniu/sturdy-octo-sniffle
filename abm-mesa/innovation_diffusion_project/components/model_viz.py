"""
可复用的模型可视化组件
"""
import solara
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from models import InnovationDiffusionModel
from utils import agent_portrayal


@solara.component
def ModelVisualization(
    scenario_config,
    show_network=True,
    show_type_rates=True,
    show_total_rate=True
):
    """
    模型可视化组件
    
    Args:
        scenario_config: ScenarioConfig 实例
        show_network: 是否显示网络图
        show_type_rates: 是否显示各类型采纳率图表
        show_total_rate: 是否显示总体采纳率图表
    """
    
    # 创建模型实例
    model = InnovationDiffusionModel(**scenario_config.get_model_params())
    
    # 创建渲染器
    renderer = SpaceRenderer(model, backend="matplotlib").render(
        agent_portrayal=agent_portrayal
    ) if show_network else None
    
    # 创建图表组件列表
    components = []
    
    # 总体采纳情况（绝对数量）
    total_chart = make_plot_component(["Buyers", "Non-Buyers"])
    components.append(total_chart)
    
    # 各类型采纳率（比例）
    if show_type_rates:
        types_rate_chart = make_plot_component([
            "Innovators_Rate",
            "Early_Adopters_Rate", 
            "Early_Majority_Rate",
            "Late_Majority_Rate",
            "Laggards_Rate"
        ])
        components.append(types_rate_chart)
    
    # 总体采纳率
    if show_total_rate:
        rate_chart = make_plot_component("Adoption_Rate")
        components.append(rate_chart)
    
    # 创建 SolaraViz
    page = SolaraViz(
        model,
        renderer,
        components=components,
        model_params=scenario_config.get_model_params_config(),
        name=scenario_config.name,
    )
    
    return page


@solara.component
def ScenarioInfo(scenario_config):
    """
    显示场景信息卡片
    
    Args:
        scenario_config: ScenarioConfig 实例
    """
    with solara.Card():
        solara.Markdown(f"## {scenario_config.name}")
        solara.Markdown(f"**描述**: {scenario_config.description}")
        
        if scenario_config.hypothesis:
            solara.Markdown(f"**假设**: {scenario_config.hypothesis}")
        
        if scenario_config.expected_outcome:
            solara.Markdown(f"**预期**: {scenario_config.expected_outcome}")
        
        # 显示参数
        solara.Markdown("### 参数配置")
        params = scenario_config.get_model_params()
        param_text = "\n".join([
            f"- **{key}**: {value}" 
            for key, value in params.items()
        ])
        solara.Markdown(param_text)

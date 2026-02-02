"""
场景对比页面：并排显示多个场景
"""
import solara
from mesa.visualization import SpaceRenderer, make_plot_component
from models import InnovationDiffusionModel
from scenarios import default_scenario, high_density_scenario, low_initial_scenario
from utils import agent_portrayal


@solara.component
def CompactScenarioViz(scenario_config):
    """
    紧凑的场景可视化（用于对比）
    
    Args:
        scenario_config: ScenarioConfig 实例
    """
    # 创建模型
    model = InnovationDiffusionModel(**scenario_config.get_model_params())
    
    # 只显示网络图和总体采纳率
    renderer = SpaceRenderer(model, backend="matplotlib").render(
        agent_portrayal=agent_portrayal
    )
    
    rate_chart = make_plot_component("Adoption_Rate")
    
    from mesa.visualization import SolaraViz
    page = SolaraViz(
        model,
        renderer,
        components=[rate_chart],
        model_params=scenario_config.get_model_params_config(),
        name=scenario_config.name,
    )
    
    return page


@solara.component
def Page():
    """场景对比页面组件"""
    
    # 标题
    solara.Markdown("# 场景对比")
    solara.Markdown("并排对比不同场景的扩散模式")
    
    # 选择对比模式
    comparison_mode = solara.use_reactive("两场景对比")
    
    with solara.Card():
        solara.Select(
            label="对比模式",
            value=comparison_mode,
            values=["两场景对比", "三场景对比"]
        )
    
    if comparison_mode.value == "两场景对比":
        # 两场景对比：左右布局
        with solara.Row():
            # 左侧：默认场景
            with solara.Column():
                solara.Markdown("### 场景1：默认参数")
                solara.Markdown(f"*{default_scenario.description}*")
                CompactScenarioViz(default_scenario)
            
            # 右侧：高密度场景
            with solara.Column():
                solara.Markdown("### 场景2：高密度网络")
                solara.Markdown(f"*{high_density_scenario.description}*")
                CompactScenarioViz(high_density_scenario)
    
    else:
        # 三场景对比：网格布局
        with solara.Columns([4, 4, 4]):
            # 场景1
            with solara.Column():
                solara.Markdown("### 场景1")
                solara.Markdown(f"*{default_scenario.description}*")
                CompactScenarioViz(default_scenario)
            
            # 场景2
            with solara.Column():
                solara.Markdown("### 场景2")
                solara.Markdown(f"*{high_density_scenario.description}*")
                CompactScenarioViz(high_density_scenario)
            
            # 场景3
            with solara.Column():
                solara.Markdown("### 场景3")
                solara.Markdown(f"*{low_initial_scenario.description}*")
                CompactScenarioViz(low_initial_scenario)
    
    # 对比说明
    with solara.Card():
        solara.Markdown("""
## 💡 对比要点

观察不同场景的：

1. **扩散速度**：哪个场景更快达到 50% 采纳率？
2. **S 曲线形态**：曲线的陡峭程度如何？
3. **最终采纳率**：是否所有场景都能达到 100%？
4. **网络结构差异**：观察节点连接的密集程度

### 预期对比结果

- **场景2（高密度）** 应该比场景1更快扩散
- **场景3（低初始）** 可能出现扩散停滞
- 场景1作为基准，处于两者之间
        """)

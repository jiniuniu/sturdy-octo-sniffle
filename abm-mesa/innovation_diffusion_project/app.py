"""
创新扩散理论 - 多场景 ABM 模拟
主入口和路由配置

运行方式:
    solara run app.py

然后访问:
    http://localhost:8765/
"""
import solara
from pages import home, scenario1, scenario2, scenario3, comparison


# 定义路由
routes = [
    solara.Route(path="/", component=home.Page, label="首页"),
    solara.Route(path="scenario1", component=scenario1.Page, label="场景1"),
    solara.Route(path="scenario2", component=scenario2.Page, label="场景2"),
    solara.Route(path="scenario3", component=scenario3.Page, label="场景3"),
    solara.Route(path="comparison", component=comparison.Page, label="场景对比"),
]


@solara.component
def Layout(children):
    """
    全局布局组件
    
    提供顶部导航栏和侧边栏
    """
    route, set_route = solara.use_route()
    
    with solara.AppLayout():
        # 顶部导航栏
        with solara.AppBar():
            solara.HTML(tag="h3", unsafe_innerHTML="🔬 创新扩散 ABM")
            
            # 导航按钮
            with solara.Row(gap="10px", style={"margin-left": "auto"}):
                solara.Button(
                    "首页",
                    on_click=lambda: set_route("/"),
                    color="primary" if route.path == "/" else None
                )
                solara.Button(
                    "场景1",
                    on_click=lambda: set_route("scenario1"),
                    color="primary" if route.path == "scenario1" else None
                )
                solara.Button(
                    "场景2",
                    on_click=lambda: set_route("scenario2"),
                    color="primary" if route.path == "scenario2" else None
                )
                solara.Button(
                    "场景3",
                    on_click=lambda: set_route("scenario3"),
                    color="primary" if route.path == "scenario3" else None
                )
                solara.Button(
                    "对比",
                    on_click=lambda: set_route("comparison"),
                    color="primary" if route.path == "comparison" else None
                )
        
        # 侧边栏
        with solara.Sidebar():
            solara.Markdown("## 📋 导航")
            
            with solara.Column(gap="5px"):
                solara.Button(
                    "🏠 首页",
                    on_click=lambda: set_route("/"),
                    block=True,
                    text=True
                )
                
                solara.Markdown("### 实验场景")
                
                solara.Button(
                    "📊 场景1：默认参数",
                    on_click=lambda: set_route("scenario1"),
                    block=True,
                    text=True
                )
                solara.Button(
                    "🌐 场景2：高密度网络",
                    on_click=lambda: set_route("scenario2"),
                    block=True,
                    text=True
                )
                solara.Button(
                    "🔻 场景3：低初始购买者",
                    on_click=lambda: set_route("scenario3"),
                    block=True,
                    text=True
                )
                solara.Button(
                    "⚖️ 场景对比",
                    on_click=lambda: set_route("comparison"),
                    block=True,
                    text=True
                )
            
            solara.Markdown("---")
            solara.Markdown("""
### 💡 使用提示

1. 选择场景查看模拟
2. 使用 ▶️ 按钮运行模型
3. 观察网络图和曲线变化
4. 调整参数进行实验
            """)
        
        # 主内容区
        with solara.Column(style={"padding": "20px"}):
            solara.Column(children=children)


# Solara 会自动使用这个 Layout

# 创新扩散理论 - 多场景 ABM 模拟

基于 **Everett Rogers 创新扩散理论**的 Agent-Based Modeling (ABM) 项目，使用 Mesa 框架和 Solara 可视化。

## 📁 项目结构

```
innovation_diffusion_project/
│
├── app.py                          # 主入口（Solara 路由）
│
├── models/                          # 模型定义
│   ├── __init__.py
│   └── innovation_diffusion.py     # 创新扩散模型
│
├── scenarios/                       # 场景配置
│   ├── __init__.py
│   ├── scenario_base.py            # 场景基类
│   ├── scenario_default.py         # 场景1：默认参数
│   ├── scenario_high_density.py    # 场景2：高密度网络
│   └── scenario_low_initial.py     # 场景3：低初始购买者
│
├── pages/                           # Solara 页面（路由）
│   ├── __init__.py
│   ├── home.py                     # 首页
│   ├── scenario1.py                # 场景1页面
│   ├── scenario2.py                # 场景2页面
│   ├── scenario3.py                # 场景3页面
│   └── comparison.py               # 对比页面
│
├── components/                      # 可视化组件
│   ├── __init__.py
│   └── model_viz.py                # 模型可视化组件
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   └── agent_portrayal.py          # Agent 绘制函数
│
└── README.md                        # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install mesa solara networkx numpy pandas matplotlib
```

或者：

```bash
pip install -U "mesa[rec]"  # 包含所有推荐依赖
```

### 2. 运行应用

```bash
cd innovation_diffusion_project
solara run app.py
```

### 3. 访问应用

打开浏览器访问: http://localhost:8765/

## 📊 场景说明

### 场景1：默认参数（基准）

- **目的**: 提供对比基准
- **参数**: 
  - 消费者: 100
  - 邻居数: 4
  - 重连概率: 0.3
  - 初始购买者: 5%
- **预期**: 经典 S 曲线扩散

### 场景2：高密度网络

- **目的**: 研究网络连接度的影响
- **参数**: 
  - 邻居数: 8（更多）
  - 重连概率: 0.8（更高）
- **预期**: 更快的扩散速度，陡峭的 S 曲线

### 场景3：低初始购买者

- **目的**: 测试临界质量的重要性
- **参数**: 
  - 初始购买者: 1%（仅1人）
- **预期**: 扩散可能停滞或缓慢

### 场景对比

- **目的**: 并排对比多个场景
- **功能**: 
  - 两场景对比（左右布局）
  - 三场景对比（网格布局）

## 🎯 路由系统

| 路由 | 页面 | 说明 |
|------|------|------|
| `/` | home.py | 项目首页和介绍 |
| `/scenario1` | scenario1.py | 场景1：默认参数 |
| `/scenario2` | scenario2.py | 场景2：高密度网络 |
| `/scenario3` | scenario3.py | 场景3：低初始购买者 |
| `/comparison` | comparison.py | 场景对比 |

## 🔧 扩展指南

### 添加新场景（3步）

**步骤1**: 创建场景配置

```python
# scenarios/scenario_new.py
from .scenario_base import ScenarioConfig

new_scenario = ScenarioConfig(
    name="新场景",
    description="场景描述",
    params={
        "n_consumers": 150,
        "k_neighbors": 6,
        # ...
    }
)
```

**步骤2**: 创建页面

```python
# pages/scenario_new.py
import solara
from scenarios import new_scenario
from components import ScenarioInfo, ModelVisualization

@solara.component
def Page():
    ScenarioInfo(new_scenario)
    ModelVisualization(scenario_config=new_scenario)
```

**步骤3**: 添加路由

```python
# app.py
from pages import scenario_new

routes = [
    # ... 现有路由
    solara.Route(path="scenario_new", component=scenario_new.Page, label="新场景"),
]
```

## 📚 理论背景

### 创新扩散理论

由 Everett Rogers (1962) 提出，将采纳者分为五类：

| 类型 | 比例 | 特征 |
|------|------|------|
| 🔴 创新者 | 2.5% | 冒险，不需他人影响 |
| 🟠 早期采纳者 | 13.5% | 意见领袖，较早接受 |
| 🟡 早期大众 | 34% | 谨慎但愿意尝试 |
| 🔵 晚期大众 | 34% | 保守，需多数人采纳 |
| ⚪ 落后者 | 16% | 非常保守，最后采纳 |

### 小世界网络

使用 Watts-Strogatz 模型模拟社交网络：

- **聚类性**: 朋友的朋友也是朋友
- **短路径**: 通过少数连接可达任何人
- **重连概率**: 控制网络的随机性

## 🛠️ 技术栈

- **Mesa**: Agent-Based Modeling 框架
- **Solara**: 响应式 Web 应用框架
- **NetworkX**: 网络图分析
- **Matplotlib**: 数据可视化
- **NumPy/Pandas**: 数据处理

## 📖 参考文献

1. Rogers, E. M. (2003). *Diffusion of innovations* (5th ed.). Free Press.
2. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.
3. Mesa Documentation: https://mesa.readthedocs.io/
4. Solara Documentation: https://solara.dev/

## 📝 许可

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**祝实验顺利！** 🎉

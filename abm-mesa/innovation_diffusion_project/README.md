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
│                                    # - ModelVisualization: 完整场景可视化
│                                    # - ScenarioInfo: 场景信息卡片
│
├── utils/                           # 工具函数
│   ├── __init__.py
│   └── agent_portrayal.py          # Agent 绘制函数（颜色、形状等）
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
  - 可切换对比模式

## 🎨 可视化功能

每个场景提供以下可视化内容：

### 网络图
- 实时显示消费者网络结构
- 不同颜色代表不同消费者类型（创新者、早期采纳者等）
- 节点大小或形状区分购买状态

### 数据图表
1. **总体采纳情况**：显示已购买 vs 未购买消费者的绝对数量
2. **各类型采纳率**：5 种消费者类型的采纳率曲线
   - 🔴 创新者采纳率
   - 🟠 早期采纳者采纳率
   - 🟡 早期大众采纳率
   - 🔵 晚期大众采纳率
   - ⚪ 落后者采纳率
3. **总体采纳率**：整体市场的采纳率 S 曲线

### 交互控制
- ▶️ 开始/暂停模拟
- ⏭️ 单步执行
- 🔄 重置模型
- 🎛️ 实时调节参数（消费者数量、邻居数、重连概率、初始购买者比例）

## 🎯 路由系统

| 路由 | 页面 | 说明 |
|------|------|------|
| `/` | home.py | 项目首页和介绍 |
| `/scenario1` | scenario1.py | 场景1：默认参数 |
| `/scenario2` | scenario2.py | 场景2：高密度网络 |
| `/scenario3` | scenario3.py | 场景3：低初始购买者 |
| `/comparison` | comparison.py | 场景对比 |

## 💻 使用示例

### 运行单个场景

1. 启动应用后访问任一场景页面（如场景1）
2. 调节参数滑块（可选）：
   - 消费者数量：50-200
   - 邻居数量：2-10
   - 重连概率：0.0-1.0
   - 初始购买者比例：0.0-0.15
3. 点击 ▶️ 按钮开始模拟
4. 观察网络图和曲线变化
5. 点击 🔄 重置按钮重新开始

### 场景对比

1. 访问 `/comparison` 页面
2. 选择对比模式：
   - **两场景对比**：左右并排显示场景1和场景2
   - **三场景对比**：网格布局显示所有三个场景
3. 同时运行多个场景，观察差异

### 编程方式使用

```python
from models import InnovationDiffusionModel

# 创建模型
model = InnovationDiffusionModel(
    n_consumers=100,
    k_neighbors=4,
    p_rewire=0.3,
    initial_buyers_pct=0.05,
    seed=42
)

# 运行模拟
for i in range(100):
    model.step()
    if not model.running:
        break

# 获取数据
df = model.datacollector.get_model_vars_dataframe()
print(df.head())

# 分析结果
final_adoption_rate = df["Adoption_Rate"].iloc[-1]
print(f"最终采纳率: {final_adoption_rate:.2%}")
```

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
        "p_rewire": 0.5,
        "initial_buyers_pct": 0.08,
    },
    hypothesis="研究假设（可选）",
    expected_outcome="预期结果（可选）"
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
    ModelVisualization(
        scenario_config=new_scenario,
        show_network=True,
        show_type_rates=True,
        show_total_rate=True
    )
```

**步骤3**: 添加路由和导航

```python
# app.py
from pages import scenario_new

routes = [
    # ... 现有路由
    solara.Route(path="scenario_new", component=scenario_new.Page, label="新场景"),
]

# 在 Layout 组件中添加导航按钮
```

### 自定义可视化

可以通过 `ModelVisualization` 组件的参数控制显示内容：

```python
ModelVisualization(
    scenario_config=scenario_config,
    show_network=True,         # 显示网络图
    show_type_rates=True,      # 显示各类型采纳率
    show_total_rate=True       # 显示总体采纳率
)
```

## 📚 理论背景

### 创新扩散理论

由 Everett Rogers (1962) 提出，将采纳者分为五类：

| 类型 | 比例 | 特征 | 从众阈值 |
|------|------|------|----------|
| 🔴 创新者 | 2.5% | 冒险，不需他人影响 | ~0.1 |
| 🟠 早期采纳者 | 13.5% | 意见领袖，较早接受 | ~0.25 |
| 🟡 早期大众 | 34% | 谨慎但愿意尝试 | ~0.45 |
| 🔵 晚期大众 | 34% | 保守，需多数人采纳 | ~0.65 |
| ⚪ 落后者 | 16% | 非常保守，最后采纳 | ~0.85 |

### 小世界网络

使用 Watts-Strogatz 模型模拟社交网络：

- **聚类性**: 朋友的朋友也是朋友（局部连接）
- **短路径**: 通过少数连接可达任何人（全局连通性）
- **重连概率 (p_rewire)**: 控制网络的随机性
  - p = 0：规则环形网络（高聚类，长路径）
  - 0 < p < 1：小世界网络（高聚类，短路径）
  - p = 1：随机网络（低聚类，短路径）

### 从众决策机制

每个消费者在每一步：

1. 观察邻居中已购买的比例
2. 如果该比例 ≥ 自己的从众阈值，则购买
3. 不同类型消费者有不同的阈值分布：
   - 创新者：阈值 ~ N(0.1, 0.05²)，几乎不需要他人影响
   - 早期采纳者：阈值 ~ N(0.25, 0.08²)
   - 早期大众：阈值 ~ N(0.45, 0.1²)
   - 晚期大众：阈值 ~ N(0.65, 0.1²)
   - 落后者：阈值 ~ N(0.85, 0.08²)

## 🛠️ 技术栈

- **Mesa**: Agent-Based Modeling 框架（核心模型）
- **Solara**: 响应式 Web 应用框架（UI 和路由）
- **NetworkX**: 网络图分析（小世界网络生成）
- **Matplotlib**: 数据可视化（图表渲染）
- **NumPy**: 数值计算（阈值分布采样）

## 📈 数据收集

模型自动收集以下数据指标：

### 总体指标
- `Buyers`: 已购买消费者数量
- `Non-Buyers`: 未购买消费者数量
- `Adoption_Rate`: 总体采纳率

### 分类型指标（绝对数量）
- `Innovators_Bought`: 已购买的创新者数量
- `Early_Adopters_Bought`: 已购买的早期采纳者数量
- `Early_Majority_Bought`: 已购买的早期大众数量
- `Late_Majority_Bought`: 已购买的晚期大众数量
- `Laggards_Bought`: 已购买的落后者数量

### 分类型指标（采纳率）
- `Innovators_Rate`: 创新者采纳率
- `Early_Adopters_Rate`: 早期采纳者采纳率
- `Early_Majority_Rate`: 早期大众采纳率
- `Late_Majority_Rate`: 晚期大众采纳率
- `Laggards_Rate`: 落后者采纳率

所有数据可通过 `model.datacollector.get_model_vars_dataframe()` 导出为 Pandas DataFrame

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

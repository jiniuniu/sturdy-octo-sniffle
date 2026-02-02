# 🚀 快速开始指南

## 项目已重构完成！

您的创新扩散模型已经被重构为**多页面路由架构**，现在可以轻松管理多个实验场景。

---

## 📁 新项目结构

```
innovation_diffusion_project/
├── app.py                    # 🌟 主入口（启动这个）
├── models/                   # 模型定义
│   └── innovation_diffusion.py
├── scenarios/                # 场景配置（易于扩展）
│   ├── scenario_default.py
│   ├── scenario_high_density.py
│   └── scenario_low_initial.py
├── pages/                    # 页面组件（每个场景一个）
│   ├── home.py
│   ├── scenario1.py
│   ├── scenario2.py
│   ├── scenario3.py
│   └── comparison.py
├── components/               # 可复用组件
│   └── model_viz.py
└── utils/                    # 工具函数
    └── agent_portrayal.py
```

---

## ▶️ 运行步骤

### 1. 进入项目目录

```bash
cd innovation_diffusion_project
```

### 2. 启动应用

```bash
solara run app.py
```

### 3. 访问浏览器

打开: **http://localhost:8765/**

---

## 🎯 功能特性

### ✅ 多页面路由

每个场景都有独立的 URL 路径：

| URL | 页面 | 说明 |
|-----|------|------|
| `/` | 首页 | 项目介绍 |
| `/scenario1` | 场景1 | 默认参数（基准） |
| `/scenario2` | 场景2 | 高密度网络 |
| `/scenario3` | 场景3 | 低初始购买者 |
| `/comparison` | 对比 | 并排对比场景 |

### ✅ 导航系统

- **顶部导航栏**: 快速切换页面
- **侧边栏**: 详细导航和使用提示
- **按钮高亮**: 当前页面按钮会高亮显示

### ✅ 模块化设计

- **场景配置**: 所有参数集中在 `scenarios/` 目录
- **可复用组件**: `ModelVisualization` 组件可在任何页面使用
- **清晰分离**: 模型、场景、页面各司其职

---

## 🔧 添加新场景（超简单！）

### 步骤 1: 创建场景配置

创建文件 `scenarios/scenario_custom.py`:

```python
from .scenario_base import ScenarioConfig

custom_scenario = ScenarioConfig(
    name="场景4：自定义实验",
    description="您的场景描述",
    params={
        "n_consumers": 150,
        "k_neighbors": 6,
        "p_rewire": 0.5,
        "initial_buyers_pct": 0.08,
    },
    hypothesis="您的研究假设",
    expected_outcome="预期结果"
)
```

### 步骤 2: 创建页面

创建文件 `pages/scenario4.py`:

```python
import solara
from scenarios.scenario_custom import custom_scenario
from components import ScenarioInfo, ModelVisualization

@solara.component
def Page():
    with solara.Column(gap="20px"):
        ScenarioInfo(custom_scenario)
        ModelVisualization(scenario_config=custom_scenario)
```

### 步骤 3: 添加路由

在 `app.py` 中添加：

```python
from pages import home, scenario1, scenario2, scenario3, scenario4, comparison

routes = [
    # ... 现有路由
    solara.Route(path="scenario4", component=scenario4.Page, label="场景4"),
]
```

**完成！** 🎉 新场景已添加！

---

## 📊 场景说明

### 🔵 场景1：默认参数

- **目的**: 基准对比
- **参数**: 标准小世界网络 (n=100, k=4, p=0.3, initial=5%)
- **预期**: 经典 S 曲线扩散

### 🟢 场景2：高密度网络

- **目的**: 研究网络密度影响
- **参数**: 更多邻居 (k=8) + 高重连 (p=0.8)
- **预期**: 扩散更快，曲线更陡

### 🔴 场景3：低初始购买者

- **目的**: 测试临界质量
- **参数**: 仅 1% 初始购买者
- **预期**: 可能无法形成扩散

### 🟡 场景对比

- **功能**: 并排显示 2-3 个场景
- **用途**: 直观对比扩散差异

---

## 🎨 UI 效果

```
┌──────────────────────────────────────────────────┐
│ 🔬 创新扩散 ABM    [首页][场景1][场景2][场景3][对比] │ ← 顶部导航
├──────────┬───────────────────────────────────────┤
│ 📋 导航   │  场景1：默认参数                       │
│          │  ════════════════════════              │
│ 🏠 首页   │  **描述**: 标准小世界网络...            │
│          │                                       │
│ 📊 场景1  │  [网络可视化图]                        │
│ 🌐 场景2  │                                       │
│ 🔻 场景3  │  [采纳率曲线图]                        │
│ ⚖️ 对比   │                                       │
│          │  [控制面板: ▶️ ⏸️ ↻]                  │
│          │                                       │
│ 💡 提示   │                                       │
└──────────┴───────────────────────────────────────┘
```

---

## 💡 使用技巧

### 实验流程

1. **首页**: 了解理论背景
2. **场景1**: 运行基准场景，观察标准扩散
3. **场景2**: 对比高密度网络的影响
4. **场景3**: 观察临界质量的重要性
5. **对比**: 并排对比，找出差异

### 参数调整

每个场景页面都有参数滑块，可以：
- 调整消费者数量
- 改变网络结构
- 修改初始购买者比例
- 实时观察影响

### 数据分析

图表显示：
- **绝对数量**: 已购买 vs 未购买
- **各类型采纳率**: 5 条曲线（比例 0-1）
- **总体采纳率**: S 曲线

---

## 🔍 与旧版本对比

| 特性 | 旧版本 | 新版本（重构后） |
|------|--------|-----------------|
| **结构** | 单文件 | 多模块 |
| **场景管理** | 硬编码 | 配置文件 |
| **页面** | 单页面 | 多页面路由 |
| **导航** | 无 | 顶部栏+侧边栏 |
| **扩展性** | 困难 | 3步添加场景 |
| **复用性** | 低 | 高（组件化） |
| **对比** | 不支持 | 并排对比 |

---

## 📚 代码组织优势

### 1. 清晰的职责分离

```
models/        → 模型逻辑（ABM 核心）
scenarios/     → 场景配置（参数）
pages/         → 页面展示（UI）
components/    → 可复用组件
utils/         → 工具函数
```

### 2. 易于维护

- 修改模型 → 只改 `models/`
- 添加场景 → 只加 `scenarios/` + `pages/`
- 调整UI → 只改 `components/`

### 3. 易于扩展

- 添加新场景：3个文件
- 添加新图表：修改 `components/model_viz.py`
- 添加新分析：创建新组件

---

## 🐛 常见问题

### Q: 如何更改默认端口？

```bash
solara run app.py --port 8080
```

### Q: 如何在 Jupyter 中使用？

```python
# 不推荐在 Jupyter 中使用路由版本
# 如需在 Jupyter 中使用，直接导入组件：
from components import ModelVisualization
from scenarios import default_scenario

ModelVisualization(default_scenario)
```

### Q: 场景运行缓慢怎么办？

- 减少 `n_consumers` 数量
- 在对比页面选择"两场景对比"而非"三场景"

### Q: 如何导出数据？

```python
# 在任何页面运行后，可以访问模型数据
model = InnovationDiffusionModel(**scenario.get_model_params())
for i in range(50):
    model.step()

df = model.datacollector.get_model_vars_dataframe()
df.to_csv('results.csv')
```

---

## 🎯 下一步

### 学习建议

1. ✅ 先运行默认场景，熟悉界面
2. ✅ 对比不同场景，理解参数影响
3. ✅ 尝试添加自定义场景
4. ✅ 修改可视化组件，定制界面

### 扩展方向

1. **新场景**：
   - 场景4：意见领袖影响力
   - 场景5：营销干预实验
   - 场景6：产品质量影响

2. **新功能**：
   - 批量实验运行
   - 数据导出按钮
   - 参数扫描可视化
   - 历史记录对比

3. **新可视化**：
   - 动画演示
   - 3D 网络图
   - 热力图
   - 交互式图表

---

## 🎉 总结

您现在拥有一个：

- ✅ **结构清晰**的多页面 ABM 项目
- ✅ **易于扩展**的场景系统
- ✅ **可复用**的组件架构
- ✅ **专业**的路由和导航

**开始您的创新扩散实验吧！** 🚀

---

需要帮助？查看 `README.md` 或项目文档。

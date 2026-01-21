# Innovation Diffusion Simulation Framework

基于 TPB（Theory of Planned Behavior）与 LLM 的微观—宏观创新扩散仿真框架。

## 项目概述

这是一个用于模拟创新扩散过程的 Agent-Based 仿真框架，核心特点：

- **微观决策模型**：使用 TPB 理论描述个体采纳决策
- **社会网络结构**：支持小世界网络、随机块模型等
- **LLM 增强**（后续阶段）：用 LLM 生成异质性 agent 特征
- **可解释性**：清晰的因果链（画像 → 心理特征 → 行为意向 → 扩散）
- **模块化设计**：所有组件可插拔、可替换

## 当前进度

**✅ 阶段 1：基础框架（已完成）**

- [x] 核心数据结构和协议定义
- [x] 小世界网络生成器（Watts-Strogatz）
- [x] 基础 Agent 状态管理
- [x] 简单扩散仿真引擎（share=1 验证模式）
- [x] 网络可视化和时序图
- [x] 基础实验脚本

**✅ 阶段 2：TPB 决策模型（已完成）**

- [x] TPB 决策组件（社会规范、意向计算、摩擦建模）
- [x] 完整的 TPB 决策模型实现
- [x] TPB Agent 与特征管理
- [x] 人群分层与 traits 采样（Rogers 分类）
- [x] TPB 仿真引擎
- [x] 纯数值 TPB 实验脚本

**⏳ 阶段 3：LLM 集成（待实现）**
**⏳ 阶段 4：增强与优化（待实现）**

## 项目结构

```
diffusion_sim/
├── src/diffusion_sim/          # 核心代码
│   ├── core/                   # 核心抽象（schemas, protocols）
│   ├── agents/                 # Agent 实现（BaseAgent, TPBAgent）
│   ├── networks/               # 网络生成器
│   ├── decisions/              # 决策模型（TPB 等）
│   │   └── components/         # 决策组件（SN, Intention, Friction）
│   ├── population/             # 人群生成与采样
│   ├── simulation/             # 仿真引擎
│   └── visualization/          # 可视化
├── experiments/                # 实验脚本
│   ├── base_diffusion.py       # Step 1: 基础扩散验证
│   └── tpb_numerical.py        # Step 2: TPB 数值实验
├── configs/                    # 配置文件
├── data/                       # 数据目录
│   ├── cache/                  # LLM 缓存
│   └── outputs/                # 仿真输出
└── docs/                       # 设计文档
```

## 快速开始

### 依赖安装

```bash
pip install networkx matplotlib pydantic
```

### 运行实验

**基础扩散实验（Step 1）：**
```bash
cd experiments
python base_diffusion.py
```

**TPB 数值实验（Step 2）：**
```bash
cd experiments
python tpb_numerical.py
```

这将生成可视化结果到 `data/outputs/`

### 示例代码

```python
from diffusion_sim import (
    NetworkConfig,
    SimulationConfig,
    SmallWorldNetworkBuilder,
    SimpleDiffusionEngine,
    TimeSeriesVisualizer,
)

# 配置网络
network_config = NetworkConfig(
    network_type="small_world",
    n=100,  # 100个节点
    k=6,    # 平均度为6
    p=0.1,  # 小世界重连概率
    seed=42
)

# 配置仿真
sim_config = SimulationConfig(
    max_steps=30,
    initial_adopters=1,
    share_probability=1.0,
    adopt_probability=1.0,
    seed=42
)

# 构建网络
builder = SmallWorldNetworkBuilder()
graph = builder.build(network_config)

# 运行仿真
engine = SimpleDiffusionEngine(graph, sim_config, network_config)
result = engine.run()

# 可视化
TimeSeriesVisualizer.create_summary_plot(result)
```

**TPB 实验示例：**
```python
from diffusion_sim import (
    NetworkConfig,
    SimulationConfig,
    SmallWorldNetworkBuilder,
    TPBDiffusionEngine,
    TPBDecisionModel,
    PopulationGenerator,
)

# 构建网络
network_config = NetworkConfig(n=100, k=6, p=0.1, seed=42)
builder = SmallWorldNetworkBuilder()
graph = builder.build(network_config)

# 生成异质性人群（Rogers分类）
pop_gen = PopulationGenerator(seed=42)
pop_gen.generate_population(graph)

# 选择创新者作为种子
initial_ids = pop_gen.select_initial_adopters(graph, 2, strategy="innovators")

# 创建TPB决策模型
decision_model = TPBDecisionModel(
    w_attitude=0.4,
    w_social_norm=0.35,
    w_pbc=0.25
)

# 运行仿真
sim_config = SimulationConfig(max_steps=80, seed=42)
engine = TPBDiffusionEngine(graph, sim_config, network_config, decision_model)
engine.initialize(initial_ids)
result = engine.run()
```

## 核心概念

### Agent 状态

每个 agent 有三个核心状态：
- `aware`: 是否知晓创新
- `adopted`: 是否采纳创新
- `shared_with`: 已分享过的邻居集合

### 扩散流程

**阶段 1（基础模式）：**
1. 初始化：随机选择种子节点，标记为 aware + adopted
2. 每个时间步：
   - 已采纳的 agent 向邻居分享（概率=1）
   - 接收信息的 agent 变为 aware
   - aware 的 agent 立即采纳（概率=1）
3. 重复直到饱和或达到最大步数

**阶段 2（TPB 模式）：**
1. 初始化：根据策略选择种子（如选择 innovators）
2. 每个时间步：
   - **分享决策**：已采纳者基于 intention 和 share_propensity 决定是否分享
   - **采纳决策**：aware 者基于 TPB 计算 intention，考虑摩擦后决定采纳
   - TPB 公式：`I = w_A * Attitude + w_SN * SocialNorm + w_PBC * PBC`
3. 异质性来源：不同 Rogers 分类有不同的 trait 分布

### TPB 决策模型

**TPB（Theory of Planned Behavior）三要素：**
1. **Attitude（态度）**：个体对创新的正面/负面评价
2. **Social Norm（社会规范）**：从邻居采纳率计算的社会压力
3. **Perceived Behavioral Control（感知行为控制）**：感知的采纳难易程度

**Agent Traits（特征）：**
- `attitude`: 基础态度 (0-1)
- `pbc`: 感知控制 (0-1)
- `conformity`: 从众敏感度 (0-1)
- `risk_aversion`: 风险厌恶 (0-1)
- `share_propensity`: 分享倾向 (0-1)
- `innovativeness`: 创新性 (0-1)

**Rogers 人群分类：**
- Innovators (2.5%): 高创新性、低风险厌恶
- Early Adopters (13.5%): 高于平均创新性
- Early Majority (34%): 略高于平均、中等从众性
- Late Majority (34%): 低于平均、高从众性
- Laggards (16%): 低创新性、高风险厌恶

### 网络结构

**Watts-Strogatz 小世界网络**：
- 高聚类系数（朋友圈效应）
- 短平均路径长度（弱连接桥接）
- 参数：
  - `n`: 节点数
  - `k`: 平均度（必须为偶数）
  - `p`: 重连概率（0=规则网络，1=随机网络）

## 输出结果

仿真会生成：
1. **Summary plot**：包含扩散曲线、采纳率、网络统计
2. **Network snapshots**：不同时间点的网络状态快照
3. **SimulationResult** 对象：完整的仿真数据，可导出为 JSON

## 下一步计划

### 阶段 3：LLM 集成（即将开始）
- Persona 原型生成（创业想法 → 目标人群画像）
- Traits 推理（persona → TPB 参数）
- 缓存机制（SQLite + 结构化输出）

### 阶段 4：增强与优化
- 动画生成
- 指标收集与导出
- 参数敏感性分析
- 配置文件系统

## 参考文献

- Rogers, E. M. (2003). *Diffusion of Innovations*
- Ajzen, I. (1991). Theory of Planned Behavior
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks

## 许可证

MIT License

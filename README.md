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

**⏳ 阶段 2：TPB 决策模型（待实现）**
**⏳ 阶段 3：LLM 集成（待实现）**
**⏳ 阶段 4：增强与优化（待实现）**

## 项目结构

```
diffusion_sim/
├── src/diffusion_sim/          # 核心代码
│   ├── core/                   # 核心抽象（schemas, protocols）
│   ├── agents/                 # Agent 实现
│   ├── networks/               # 网络生成器
│   ├── decisions/              # 决策模型（TPB 等）
│   ├── simulation/             # 仿真引擎
│   └── visualization/          # 可视化
├── experiments/                # 实验脚本
│   └── base_diffusion.py       # Step 1: 基础扩散验证
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

### 运行基础实验

```bash
cd experiments
python base_diffusion.py
```

这将：
1. 生成一个 100 节点的小世界网络
2. 运行基础扩散仿真（share=1, adopt=1）
3. 生成可视化结果到 `data/outputs/`

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

## 核心概念

### Agent 状态

每个 agent 有三个核心状态：
- `aware`: 是否知晓创新
- `adopted`: 是否采纳创新
- `shared_with`: 已分享过的邻居集合

### 扩散流程（阶段 1）

1. 初始化：随机选择种子节点，标记为 aware + adopted
2. 每个时间步：
   - 已采纳的 agent 向邻居分享
   - 接收信息的 agent 变为 aware
   - aware 的 agent 立即采纳（阶段 1 简化）
3. 重复直到饱和或达到最大步数

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

### 阶段 2：TPB 决策模型
- 实现完整的 TPB Agent（态度、主观规范、感知行为控制）
- 人群分层（innovators, early majority, late majority, laggards）
- 从分布采样 traits（Beta 分布）

### 阶段 3：LLM 集成
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

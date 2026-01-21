# 创新扩散仿真框架 - 项目结构设计

## 设计原则

1. **关注点分离**：网络、Agent、决策逻辑、LLM 推理各自独立
2. **协议优于实现**：通过 Protocol/ABC 定义接口，支持多种实现
3. **配置驱动**：所有参数通过配置文件管理，支持实验复现
4. **可插拔组件**：网络生成器、决策模型、LLM 后端均可替换

## 目录结构

```
diffusion_sim/
│
├── pyproject.toml                 # 项目配置与依赖
├── README.md
│
├── configs/                       # 配置文件目录
│   ├── base_experiment.yaml       # 基础实验配置
│   ├── networks/                  # 网络配置
│   │   ├── small_world.yaml
│   │   └── stochastic_block.yaml
│   ├── populations/               # 人群配置
│   │   └── consumer_market.yaml
│   └── products/                  # 产品/创新配置
│       └── saas_product.yaml
│
├── src/
│   └── diffusion_sim/
│       │
│       ├── __init__.py
│       │
│       ├── core/                  # 核心抽象与协议
│       │   ├── __init__.py
│       │   ├── protocols.py       # 接口定义 (Protocol classes)
│       │   ├── schemas.py         # Pydantic 数据模型
│       │   └── exceptions.py      # 自定义异常
│       │
│       ├── agents/                # Agent 相关
│       │   ├── __init__.py
│       │   ├── base.py            # BaseAgent 抽象类
│       │   ├── tpb_agent.py       # TPB Agent 实现
│       │   └── state.py           # Agent 状态管理
│       │
│       ├── networks/              # 网络生成与管理
│       │   ├── __init__.py
│       │   ├── base.py            # NetworkBuilder 协议
│       │   ├── small_world.py     # Watts-Strogatz 实现
│       │   ├── sbm.py             # 随机块模型实现
│       │   └── multilayer.py      # 多层网络（预留）
│       │
│       ├── decisions/             # 决策模型
│       │   ├── __init__.py
│       │   ├── base.py            # DecisionModel 协议
│       │   ├── tpb.py             # TPB 决策实现
│       │   ├── threshold.py       # 阈值模型（对照）
│       │   └── components/        # 可组合的决策组件
│       │       ├── __init__.py
│       │       ├── intention.py   # 意向计算
│       │       ├── social_norm.py # SN 聚合策略
│       │       └── friction.py    # 摩擦/障碍建模
│       │
│       ├── llm/                   # LLM 集成
│       │   ├── __init__.py
│       │   ├── base.py            # TraitsInferencer 协议
│       │   ├── openai_backend.py  # OpenAI 实现
│       │   ├── anthropic_backend.py # Claude 实现
│       │   ├── prompts/           # Prompt 模板
│       │   │   ├── __init__.py
│       │   │   ├── persona_gen.py
│       │   │   └── traits_infer.py
│       │   └── cache.py           # 缓存管理
│       │
│       ├── population/            # 人群生成
│       │   ├── __init__.py
│       │   ├── generator.py       # 人群生成器
│       │   ├── personas.py        # Persona 原型管理
│       │   └── sampling.py        # 从原型采样个体
│       │
│       ├── simulation/            # 仿真引擎
│       │   ├── __init__.py
│       │   ├── engine.py          # 主仿真循环
│       │   ├── scheduler.py       # 更新调度（同步/异步）
│       │   ├── events.py          # 事件系统
│       │   └── hooks.py           # 生命周期钩子
│       │
│       ├── metrics/               # 指标与分析
│       │   ├── __init__.py
│       │   ├── collectors.py      # 数据收集器
│       │   ├── diffusion.py       # 扩散曲线指标
│       │   ├── network.py         # 网络指标
│       │   └── export.py          # 数据导出
│       │
│       ├── visualization/         # 可视化
│       │   ├── __init__.py
│       │   ├── network_viz.py     # 网络可视化
│       │   ├── timeseries.py      # 时序图
│       │   ├── animation.py       # 动画生成
│       │   └── dashboard.py       # 交互式仪表板（可选）
│       │
│       └── utils/                 # 工具函数
│           ├── __init__.py
│           ├── config.py          # 配置加载
│           ├── random.py          # 随机数管理
│           └── logging.py         # 日志配置
│
├── experiments/                   # 实验脚本
│   ├── __init__.py
│   ├── base_diffusion.py          # Step 1: 基础扩散验证
│   ├── tpb_numerical.py           # Step 2: 纯数值 TPB
│   ├── tpb_llm.py                 # Step 3: LLM 异质性
│   └── sensitivity_analysis.py    # 参数敏感性分析
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_network_exploration.ipynb
│   ├── 02_tpb_calibration.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                         # 测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_agents.py
│   │   ├── test_decisions.py
│   │   └── test_networks.py
│   ├── integration/
│   │   └── test_simulation.py
│   └── fixtures/
│       └── sample_data.py
│
└── data/                          # 数据目录
    ├── cache/                     # LLM 响应缓存
    ├── outputs/                   # 仿真输出
    └── validation/                # 验证数据（如有）
```

## 核心模块说明

### 1. core/ - 核心抽象层

定义所有组件必须遵循的协议（Protocol），确保可替换性。

```python
# protocols.py 示例
from typing import Protocol, runtime_checkable

@runtime_checkable
class NetworkBuilder(Protocol):
    def build(self, config: NetworkConfig) -> nx.Graph: ...

@runtime_checkable
class DecisionModel(Protocol):
    def compute_intention(self, agent: Agent, context: Context) -> float: ...
    def decide_adopt(self, agent: Agent, intention: float) -> bool: ...
    def decide_share(self, agent: Agent, intention: float, neighbor: Agent) -> bool: ...
```

### 2. decisions/components/ - 可组合决策组件

将 TPB 各部分拆分为独立组件，支持灵活组合：

- `SocialNormAggregator`: 多种 SN 计算策略（均值、加权、阈值）
- `IntentionCalculator`: 意向计算（线性、非线性、神经网络）
- `FrictionModel`: 摩擦建模（固定、动态、情境依赖）

### 3. simulation/hooks.py - 生命周期钩子

支持在仿真各阶段注入自定义逻辑：

```python
class SimulationHooks:
    def on_step_start(self, step: int, state: SimState): ...
    def on_agent_decide(self, agent: Agent, decision: Decision): ...
    def on_step_end(self, step: int, state: SimState): ...
    def on_simulation_end(self, results: SimResults): ...
```

### 4. llm/cache.py - 智能缓存

基于输入 hash 的缓存，支持：

- 本地 SQLite 存储
- 可配置过期策略
- 缓存命中率统计

## 配置文件示例

```yaml
# configs/base_experiment.yaml
experiment:
  name: "saas_diffusion_v1"
  seed: 42
  max_steps: 100

network:
  type: "small_world"
  params:
    n: 1000
    k: 6
    p: 0.1

population:
  layers:
    - name: "innovators"
      ratio: 0.025
      traits_prior:
        innovativeness: [0.7, 0.95]
        risk_aversion: [0.1, 0.3]
    - name: "early_majority"
      ratio: 0.34
      traits_prior:
        conformity: [0.5, 0.8]

decision:
  model: "tpb"
  weights:
    attitude: 0.4
    social_norm: 0.35
    pbc: 0.25

llm:
  enabled: true
  backend: "anthropic"
  cache_dir: "./data/cache"
```

## 扩展点

### 短期扩展

- [ ] 异步更新调度器
- [ ] 更多网络生成器（BA 无标度、真实网络导入）
- [ ] 可视化仪表板（Streamlit/Gradio）

### 中期扩展

- [ ] 多产品竞争扩散
- [ ] 动态 traits（学习/遗忘）
- [ ] 外部事件注入（营销活动、新闻）

### 长期扩展

- [ ] 真实数据校准模块
- [ ] 分布式仿真支持
- [ ] 强化学习策略优化

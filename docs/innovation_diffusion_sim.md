# 基于 TPB 与 LLM 的微观—宏观创新扩散仿真框架设计

## 1. 问题背景与目标

目标是模拟一个**创新扩散（innovation diffusion）**过程，从**微观个体决策**出发，经由**社会网络结构**，涌现出**宏观扩散曲线**。

核心思想：

- 用 **小世界网络（朋友圈）** 描述社会结构；
- 用 **TPB（Theory of Planned Behavior）** 描述单个节点的采纳与分享决策；
- 用 **LLM** 基于用户画像推导个体层面的心理/行为异质性；
- 在 **离散时间仿真**中观察信息与产品的传播。

该框架强调：

- 可解释性（TPB 结构）
- 可复现性（LLM 仅用于初始化参数）
- 可扩展性（网络 / 决策规则 / LLM 可替换）

---

## 2. 总体可行性分析

### 2.1 为什么是可行的

- 创新扩散天然是 **局部交互 + 重复博弈** 的过程，非常适合 agent-based simulation
- TPB 可自然映射为：
  - 心理变量 → 行为意向 → 概率化行为

- LLM 非常适合承担：
  - 「从画像 → 行为倾向参数」的**一次性推理任务**

> 关键原则：
> **LLM 不参与仿真循环，只参与 agent 初始化**。

### 2.2 潜在风险与规避策略

| 风险                     | 解决方案                          |
| ------------------------ | --------------------------------- |
| LLM 输出不稳定           | temperature=0 + 结构化输出 + 缓存 |
| 难以复现实验             | 固定 seed，traits 可序列化        |
| 模型“看起来合理但不可控” | 所有行为最终都走数值规则          |
| SN 过于同质              | 引入个体级「从众敏感度」          |

---

## 3. 微观建模：TPB 在 Agent 中的实现

### 3.1 Agent 的核心状态

每个节点（agent）至少包含以下状态：

- `aware`：是否已获知产品信息
- `adopted`：是否已采纳产品
- `shared_with`：已向哪些邻居分享过（避免无限重复）

### 3.2 TPB 参数拆解（个体异质性来源）

我们不让 LLM 直接输出 SN，而是输出**稳定的个体特质**：

- **Attitude (A)**：对产品的总体评价（0–1）
- **Perceived Behavioral Control (PBC)**：是否“做得到”（0–1）
- **Conformity Sensitivity**：对他人行为的敏感度
- **Share Propensity**：主动分享倾向
- **Risk Aversion / Price Sensitivity**：摩擦项
- **Innovativeness**：早期采纳倾向

这些参数在仿真中保持不变。

### 3.3 Subjective Norm（SN）的网络化定义

SN 来自邻居状态聚合，例如：

- 邻居采纳比例
- 邻居分享行为（加权）
- 关系强度（tie strength）

示例形式：

```
SN_i(t) = sigmoid(
  conformity_i * weighted_adopt_ratio_i(t)
)
```

---

## 4. 网络层建模

### 4.1 网络结构

- 使用 **Watts–Strogatz 小世界网络**
- 参数：
  - `n`：节点数
  - `k`：平均度
  - `p`：重连概率

该结构可同时体现：

- 高聚类（朋友圈）
- 短路径（弱关系桥接）

### 4.2 边属性（可选）

- `tie_strength`：关系强度
- `contact_rate`：互动频率

用于加权 SN 或分享概率。

---

## 5. 行为决策机制（数值化）

### 5.1 意向模型（Intention）

```
I_i(t) = sigmoid(
  wA * A_i
+ wSN * SN_i(t)
+ wPBC * PBC_i
+ bias
)
```

### 5.2 行为层

- **采纳行为**：

```
P(adopt) = I * (1 - friction)
```

- **分享行为**：

```
P(share) = share_propensity_i * I * tie_strength
```

Base case 中可以设：

- `P(share) = 1`（用于验证传播结构）

---

## 6. LLM 在系统中的角色

### 6.1 LLM 的唯一职责

> **从用户画像 → TPB traits（结构化）**

不参与：

- SN 计算
- 行为判定
- 时间推进

### 6.2 技术实现

- LangChain Runnable / Chain
- Pydantic schema 作为输出约束
- temperature = 0
- 基于画像 hash 的本地缓存（sqlite / diskcache）

失败策略：

- 重试 1 次
- 否则回退到默认 traits

---

## 7. Python 模块化架构设计

### 7.1 技术栈

- Python
- networkx
- langchain
- pydantic
- matplotlib (animation)

### 7.2 模块划分

```
diffusion_sim/
│
├── models.py              # Pydantic schemas
├── llm_infer.py           # profile -> traits
├── network_builder.py     # 小世界网络生成
├── decision.py            # TPB 决策逻辑
├── simulator.py           # 时间推进引擎
├── visualize.py           # 动画与统计
└── run_experiment.py      # 实验入口
```

### 7.3 核心接口约定

- `infer_traits(profile) -> TPBTraits`
- `build_graph(config) -> nx.Graph`
- `simulate(G, config) -> SimulationResult`
- `animate(G, pos, snapshots, logs)`

所有模块通过 schema 解耦。

---

## 8. 仿真与可视化

### 8.1 仿真方式

- 离散时间步（synchronous update）
- 每一步：
  1. 计算 SN
  2. 计算 I
  3. 采样 adopt / share
  4. 更新状态

### 8.2 可视化（matplotlib animation）

- networkx 网络图：
  - unaware / aware / adopted 不同颜色

- 右侧时间序列：
  - aware_count
  - adopted_count

优化建议：

- 固定 layout（spring_layout 只算一次）
- 每帧只更新颜色，不重画结构

---

## 9. 初始化设计：目标市场、分层人群与网络生成（从简单到复杂）

本章节聚焦一个关键问题：**在“创业想法”场景下，如何合理初始化人群与网络，使扩散过程既可控又贴近现实**。

核心原则：

- LLM 用于**生成结构化的人群原型与参数先验**，而不是逐个生成个体
- 网络结构从**简单可控**开始，逐步增加现实复杂度

---

### 9.1 目标市场的初始化（TAM / SAM / SOM 的仿真化）

创新扩散通常不是在“全社会”发生，而是首先在一个**目标市场子人群**中展开。

仿真中建议区分：

- **总体人群（optional）**：用于保留少量“破圈传播”可能
- **目标市场人群（primary graph）**：扩散的主要发生区域

实现方式：

- 先生成目标市场规模的 agent（如 N=500–2000）
- 可选地再生成少量“外围节点”，通过弱连接接入

---

### 9.2 创新扩散分层（简化 4 层）

采用 Rogers 创新扩散理论的简化分层：

1. **早期探索者（Innovators + Early Adopters）**
2. **早期大众（Early Majority）**
3. **晚期大众（Late Majority）**
4. **保守者（Laggards）**

> 注意：分层不是标签，而是 **TPB traits 的分布差异**。

#### 分层 → traits 分布映射（示意）

- 早期探索者：
  - innovativeness ↑
  - risk_aversion ↓
  - share_propensity ↑
  - attitude variance ↑

- 早期大众：
  - conformity_sensitivity ↑
  - pbc 中等
  - 风险中等

- 晚期大众：
  - conformity_sensitivity ↑↑
  - pbc 强依赖他人示范
  - price / risk 敏感

- 保守者：
  - attitude 偏低或保守
  - pbc 低 / 摩擦高
  - share_propensity ↓

---

### 9.3 LLM 的使用方式：生成 persona 原型，而非个体

#### 为什么不让 LLM 生成每个 agent？

- 不可复现
- 成本高
- 难以做参数敏感性分析

#### 推荐做法：两层生成

**第一层：LLM 生成 persona 原型（少量）**

输入：

- 创业想法描述
- 目标市场约束（人群、场景、地域、价格带等）
- 分层类型（如“早期探索者”）

输出（结构化）：

- persona 描述（角色 / 使用场景 / 核心动机）
- 主要障碍（price / trust / time / skill）
- TPB traits 的合理先验区间（A / PBC / 风险 / 从众 / 分享倾向）

**第二层：数值引擎采样个体**

- 从 persona 原型给定的区间采样 traits
- 每个原型可生成 10–200 个 agent

---

### 9.4 网络关系生成方案（从简单到复杂）

#### 方案 A（主方案）：WS 小世界网络 + 同质性加权（推荐起步）

**步骤：**

1. 使用 Watts–Strogatz 生成基础小世界网络
2. 固定 layout 与连边结构
3. 根据 persona / traits 相似度：
   - 调整边权（tie_strength）
   - 或对部分边进行重连（增加同层连接概率）

**优点：**

- 实现简单
- 保留朋友圈高聚类 + 短路径
- 易于对照实验（p=0 → 规则网络，p→1 → 随机网络）

适合作为：

- baseline
- 第一版完整系统

---

#### 方案 B（进阶）：随机块模型（SBM / DC-SBM）

将 persona 原型或扩散分层视为 **block / 社群**：

- block 内连边概率高
- block 间连边概率低
- 保留少量跨 block 弱连接

**适用场景：**

- 圈层明显的市场（行业 / 职业 / 兴趣社区）
- 研究“破圈扩散”“关键桥接人群”

**优点：**

- 社区结构清晰
- SN 的来源更符合现实

---

#### 方案 C（高级，可选）：多层网络

- 朋友层 / 同事层 / 线上社群层 各一张图
- 扩散规则在多层并行发生

适合后续扩展，不建议作为起点。

---

### 9.5 初始种子与冷启动策略

常见初始化方式：

1. **创业者节点作为源头**
   - 连接一小撮早期探索者
   - 符合真实冷启动/投放逻辑

2. **多种子初始化**
   - 在早期层随机选取 1–5% aware/adopted
   - 用于模拟媒体曝光或内测用户

3. **对照实验**
   - 相同网络，不同种子位置
   - 观察扩散稳定性

---

### 9.6 接触（Exposure）作为可选增强层

为避免“每步必传播”的不现实假设，可引入轻量接触机制：

- 每条边有 `contact_rate`
- 每步先采样是否发生接触
- 接触后才进入 share / adopt 决策

这会显著提升拟真度，但不会显著增加复杂度。

---

## 10. 推荐的实验路径

### Step 1：Base 扩散验证

- share = 1
- 无 TPB
- 验证小世界扩散速度

### Step 2：纯数值 TPB

- traits 从分布采样（Beta）
- 校准参数区间

### Step 3：引入 LLM 异质性

- 仅替换 traits 来源
- 与 Step 2 做对照实验

---

## 10. 这个框架的研究价值

- 从画像 → 个体决策 → 群体扩散的**完整因果链**
- 可解释、可控、可做反事实分析
- 适合：
  - 创新扩散研究
  - 营销模拟
  - 计算社会科学
  - LLM + ABM 方法论探索

---

> 后续可扩展方向：
>
> - 多产品竞争扩散
> - 动态画像 / 学习机制
> - 社交影响权重学习
> - 校准到真实数据

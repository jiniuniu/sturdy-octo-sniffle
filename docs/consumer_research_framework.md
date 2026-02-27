# 消费者行为研究框架

## 设计文档 v0.3

---

## 1. 背景与动机

### 1.1 模拟目标市场

产品上线之前，创业者最需要回答的问题往往是这样的：

> "我的目标用户真的有这个需求吗？"
> "哪类用户最可能买单，哪类用户阻力最大？"
> "如果我改变产品的某个特性，对哪类人影响最大？"

这些问题的本质是**对目标市场的模拟**——在真实用户数据缺席的情况下，构造一个足够多样的虚拟市场，在其中观察不同类型消费者对产品的反应，从反应的差异中提取洞察。

本框架的出发点是：**用 LLM 合成的 persona 群体模拟目标市场**，用 TPB（计划行为理论）作为测量每个 persona 反应的理论框架，用进化算法保证覆盖到尽可能多样的消费者类型，最终通过归因分析回答"是什么驱动了不同消费者的不同反应"。

### 1.2 从自然语言到研究设计

研究者不需要预先知道应该研究哪些变量。系统的入口是一段自然语言描述：

> "我们做了一款面向慢性病患者的用药管理 App，想在二三线城市通过社区医院推广，
>  主要想搞清楚哪类患者最可能坚持用这个 App 来辅助服药"

LLM 从这段描述中推导出结构化的研究规范：

```
scenario:    "用药管理 App，功能含用药提醒和复诊预约，通过社区医院渠道推广"
outcome_y:   "患者用药依从意愿（坚持使用 App 辅助按时按量服药的意愿）"
population:  "中国二三线城市慢性病患者，40~65岁，以高血压/糖尿病为主"
domain:      "healthcare"
purpose:     "barrier_analysis"
```

这个推导过程本身是系统的第一步，也是让普通产品经理（而非研究方法论专家）能直接使用系统的关键。

### 1.3 核心设计原则

**原则一：描述"是谁"和测量"怎么反应"必须解耦**

用"对健康 App 的接受度"描述用户，再预测其"采纳 App 的意愿"——这是循环自证。
正确做法：用**不含 Y 内容的维度**描述用户（医患信任度、家庭督促强度、疾病感知严重性……），再独立评估 Y。

**原则二：Y 的分布比 Y 的均值更有价值**

"平均依从意愿 0.6"没有决策价值。"什么特征的用户依从意愿低、低在哪个 TPB 分项上、改变哪个维度可以翻转"才是产品和营销的输入。这要求样本在 Y 上**均匀分布**，覆盖完整区间——而非反映真实人口的自然分布。

**原则三：覆盖极端情形需要主动搜索**

真实人群的极端情形（高阻力、高接受）天然稀疏，被动采样永远欠采。系统需要**主动生成**这些极端 persona，而进化算法是解决这个问题的自然选择。

**原则四：领域先验知识压缩搜索空间**

医疗领域的依从行为有稳定的影响因素结构（医患关系、家庭压力、副作用感知……）。将这些结构编码为**领域模板**，比每次让 LLM 从零生成维度更可靠、更高效。

---

## 2. 问题 Formulation

### 2.1 研究规范三元组

任何消费者行为研究问题被规范化为：

```
ResearchSpec = (scenario, outcome_y, population)
```

| 字段 | 含义 | 示例 |
|------|------|------|
| `scenario` | 产品/干预的上下文描述 | "用药管理 App，通过社区医院渠道推广" |
| `outcome_y` | 目标变量，必须可操作、有方向、不循环 | "患者用药依从意愿" |
| `population` | 目标人群的范围约束 | "二三线城市慢病患者，40~65岁" |

三元组由 LLM 从自然语言输入中自动推导，推导过程同时确定 `domain`（领域标签）和 `purpose`（研究目的）。

**outcome_y 的质量标准：**

| 标准 | 说明 | 反例 → 修正 |
|------|------|------------|
| 可操作 | 能对应具体行为或可测量态度 | "用户满意度" → "30天内主动打开超过15次的意愿" |
| 有方向 | Y 高/低分别意味着什么清晰 | "接受度" → "相比现有用药方式，增加依从的意愿" |
| 不循环 | 不能用 Y 的内容描述用户维度 | "对 App 的开放程度" 不能作为描述维度 |
| 因果相关 | 与 scenario 有合理的因果联系 | - |

### 2.2 形式化定义

给定 ResearchSpec，研究被形式化为一个**全局敏感性分析**问题：

```
Y_i = f(X_i) + ε_i

X_i = (x_1, x_2, ..., x_D)  ← persona i 在 D 个消费者维度上的取值，x_d ∈ [0,1]
Y_i ∈ [0, 1]                ← persona i 的 outcome 评分
ε_i                         ← LLM 评估的随机误差

研究目标：估计 f 的结构——哪些 x_d 对 Y 的方差贡献最大，
          以及 x_d 之间的交互效应如何影响 Y
```

这不是预测问题，而是**敏感性分析**：在合理假设下，Y 对哪些消费者特征最敏感？

---

## 3. 项目设计框架

### 3.1 全景架构

```
 自然语言输入
 "我们的用药 App 想搞清楚哪类慢病患者会坚持用..."
         │
         ▼
┌────────────────────────────────────────────────────┐
│  Step 0  研究规范化                                 │
│  LLM 推导 ResearchSpec + 匹配 Domain Template      │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  Step 1  维度生成                                   │
│  领域模板骨架 + LLM 场景特化 → D 个 DimensionAxis  │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  Step 1.5  问卷生成（整个研究只做1次）              │
│  LLM 生成 12道 TPB 问卷（4题×3维度）               │
│  问卷与所有 persona 共用，固定不变                  │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  Step 2  初始种群生成                               │
│  Sobol 采样 → N₀ 个 Persona（narrative）           │
│  每个 persona 作答问卷 → 纯计算得 Y                │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  Step 3  MAP-Elites 进化循环          ◄──────┐     │
│  ┌─────────────────────────────────┐         │     │
│  │ 检查 Archive：哪些 Y 区间覆盖不足 │         │     │
│  └────────────┬────────────────────┘         │     │
│               │                              │     │
│               ▼                              │     │
│  ┌─────────────────────────────────┐         │     │
│  │ 从 Archive 选亲本 persona        │         │     │
│  │ 施加变异（维度数值 / 叙事文本）   │         │     │
│  └────────────┬────────────────────┘         │     │
│               │                              │     │
│               ▼                              │     │
│  ┌─────────────────────────────────┐         │     │
│  │ 新 persona 作答同一份问卷 → Y    │         │     │
│  │ 尝试放入 Archive 对应 cell       │         │     │
│  └────────────┬────────────────────┘         │     │
│               │                              │     │
│               ▼                              │     │
│  ┌─────────────────────────────────┐         │     │
│  │ 终止条件检查                     │─ 未满足 ─┘     │
│  │ · Y 各区间覆盖 ≥ 阈值            │                │
│  │ · 达到最大代数                   │                │
│  └────────────┬────────────────────┘                │
│               │ 满足                                 │
└───────────────┼─────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────────────────────┐
│  Step 4  归因分析                                   │
│  SHAP + 聚类 + 交互效应 → AttributionResult        │
└────────────────────────────────────────────────────┘
```

### 3.2 领域模板系统

领域模板将领域专家知识结构化，为维度生成提供可靠骨架。

**支持的领域：**

| 领域标签 | 典型 scenario | 典型 outcome_y |
|----------|--------------|----------------|
| `healthcare` | 慢病管理 App、用药提醒、复诊管理 | 用药依从性、复诊率、健康行为改变 |
| `consumer_tech` | 新 App、硬件、订阅服务 | 7日留存、首次付费、口碑传播 |
| `fintech` | 理财 App、信贷、保险 | 首次入金、续费意愿、风险决策行为 |
| `education` | 在线课程、职业培训 | 完课率、续费、自发探索频率 |
| `retail` | 新品上市、渠道下沉 | 复购意愿、价格临界、推荐意愿 |
| `policy` | 公共卫生干预、政策推广 | 自愿参与率、行为改变持续性 |

**`healthcare` 模板示例：**

```yaml
domain: healthcare

dimension_skeleton:
  - 疾病感知严重性       # 对自身病情严重性的主观感知
  - 医患信任度           # 对医生建议的接受和信任程度
  - 现有治疗方案依赖深度  # 对既有用药习惯的依赖程度
  - 家庭支持环境         # 家人督促、提醒、协助的强度
  - 数字工具接受度       # 使用手机 App 的基础意愿和能力
  - 副作用感知强度       # 对用药副作用的担忧程度
  - 经济约束强度         # 医疗费用对用药行为的限制

tpb_weights:
  attitude:    0.30
  social_norm: 0.35   # 家庭/医生规范影响大
  pbc:         0.35   # 坚持能力感知影响大

known_interactions:
  - [疾病感知严重性, 副作用感知强度]  # 交叉决定用药态度
  - [家庭支持环境, 数字工具接受度]   # 家人可弥补数字能力不足

y_templates:
  用药依从意愿: "按时按量服药的主观意愿，含坚持意愿和遗忘后的主动补救"
  复诊意愿:    "主动预约下次复诊，而非被动等待"
  App持续使用: "下载后30天内仍主动打开的意愿"
```

---

## 4. MAP-Elites 进化算法

### 4.1 为什么需要进化算法

Sobol 序列在维度空间（X 空间）均匀，但无法保证 Y 空间均匀——因为 X → Y 的映射是未知且非线性的。初始种群生成后，Y 的分布往往集中在某个区间，低 Y 段（高阻力人群）或高 Y 段（强接受人群）严重稀疏。

进化算法解决的问题：**主动搜索 Y 空间的稀疏区域**，而不是被动等待随机采样"碰巧"命中。

### 4.2 MAP-Elites 在本框架中的映射

标准 MAP-Elites 维护一个行为空间（Behavior Space）的网格，每个格子（cell）存放当前最优个体。本框架的映射如下：

| MAP-Elites 概念 | 本框架对应 | 说明 |
|----------------|-----------|------|
| **个体（Solution）** | Persona | 含 dimension_values + narrative + Y |
| **行为描述子（BD）** | Y 值（+ 可选的关键维度） | 决定 persona 落入哪个 cell |
| **适应度（Fitness）** | narrative 质量分 | 同一 cell 内保留叙事最清晰的 persona |
| **变异（Mutation）** | 维度数值扰动 / LLM 叙事改写 | 见 4.4 节 |
| **Archive** | 按 Y 区间组织的 persona 库 | 目标：每个 cell ≥ K 个 persona |

**Archive 结构（1D，按 Y 分格）：**

```
Archive cells（Y 轴，K=5格）：

 cell_0: Y ∈ [0.0, 0.2)  │ target: ≥30 │ current: 8   ← 严重稀疏
 cell_1: Y ∈ [0.2, 0.4)  │ target: ≥30 │ current: 24  ← 稀疏
 cell_2: Y ∈ [0.4, 0.6)  │ target: ≥30 │ current: 71  ← 过密
 cell_3: Y ∈ [0.6, 0.8)  │ target: ≥30 │ current: 58  ✓
 cell_4: Y ∈ [0.8, 1.0]  │ target: ≥30 │ current: 39  ✓
```

**可选的 2D Archive（Y × 关键维度，更精细）：**

```
                 数字工具接受度
              低(0-0.33) 中(0.33-0.67) 高(0.67-1.0)
          ┌──────────┬──────────────┬──────────┐
 Y: 高    │  ★ 稀疏  │     ✓        │    ✓     │
(0.6-1.0) │  [特殊群]│              │          │
          ├──────────┼──────────────┼──────────┤
 Y: 中    │    ✓     │     ✓        │    ✓     │
(0.3-0.6) │          │              │          │
          ├──────────┼──────────────┼──────────┤
 Y: 低    │    ✓     │   ★ 稀疏    │  ★ 稀疏  │
(0.0-0.3) │          │   [高数字能力 │          │
          │          │    但低依从] │          │
          └──────────┴──────────────┴──────────┘
```

2D Archive 能捕捉到一维 Archive 遗漏的**交互边界**（如"数字能力高但依从意愿低"的反直觉群体）。

### 4.3 亲本选择策略

每次变异前，从 Archive 中选择亲本 persona，选择策略决定进化方向：

| 策略 | 选择方式 | 适用时机 |
|------|----------|----------|
| **稀疏优先** | 从最欠采样的 cell 中随机选 | 默认策略，快速填充空白 |
| **边界探索** | 从相邻 cell 的边界附近选（Y 接近格子边缘的 persona） | Y 分布基本覆盖后，精细化边界 |
| **随机** | 从整个 Archive 随机选 | 防止局部收敛，定期引入 |

### 4.4 变异算子

变异是本框架的核心创新——**persona 的叙事文本可以被 LLM 改写**，这使得"变异"不只是数值扰动，而是语义层面的人物特征改变。

#### 变异类型 A：维度数值扰动（Numeric Mutation）

最轻量的变异，直接修改 dimension_values，然后基于新数值重新生成 narrative：

```
输入：亲本 persona P，目标 cell（对应 Y 区间 [y_low, y_high]）

操作：
  1. 识别与 Y 相关性最高的维度 d*
  2. 如果目标 Y 高于亲本 Y：将 x_{d*} 向高端扰动（+δ，δ~Uniform(0.1, 0.3)）
     如果目标 Y 低于亲本 Y：将 x_{d*} 向低端扰动（-δ）
  3. 其余维度加小噪声（σ=0.05）保持多样性
  4. LLM 基于新的 dimension_values 重新生成 narrative
  5. 评估新 persona 的 Y

成本：1次 LLM 调用（narrative 生成）+ 1次 LLM 调用（Y 评估）
```

#### 变异类型 B：叙事改写（Narrative Mutation）

更语义化的变异，LLM 直接改写 persona 的叙事文本，而非先改数值再重写：

```
Prompt 模板：
─────────────────────────────────────────────────
你是一位消费者研究专家。

当前 persona：
{persona.narrative}

当前 Y 值（{outcome_y}）：{persona.y_value:.2f}

目标：生成一个变体 persona，使其 Y 值落在 [{y_low:.1f}, {y_high:.1f}] 区间。

要求：
- 保留原 persona 的基本人口学特征（年龄、地域、职业方向）
- 修改以下方面来影响 Y：[根据目标方向，列出 1~2 个应该改变的特征]
- 修改幅度合理，不要制造不真实的极端人物
- 输出格式：新的 persona 描述（150~300字）+ 各维度新值的 JSON

新 persona：
─────────────────────────────────────────────────

成本：1次 LLM 调用（含 narrative + 维度值）+ 1次 LLM 调用（Y 评估）
```

#### 变异类型 C：维度交叉（Crossover）

将两个亲本的维度值组合，生成兼具双方特征的新 persona：

```
输入：亲本 P1（来自高 Y cell），亲本 P2（来自低 Y cell）

目标：探索中间区间，或探索"P1 的某些高维度 + P2 的某些低维度"的交叉情形

操作：
  1. 随机掩码 mask ∈ {0,1}^D
  2. x_child = mask * x_P1 + (1-mask) * x_P2
  3. LLM 基于 x_child 生成 narrative（需要保证内部一致性）
  4. 评估 Y

成本：1次 LLM 调用 + 1次 LLM 调用
注意：交叉生成的 persona 需要额外的叙事一致性检查，
      防止维度值组合产生"不真实"的人物
```

### 4.5 进化循环完整逻辑

```python
def run_map_elites(
    research_spec: ResearchSpec,
    dimensions: list[DimensionAxis],
    archive_config: ArchiveConfig,
    max_generations: int = 50,
    batch_size: int = 10,
) -> Archive:

    # 初始化：Sobol 采样 N0 个 persona 建立初始 Archive
    initial_personas = sobol_sample(dimensions, n=archive_config.n_initial)
    for p in initial_personas:
        p.y_value = evaluate_y(p, research_spec)
        archive.try_insert(p)

    # 进化主循环
    for gen in range(max_generations):

        # 终止检查
        coverage = archive.coverage_stats()
        if coverage.all_cells_satisfied(min_count=archive_config.target_per_cell):
            break

        # 选出本代需要填充的 cells（稀疏优先）
        target_cells = archive.select_sparse_cells(batch_size)

        # 对每个目标 cell 生成新 persona
        new_personas = []
        for cell in target_cells:

            # 选择变异类型（按比例）
            mutation_type = sample_mutation_type(
                weights={"numeric": 0.5, "narrative": 0.4, "crossover": 0.1}
            )

            # 选择亲本
            parent = archive.select_parent(
                target_cell=cell,
                strategy="boundary"   # 优先选边界附近的 persona
            )

            # 施加变异
            child = mutate(parent, cell.y_range, mutation_type, dimensions)

            # 评估 Y
            child.y_value = evaluate_y(child, research_spec)
            child.generation = gen

            new_personas.append(child)

        # 尝试将新 persona 插入 Archive
        for p in new_personas:
            archive.try_insert(p)   # 若 cell 内已有更优个体则丢弃

        log_generation(gen, archive.coverage_stats())

    return archive
```

### 4.6 Archive 插入规则

同一 cell 内可存放多个 persona（不止一个 elite），但有竞争：

```
try_insert(persona, cell):
    existing = archive.get(cell)

    if len(existing) < target_per_cell:
        # cell 未满，直接插入
        archive.add(cell, persona)

    else:
        # cell 已满，与质量最低的竞争
        worst = min(existing, key=lambda p: p.narrative_quality_score)
        if persona.narrative_quality_score > worst.narrative_quality_score:
            archive.replace(cell, worst, persona)
```

`narrative_quality_score` 由 LLM 评估叙事的内部一致性和真实感（0-1），防止维度数值和叙事文本脱节的低质量 persona 占用格子。

### 4.7 进化过程的诊断指标

| 指标 | 计算方式 | 用途 |
|------|----------|------|
| **Coverage** | 已满足最低数量的 cell 比例 | 进化进度的核心指标 |
| **Y Entropy** | Archive 内 Y 分布的信息熵 | 越高越均匀 |
| **Mutation Success Rate** | 成功插入 Archive / 总变异次数 | 过低说明变异方向需调整 |
| **Narrative Consistency** | LLM 评估叙事与维度值的一致性均值 | 质量控制 |
| **Generation Yield** | 每代新插入的 persona 数 | 收敛速度 |

---

## 5. Y 评估机制：问卷方式

### 5.1 为什么用问卷而不是直接打分

最直觉的 Y 评估方式是：把 persona 描述给 LLM，让它直接输出一个 0~1 的 TPB 分数。这样做有两个根本缺陷：

**缺陷一：抽象数值缺乏锚定**
"请给这个人的用药依从态度打 0~1 分"——LLM 没有稳定的参照系，同一个 persona 在不同调用中可能得到差异较大的分数。

**缺陷二：推理过程不可检查**
直接输出的数值无法追溯"为什么是这个分数"，无法区分是 Attitude 不足还是 PBC 不足，失去了 TPB 的诊断价值。

**问卷方式的优势**：
- 每道题是具体的行为陈述（"我觉得按时服药对控制病情有明显帮助"），语义锚定更强，LLM 回答更稳定
- persona 以角色扮演方式作答，而非元层面地"评估自己"，更符合 LLM 的能力边界
- 每题有推理文本记录，可以事后检查哪道题驱动了哪个分项
- 问卷固定，所有 persona 用同一把尺子量，跨 persona 比较有效

### 5.2 问卷生成（Step 1.5，整个研究只做1次）

**输入：** `ResearchSpec`（scenario + outcome_y + population）

**输出：** 一份包含 12 道题的 TPB 问卷，每个维度 4 题，其中至少 1 题为反向计分题

```
问卷结构：

Attitude（q01~q04）
  · 考察：persona 对"执行目标行为"的评价性态度
  · 不是对产品的喜好，而是对行为本身的价值判断
  · 反向题示例："我觉得每天按时吃药太麻烦了，不值得这么做"

Subjective Norm（q05~q08）
  · 考察：persona 感知到的重要他人的期待和社会压力
  · 示例："我的家人很希望我能按时按量服药"
  · 示例："我的主治医生非常强调坚持用药的重要性"

Perceived Behavioral Control（q09~q12）
  · 考察：persona 对自己能否完成该行为的信心和控制感
  · 示例："我觉得自己能养成每天按时服药的习惯"
  · 反向题示例："我经常因为忘记而漏服药，觉得自己很难坚持"
```

**题目生成的约束：**

| 约束 | 说明 |
|------|------|
| 第一人称 | 所有题目用"我"开头，符合 persona 角色扮演的语境 |
| 行为导向 | 围绕 outcome_y 对应的具体行为，而非对产品的抽象评价 |
| 符合人群 | 语言风格匹配 population（40~65岁慢病患者 → 口语化，不用技术词汇） |
| 反向题分布 | 每个维度至少1道，防止 yes-saying 偏差 |
| 独立性 | 各题测量不同侧面，不重复 |

### 5.3 Persona 作答（每个 persona 独立，LLM 角色扮演）

每个 persona 收到同一份问卷，LLM 扮演该 persona 逐题作答：

```
Prompt 结构：
─────────────────────────────────────────────────────────
你是以下这个人：

{persona.narrative}

请以这个人的身份，对下面每道题用 1~5 分作答：
  1 = 完全不同意
  2 = 比较不同意
  3 = 一般
  4 = 比较同意
  5 = 完全同意

要求：
- 完全保持角色，不要跳出角色说"作为 AI"
- 回答要反映这个人的真实处境和心理，低分就给低分
- 每题附一句话说明你为什么这样评分

题目：
q01 [态度] 我觉得按时按量服药对控制病情有明显帮助
q02 [态度] 坚持用药是我愿意为健康付出的事情
q03 [态度-反向] 我觉得每天记着吃药太麻烦了，不值得
q04 [态度] 用 App 提醒我吃药，能让我觉得更安心
q05 [社会规范] 我的家人很希望我能坚持按时服药
...（共12题）
─────────────────────────────────────────────────────────
```

**输出示例（节选）：**

```json
{
  "q01": {"score": 4, "reasoning": "知道血压不控制会有风险，但有时候觉得症状不明显就忘了"},
  "q03": {"score": 4, "reasoning": "确实嫌麻烦，有时候出门忘带药就算了"},
  "q05": {"score": 2, "reasoning": "老伴不太管我这些，孩子在外地，没人特别督促"}
}
```

### 5.4 Y 值计算（纯数学，不再调用 LLM）

```python
def compute_y(response: QuestionnaireResponse,
              questionnaire: Questionnaire,
              tpb_weights: dict[str, float]) -> TPBScores:

    scores_by_dim = {"attitude": [], "social_norm": [], "pbc": []}

    for item in questionnaire.items:
        raw = response.responses[item.id].score          # 1~5
        # 反向题处理
        adjusted = (6 - raw) if item.reverse_scored else raw
        # 归一化到 [0, 1]
        normalized = (adjusted - 1) / 4.0
        scores_by_dim[item.tpb_dimension].append(normalized)

    attitude    = mean(scores_by_dim["attitude"])        # [0, 1]
    social_norm = mean(scores_by_dim["social_norm"])     # [0, 1]
    pbc         = mean(scores_by_dim["pbc"])             # [0, 1]

    intention = (
        tpb_weights["attitude"]    * attitude +
        tpb_weights["social_norm"] * social_norm +
        tpb_weights["pbc"]         * pbc
    )

    return TPBScores(
        attitude=attitude,
        social_norm=social_norm,
        pbc=pbc,
        intention=intention,          # 即最终 Y 值
    )
```

**默认权重**（来自领域模板，可被场景覆盖）：

| 领域 | Attitude | Social Norm | PBC |
|------|----------|-------------|-----|
| `healthcare` | 0.30 | 0.35 | 0.35 |
| `consumer_tech` | 0.40 | 0.30 | 0.30 |
| `general` | 0.40 | 0.30 | 0.30 |

### 5.5 每次变异的评估成本

MAP-Elites 进化循环中，每产生一个新 persona 需要：

```
1次 LLM 调用：narrative 生成（或叙事改写）
1次 LLM 调用：persona 作答 12道问卷（12题打包为单次调用）
0次 LLM 调用：Y 值计算（纯数学）

合计：2次 LLM 调用 / 新 persona
```

问卷生成只在研究开始时做 1次，之后固定复用，不计入变异成本。

### 5.6 问卷质量的影响

问卷本身由 LLM 生成，存在质量风险，需要在 Step 1.5 之后做验证：

| 验证项 | 检查方法 |
|--------|----------|
| 维度纯净性 | 每题只属于一个 TPB 维度，LLM 判断是否混入了其他维度的内容 |
| 行为聚焦 | 题目围绕 outcome_y 的具体行为，而非对产品的整体印象 |
| 反向题自然性 | 反向题读起来是真实的负面陈述，而非机械地加"不" |
| 人群匹配 | 语言风格符合 population 描述（年龄、教育程度、文化背景） |

若验证不通过，重新生成问卷（最多3次），之后锁定不再更改。

---

## 6. 核心数据结构

### 5.1 输入层

```python
class ResearchSpec(BaseModel):
    # ── 由 LLM 从自然语言输入中推导 ─────────────────────────────
    raw_input: str                    # 用户的原始自然语言描述
    scenario: str                     # 推导出的场景描述
    outcome_y: str                    # 推导出的目标变量
    population: str                   # 推导出的目标人群
    domain: str                       # 推导出的领域标签

    # ── 研究配置 ─────────────────────────────────────────────────
    purpose: Literal[
        "target_identification",      # 找目标人群
        "barrier_analysis",           # 找阻力来源
        "threshold_detection",        # 找临界条件
        "intervention_priority",      # 确定干预优先级
        "segment_comparison",         # 跨分群比较
    ] = "barrier_analysis"

    n_initial: int = 100              # 初始 Sobol 种群大小
    target_per_cell: int = 30         # 每个 Archive cell 的目标 persona 数
    max_generations: int = 50         # MAP-Elites 最大迭代代数
    archive_dims: int = 1             # Archive 的维度数（1=只按 Y；2=Y×关键维度）
```

### 5.2 维度轴

```python
class DimensionAxis(BaseModel):
    name: str                         # 例："疾病感知严重性"
    description: str                  # 完整含义说明
    low_anchor: str                   # 取值≈0 时的典型人物场景（1~2句）
    high_anchor: str                  # 取值≈1 时的典型人物场景（1~2句）
    causal_path_to_y: str             # 该维度影响 Y 的因果路径
    domain_source: Literal["template", "scenario_specific"]
    y_correlation_prior: float        # 理论预期与 Y 的相关方向，-1 或 +1
```

### 6.3 问卷结构

```python
class QuestionnaireItem(BaseModel):
    id: str                           # "q01" ~ "q12"
    tpb_dimension: Literal["attitude", "social_norm", "pbc"]
    question_text: str                # 第一人称陈述句，符合目标人群语言风格
    reverse_scored: bool              # 是否反向计分


class Questionnaire(BaseModel):
    research_spec_id: str             # 关联的 ResearchSpec
    behavior_goal: str                # 问卷聚焦的具体行为目标
    items: list[QuestionnaireItem]    # 固定12题：attitude×4, social_norm×4, pbc×4
    # 问卷在 Step 1.5 生成后锁定，整个研究复用


class SingleResponse(BaseModel):
    score: int                        # 1~5（Likert 量表）
    reasoning: str                    # persona 的作答理由（1~2句）


class QuestionnaireResponse(BaseModel):
    persona_id: str
    responses: dict[str, SingleResponse]  # {"q01": SingleResponse, ...}
```

### 6.4 合成 Persona

```python
class Persona(BaseModel):
    persona_id: str                   # 格式："P_{代数}_{序号}"，例 "P_03_042"
    generation: int                   # 产生于第几代（0 = 初始 Sobol 种群）
    origin: Literal["sobol", "numeric_mutation", "narrative_mutation", "crossover"]
    parent_ids: list[str]             # 亲本 ID（初始种群为空）

    dimension_values: dict[str, float]  # 各维度取值，均 ∈ [0, 1]
    narrative: str                    # LLM 生成的人物叙事文本（150~300字）
    narrative_quality_score: float    # 叙事与维度值的一致性评分，[0, 1]

    questionnaire_response: QuestionnaireResponse  # 问卷作答记录
    tpb_scores: TPBScores             # 由作答结果计算得出，无需额外 LLM 调用
    y_value: float                    # = tpb_scores.intention，最终 Y 值

    cell_id: str                      # 在 Archive 中所属 cell


class TPBScores(BaseModel):
    attitude: float                   # Attitude 维度均值，[0, 1]
    social_norm: float                # Subjective Norm 维度均值，[0, 1]
    pbc: float                        # Perceived Behavioral Control 均值，[0, 1]
    intention: float                  # 加权汇总 = Y 值，[0, 1]
    # 注：无 friction 字段，摩擦效应通过 pbc 低分和反向题自然体现
```

### 6.5 Archive

```python
class ArchiveCell(BaseModel):
    cell_id: str                      # 例："Y[0.0,0.2)" 或 "Y[0.0,0.2)_D1[0.0,0.33)"
    y_range: tuple[float, float]
    aux_dim_range: Optional[tuple[float, float]]  # 2D Archive 时的辅助维度范围
    personas: list[Persona]           # 该 cell 内的所有 persona
    is_satisfied: bool                # 是否达到 target_per_cell


class Archive(BaseModel):
    research_spec: ResearchSpec
    dimensions: list[DimensionAxis]
    cells: dict[str, ArchiveCell]
    total_generations: int
    total_evaluations: int            # LLM 调用总次数（Y 评估）

    def coverage_stats(self) -> CoverageStats: ...
    def select_sparse_cells(self, n: int) -> list[ArchiveCell]: ...
    def select_parent(self, target_cell, strategy) -> Persona: ...
    def try_insert(self, persona: Persona) -> bool: ...


class CoverageStats(BaseModel):
    total_cells: int
    satisfied_cells: int
    coverage_ratio: float             # satisfied / total
    y_entropy: float                  # Y 分布的信息熵
    min_cell_count: int
    max_cell_count: int
```

### 6.6 归因结果

```python
class AttributionResult(BaseModel):
    research_spec: ResearchSpec
    archive_summary: CoverageStats

    feature_importance: list[FeatureImportance]   # 按重要性降序
    segments: list[PersonaSegment]                 # 按 Y 区间分群
    interaction_effects: list[InteractionEffect]   # 关键交互效应
    threshold_conditions: list[ThresholdCondition] # Y 翻转条件
    recommendations: list[Recommendation]          # 可操作建议


class FeatureImportance(BaseModel):
    dimension_name: str
    importance_score: float           # 归一化重要性 [0, 1]
    shap_mean_abs: float
    correlation_with_y: float         # Spearman 相关系数
    direction: Literal["positive", "negative", "nonlinear"]
    interpretation: str               # 一句话解释


class PersonaSegment(BaseModel):
    segment_id: str                   # 例："S_low_barrier"
    y_range: tuple[float, float]
    size: int
    label: str                        # 例："高阻力群"
    centroid: dict[str, float]        # 各维度中位数
    key_characteristics: list[str]    # 2~4条文字描述
    tpb_bottleneck: str               # 该群体 TPB 的主要限制分项
    representative_persona_id: str    # Archive 中最典型的 persona


class InteractionEffect(BaseModel):
    dim_a: str
    dim_b: str
    effect_description: str
    effect_strength: float
    y_direction: str                  # "高×高→Y显著升" 等


class ThresholdCondition(BaseModel):
    trigger_dimension: str
    threshold_value: float
    y_change: float
    description: str


class Recommendation(BaseModel):
    priority: int
    target_segment: str
    action: str
    rationale: str
    expected_y_lift: Optional[float]
```

---

## 7. 流转逻辑

### 7.1 Step 0：自然语言 → 研究规范

```
输入：自然语言描述

LLM 任务：
  1. 提取 scenario（产品/干预的核心描述，去除营销语言）
  2. 识别 outcome_y（最接近用户意图的可操作变量）
     · 如果用户描述模糊，主动澄清并给出默认值
     · 如果 outcome_y 存在循环风险，自动修正并说明
  3. 提取 population（人群范围约束）
  4. 匹配 domain（关键词匹配优先，LLM 分类兜底）
  5. 推断 purpose（从"哪类用户"→target_identification；
                    从"为什么不用"→barrier_analysis；等）

输出：ResearchSpec（含推导过程的透明说明）
```

**推导示例：**

```
原始输入：
  "我们做了一款面向慢性病患者的用药管理 App，想在二三线城市通过社区医院推广，
   主要想搞清楚哪类患者最可能坚持用这个 App 来辅助服药"

推导过程：
  · scenario → "用药管理 App，含用药提醒和复诊预约，通过社区医院渠道推广"
  · outcome_y → "患者用药依从意愿"
    （"坚持用 App 辅助服药" = 用药依从行为，App 是手段，依从是目标）
  · population → "中国二三线城市慢性病患者，40~65岁，社区医院就诊人群"
  · domain → "healthcare"（关键词：慢性病、患者、用药、社区医院）
  · purpose → "target_identification"（"哪类患者最可能"）
    注：同时建议运行 barrier_analysis（了解阻力来源对产品更有价值）
```

### 7.2 Step 1：维度生成

```
输入：ResearchSpec + DomainTemplate

流程：
  1. 加载领域模板的 dimension_skeleton
  2. LLM 对每个骨架维度做场景特化：
     · 将通用名称具体化（"数字工具接受度"在医疗场景下 →
       "40~65岁慢病患者使用智能手机 App 的意愿和能力"）
     · 生成符合目标人群的低端/高端锚点例句
     · 说明与 outcome_y 的因果路径
  3. LLM 判断是否需要补充骨架外的场景特有维度
  4. 验证：
     · 维度间两两 Spearman 相关系数 < 0.4（正交性）
     · 维度描述中不含 outcome_y 关键词（无循环）
     · 每个维度有清晰的低端/高端锚点（可 persona 化）

输出：List[DimensionAxis]（通常 5~8 个）
```

### 7.3 Step 1.5：问卷生成

```
输入：ResearchSpec（scenario + outcome_y + population）

流程：
  1. LLM 根据 outcome_y 识别对应的具体行为目标（behavior_goal）
     例："患者用药依从意愿" → "坚持按时按量服药，不自行停药或减量"
  2. LLM 生成 12 道 TPB 问卷题（4题×3维度）：
     · 第一人称陈述句，语言匹配 population
     · 每个维度至少1道反向计分题
     · 题目聚焦行为本身，不涉及 scenario 产品的主观评价
  3. 验证：维度纯净性、反向题自然性、人群匹配度
     · 不通过则重新生成（最多3次）
  4. 锁定问卷，后续所有 persona 共用

输出：Questionnaire（整个研究唯一，固定不变）
成本：1次 LLM 调用
```

### 7.4 Step 2：初始种群生成

```
输入：List[DimensionAxis]，Questionnaire，n_initial

流程：
  1. Sobol 序列在 D 维 [0,1]^D 空间生成 n_initial 个采样点
  2. 对每个采样点，LLM 生成 persona narrative（N₀次并行调用）：
     · 输入：各维度取值 + 低端/高端锚点参考
     · 输出：narrative（150~300字）
  3. 一致性检查：narrative 是否与 dimension_values 一致
     · 不一致则重新生成（最多3次）
  4. 每个 persona 作答问卷（N₀次并行调用）：
     · LLM 扮演 persona 对 12道题打 1~5 分 + 推理
  5. 纯数学计算 TPBScores → Y 值（无额外 LLM 调用）
  6. 初始化 Archive，按 Y 值放入对应 cell

输出：初始 Archive（覆盖不均匀，待进化循环填充）
成本：n_initial × 2次 LLM 调用（narrative + 问卷作答）
```

### 7.5 Step 3：MAP-Elites 进化循环

详见第 4 节，核心逻辑（Y 评估改为问卷作答）：

```
while not archive.is_satisfied() and gen < max_generations:
    sparse_cells = archive.select_sparse_cells(batch_size)
    for cell in sparse_cells:
        parent = archive.select_parent(cell, strategy)
        child  = mutate(parent, cell.y_range, mutation_type)
        # Y 评估：问卷作答（1次 LLM）+ 纯数学计算
        child.questionnaire_response = answer_questionnaire(child, questionnaire)
        child.tpb_scores = compute_tpb(child.questionnaire_response, questionnaire)
        child.y_value    = child.tpb_scores.intention
        archive.try_insert(child)
    gen += 1
```

### 7.6 Step 4：归因分析

```
输入：Archive（满足覆盖条件的 persona 库）

流程：
  1. 构建数据矩阵 X（N × D）和 Y 向量（N,）
     · 对 Archive 中 persona 做加权采样，确保 Y 区间均匀表示

  2. 特征重要性
     · Spearman 相关系数（线性关联）
     · 随机森林 feature importance（非线性）
     · SHAP 值（方向性 + 局部解释）

  3. 分群分析
     · K-means（K=3~5，按 Y 区间划分）
     · 每群计算维度中位数、TPB 分项均值、关键特征

  4. 交互效应
     · 优先检验 DomainTemplate.known_interactions
     · 对重要维度对做 2D SHAP 交互图

  5. 临界条件
     · 对最重要维度：扫描维度值从 0→1 时 Y 的变化，识别跳升点

  6. 建议生成
     · LLM 将归因结论转化为可操作的产品/营销建议

输出：AttributionResult
```

---

## 8. 端到端示例（医疗 App 场景）

### 输入

```
"我们做了一款面向慢性病患者的用药管理 App，想在二三线城市
 通过社区医院推广，主要想搞清楚哪类患者最可能坚持用这个 App"
```

### Step 0 推导结果

```yaml
scenario:   "用药管理 App，含用药提醒和复诊预约，通过社区医院渠道推广"
outcome_y:  "患者用药依从意愿"
population: "中国二三线城市慢性病患者，40~65岁"
domain:     "healthcare"
purpose:    "target_identification"
```

### Step 0 → Step 1 生成的 7 个维度

```
D1 疾病感知严重性
   低端："觉得血压稍高，无症状不在意"
   高端："曾因血压急性发作住院，高度警觉"

D2 医患信任度
   低端："走程序拿处方，不怎么信医生"
   高端："医生说什么就做什么，绝对信任"

D3 现有用药习惯稳定性
   低端："经常忘记吃药，自觉无副作用就停药"
   高端："多年固定仪式，不需要任何提醒"

D4 家庭督促环境
   低端："独居或家人不关心，无人提醒"
   高端："子女/配偶每日检查，高度关注"

D5 数字工具接受度
   低端："连微信都费劲，对 App 强烈抵触"
   高端："常用手机 App，对新工具接受度高"

D6 副作用感知强度
   低端："高度担忧副作用，经常自行减量"
   高端："完全信任医嘱用药，无明显顾虑"

D7 经济约束强度
   低端："医疗费用占比高，会因费用减少用药"
   高端："经济压力小，费用不影响用药决策"
```

### Step 1.5 生成的问卷（节选）

```
behavior_goal: "坚持每天按时按量服药，不自行停药或减量"

q01 [态度]       我觉得按时服药对控制血压/血糖有明显帮助
q02 [态度]       坚持用药是我愿意为自己健康做的事
q03 [态度-反向]  我觉得每天记着吃药太麻烦了，不值得这么坚持
q04 [态度]       用手机 App 提醒我吃药，能让我觉得更踏实

q05 [社会规范]   我的家人非常希望我能坚持按时服药
q06 [社会规范]   我的医生很强调坚持用药的重要性，我不想让他失望
q07 [社会规范-反] 周围很少有人像我这样认真对待每天吃药这件事
q08 [社会规范]   如果我漏服药，家人会担心，我不想让他们操心

q09 [PBC]        我相信自己能养成每天定时服药的习惯
q10 [PBC]        就算出门在外，我也有办法记得按时吃药
q11 [PBC-反向]   我经常因为忘记或嫌麻烦而漏服药，很难改掉这个习惯
q12 [PBC]        用 App 提醒的话，我觉得自己完全能做到按时用药
```

### Step 2 示例：Persona P_042 的问卷作答

```
P_042 基本特征（维度值）：
  D1 疾病感知严重性: 0.85  D4 家庭督促环境: 0.28
  D2 医患信任度:    0.18  D5 数字工具接受度: 0.12
  D3 用药习惯稳定性: 0.60  D6 副作用感知: 0.72

Narrative：
  "老陈，62岁，退休工人，有高血压十多年了。曾经因血压飙升住过一次院，
   从那以后知道要吃药，但觉得医生就是例行公事，不怎么信那套。平时一个
   人住，孩子在外地，没人管。手机主要用来打电话和发语音，装了个微信也
   只是偶尔看看，对 App 没什么兴趣，觉得捣鼓这些费事。担心长期吃降压
   药对肾不好，有时候血压正常了就自己停几天。"

问卷作答：
  q01: 4分 "住过院之后知道不吃药不行"
  q02: 3分 "知道是应该坚持，但有时候就算了"
  q03: 4分 "确实烦，出门忘带就忘了"
  q04: 1分 "不想装这些 App，不会用"
  q05: 2分 "孩子在外地，老伴不太管这些"
  q06: 2分 "医生就是走程序的，说了也就那样"
  q07: 3分 "也不知道别人咋样，反正我不太行"
  q08: 2分 "家里没人特别在乎"
  q09: 3分 "以前能坚持一段时间，就是容易忘"
  q10: 2分 "出门经常忘带药"
  q11: 4分 "确实经常忘，尤其是感觉没症状的时候"
  q12: 1分 "根本不会用这种 App"

TPB 得分（纯数学计算）：
  Attitude    = mean([4,3,2,1]) / 4 = 0.50  ← q03反向: 6-4=2
  Social Norm = mean([2,2,3,2]) / 4 = 0.31  ← q07反向: 6-3=3
  PBC         = mean([3,2,2,1]) / 4 = 0.25  ← q11反向: 6-4=2
  Y = 0.30×0.50 + 0.35×0.31 + 0.35×0.25 = 0.15 + 0.11 + 0.09 = 0.35
```

### Step 3 进化过程示意

```
代数 0（初始 Sobol，n=100）：
  cell_0 [Y<0.2]:   8 personas   ← 严重稀疏
  cell_1 [Y∈0.2-0.4]: 19 personas ← 稀疏
  cell_2 [Y∈0.4-0.6]: 38 personas
  cell_3 [Y∈0.6-0.8]: 27 personas
  cell_4 [Y>0.8]:    8 personas   ← 严重稀疏

代数 5（+50次变异）：
  cell_0: 24 personas  ← 显著改善
  cell_4: 22 personas  ← 显著改善
  典型变异轨迹：
    P_00_003（Y=0.38）× narrative_mutation → P_05_082（Y=0.16，目标cell_0）
    操作：降低 D5（数字接受度）和 D4（家庭督促），重写叙事为独居老人

代数 18（收敛）：
  所有 cell ≥ 30 personas，Coverage = 100%
  总 LLM 调用次数：约 320 次（含变异 + Y 评估）
```

### Step 4 归因输出（摘要）

```
特征重要性（SHAP 均值绝对值）：
  1. 数字工具接受度  0.28  正向  ← 最大阻力来源
  2. 医患信任度      0.22  正向
  3. 副作用感知      0.19  负向（感知越强，Y越低）
  4. 家庭督促环境    0.17  正向
  5. 疾病感知严重性  0.11  正向（有天花板效应）

关键交互效应：
  家庭督促高 × 数字接受低 → Y 不降反升（+0.18）
  解释：家人代操作 App，绕过数字能力障碍

临界条件：
  数字接受度 > 0.45 时，Y 出现跳升（+0.23）
  → 存在明确的数字能力门槛，低于此门槛的用户几乎不可能自主使用

分群：
  强阻力群（Y<0.3，n=38）：数字接受极低 + 无家庭支持
  家庭驱动群（Y=0.5~0.7，n=62）：数字接受中等，家庭督促强
  主动依从群（Y>0.7，n=45）：高医患信任 + 高疾病感知 + 低副作用顾虑

建议：
  1. 【最高优先级】设计"家人代操作"模式：将 App 主要用户从患者转移到子女/配偶
  2. 在 App 内增加权威医生解答副作用模块，降低副作用顾虑（直接命中第3大驱动）
  3. 渠道建议：由医生在复诊时主动推荐 App（医患信任是高杠杆变量，效率 > 广告渠道）
```

---

## 9. 目录结构

```
consumer_research/
│
├── docs/
│   └── consumer_research_framework.md     ← 本文件
│
├── src/
│   └── consumer_research/
│       │
│       ├── schemas/                         # 核心数据结构（Pydantic）
│       │   ├── research_spec.py            # ResearchSpec
│       │   ├── domain_template.py          # DomainTemplate
│       │   ├── dimension.py                # DimensionAxis
│       │   ├── persona.py                  # Persona, TPBScores
│       │   ├── archive.py                  # Archive, ArchiveCell, CoverageStats
│       │   └── attribution.py              # AttributionResult 及子类
│       │
│       ├── templates/                       # 领域模板库（YAML）
│       │   ├── healthcare.yaml
│       │   ├── consumer_tech.yaml
│       │   ├── fintech.yaml
│       │   ├── education.yaml
│       │   ├── retail.yaml
│       │   ├── policy.yaml
│       │   └── general.yaml               # 通用兜底模板
│       │
│       ├── pipeline/                        # 主流程各步骤
│       │   ├── normalizer.py              # Step 0：自然语言 → ResearchSpec
│       │   ├── dim_generator.py           # Step 1：维度生成
│       │   ├── questionnaire_builder.py   # Step 1.5：TPB 问卷生成与验证
│       │   ├── persona_generator.py       # Step 2：初始种群生成
│       │   ├── map_elites/                # Step 3：MAP-Elites 进化循环
│       │   │   ├── __init__.py
│       │   │   ├── archive.py             #   Archive 管理
│       │   │   ├── selection.py           #   亲本选择策略
│       │   │   └── mutation.py            #   变异算子（numeric/narrative/crossover）
│       │   ├── y_evaluator.py             # 问卷作答 + TPB 纯数学计算
│       │   └── analyzer.py                # Step 4：归因分析
│       │
│       ├── llm/                             # LLM 调用封装
│       │   ├── client.py                  # LangChain / OpenRouter 客户端
│       │   ├── prompts/                    # Prompt 模板（Jinja2）
│       │   │   ├── normalize.jinja2       # Step 0
│       │   │   ├── dim_generate.jinja2    # Step 1
│       │   │   ├── questionnaire_build.jinja2  # Step 1.5：生成12道题
│       │   │   ├── persona_narrative.jinja2    # Step 2：初始 narrative 生成
│       │   │   ├── questionnaire_answer.jinja2 # Step 2/3：persona 作答问卷
│       │   │   ├── mutation_numeric.jinja2     # Step 3：数值变异后 narrative 重写
│       │   │   └── mutation_narrative.jinja2   # Step 3：叙事改写变异
│       │   └── cache.py                   # 调用缓存（SQLite）
│       │
│       ├── sampling/
│       │   └── sobol.py                   # Sobol 序列生成
│       │
│       └── analysis/
│           ├── importance.py              # SHAP + 相关系数
│           ├── clustering.py              # K-means 分群
│           ├── interaction.py             # 交互效应检验
│           └── report.py                 # 洞察报告生成（Markdown / JSON）
│
├── experiments/
│   └── healthcare_medication.py           # 医疗 App 完整实验脚本
│
├── data/
│   ├── cache/                             # LLM 调用缓存
│   └── outputs/
│       └── {experiment_id}/
│           ├── research_spec.json
│           ├── dimensions.json
│           ├── questionnaire.json         # 本次研究的固定问卷
│           ├── archive/
│           │   ├── personas.json          # 全量 persona 数据（含问卷作答）
│           │   └── evolution_log.json     # 每代 Coverage 变化
│           └── attribution_report.md
│
└── tests/
    ├── test_schemas.py
    ├── test_normalizer.py
    ├── test_dim_generator.py
    ├── test_map_elites.py
    └── test_y_evaluator.py
```

---

## 10. 设计决策记录

| 决策 | 选择 | 理由 | 放弃的选项 |
|------|------|------|-----------|
| **入口** | 自然语言 → LLM 推导三元组 | 降低使用门槛，研究方法论不应是用户的负担 | 要求用户直接填写结构化 ResearchSpec（门槛高） |
| **Archive 维度** | 默认 1D（按 Y），可选 2D | 1D 简单可靠；2D 捕捉交互边界但指数增大 cell 数量 | 高维 Archive（维度灾难） |
| **变异算子** | 三种并存（数值/叙事/交叉），按比例采样 | 不同算子适合不同搜索阶段，组合更鲁棒 | 只用数值变异（忽视语义层面的 persona 改写能力） |
| **维度生成** | 领域模板骨架 + LLM 特化 | 可靠性 + 灵活性平衡 | 纯 LLM 生成（正交性不稳定）；纯固定模板（扩展性差） |
| **适应度定义** | narrative 质量分（而非 Y 值本身） | 目标是覆盖 Y 空间，而非最大化 Y；质量分防止低质量 persona 占据 cell | 以 Y 值为适应度（会导致所有 persona 趋向极端 Y 值） |
| **Y 评估方式** | LLM 生成问卷 → persona 作答 → 纯数学计算 | 具体题目锚定 LLM 输出，比直接打分稳定；每题推理可追溯；问卷固定保证跨 persona 可比性 | 直接让 LLM 输出 0~1 分（无锚定，不稳定，不可解释）|
| **问卷共用策略** | 整个研究生成1份问卷，所有 persona 复用 | 同一把尺子量所有 persona，成本低（只需1次生成），跨 persona 比较有效 | 每个 persona 生成专属问卷（成本高，且无法跨 persona 比较） |

---

## 参考

- Ajzen, I. (1991). Theory of Planned Behavior
- Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis*
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions (SHAP). *NeurIPS*
- Owen, A. B. (1998). Scrambling Sobol' and Niederreiter–Xing Points
- 项目内部文档：`syn_persona.md`，`innovation_diffusion_sim.md`

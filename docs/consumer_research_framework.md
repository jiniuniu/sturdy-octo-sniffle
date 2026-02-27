# 消费者行为研究框架

## 设计文档 v0.1

---

## 1. 背景与动机

### 1.1 现有方法的瓶颈

用户行为研究（用户访谈、焦点小组、问卷调研）是产品设计和营销决策的核心输入，但现有方法存在几个根本性瓶颈：

**成本与周期问题**
招募目标人群、设计问卷、执行访谈、清洗数据——一个标准用户研究项目通常需要 4-8 周和数万元预算，在产品早期迭代中几乎不可行。

**样本偏差问题**
愿意接受访谈的用户本身就是偏样本（高参与意愿、高表达欲），极端情形用户（高阻力群体、边缘情形）在真实调研中严重欠采样，而这些人恰恰是洞察的富矿。

**变量识别问题**
研究者需要事先知道"测什么"（哪些消费者特征与 Y 相关），但这本身就是研究要回答的问题，形成循环。传统方法依赖领域经验，遗漏关键变量的风险高。

**从数据到洞察的黑箱问题**
即使有了数据，从"A 组和 B 组得分不同"到"产品应该怎么做"之间，往往缺乏透明的推导链路。

### 1.2 核心洞察

本框架建立在三个核心观察上：

**观察 1：Y 的分布比 Y 的均值更有价值**
"平均而言，用户依从意愿是 0.6"没有价值。"什么特征的用户依从意愿高，什么特征的用户依从意愿低，以及为什么"才是决策输入。这要求样本在 Y 上均匀分布，覆盖完整区间。

**观察 2：描述"是谁"和测量"怎么反应"必须解耦**
如果用"对健康 App 的接受度"来描述用户，再用 LLM 预测其"对健康 App 的采纳意愿"，结论是自证的。正确做法是用**不含 Y 内容的维度**描述用户，再独立评估 Y。

**观察 3：领域先验知识可以显著压缩搜索空间**
"医疗"领域的消费者行为有相对稳定的结构（依从行为受医患关系、家庭压力、副作用感知等因素驱动），预先编码这些结构，比每次让 LLM 从零生成维度更可靠。

### 1.3 目标

构建一个系统，使研究者只需提供：

```
scenario:   "医疗App推广"
outcome_y:  "患者用药依从意愿"
population: "中国二三线城市慢病患者，40~65岁"
```

系统自动返回：
1. 针对该场景的消费者多样性维度（研究设计）
2. 合成 persona 群体及其 Y 值评估
3. 驱动 Y 变化的消费者特征归因（洞察）

---

## 2. 问题 Formulation

### 2.1 研究问题的标准结构

任何消费者行为研究问题都可以被规范化为三元组：

```
ResearchSpec = (scenario, outcome_y, population)
```

| 字段 | 含义 | 示例 |
|------|------|------|
| `scenario` | 产品/干预/政策的上下文描述 | "医疗 App 推广，核心功能是用药提醒和复诊管理" |
| `outcome_y` | 要解释或预测的目标变量，必须是可操作的行为或态度 | "患者用药依从意愿（坚持按时按量服药的意愿）" |
| `population` | 目标研究人群的范围和特征约束 | "中国二三线城市慢病患者，40~65岁，以高血压/糖尿病为主" |

**outcome_y 的质量标准：**
- 可操作（能对应具体行为，而非模糊情感）
- 有方向（Y 高/低分别意味着什么）
- 与 scenario 有因果关联（而非相关）
- 不循环（不能用 Y 的内容作为描述用户的维度）

**反例（需要在 Step 0 中自动拒绝或纠正）：**

| 原始输入 | 问题 | 纠正后 |
|----------|------|--------|
| "用户满意度" | 太模糊，不可操作 | "30天内主动打开 App 超过 15 次的意愿" |
| "对 App 的接受度" | 与 scenario 循环 | "用药依从性的改变量（vs 不用 App）" |
| "是否喜欢这个功能" | 态度而非行为 | "是否会向家庭成员推荐此 App" |

### 2.2 研究目的的分类

不同的研究目的对应不同的分析重点：

| 研究目的 | 核心问题 | 分析侧重 |
|----------|----------|----------|
| **目标人群识别** | 谁最可能 Y 值高？ | Y 高分群的特征画像 |
| **阻力归因** | 为什么 Y 值低？ | Y 低分群的共性特征 + 阻力来源 |
| **临界条件识别** | 什么条件下 Y 从低变高？ | 维度交互效应 + 临界值 |
| **干预优先级** | 改变哪个特征对 Y 影响最大？ | 单维度边际效应 |
| **分群比较** | 不同子人群之间 Y 的驱动因素有何不同？ | 分层分析 |

### 2.3 形式化定义

给定 ResearchSpec，研究问题被形式化为：

```
找到函数 f，使得：

Y_i = f(X_i) + ε_i

其中：
  X_i = (x_1, x_2, ..., x_D)  ← persona i 在 D 个维度上的取值
  Y_i ∈ [0, 1]                ← persona i 的 outcome 评分
  ε_i                         ← LLM 评估的随机误差

研究目标：估计 f 的结构，识别哪些 x_d 对 Y 的方差贡献最大
```

这不是一个预测问题（不是要拟合真实数据），而是一个**敏感性分析**问题：在合理假设下，Y 对哪些消费者特征最敏感？

---

## 3. 项目设计框架

### 3.1 全景架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户输入层                                 │
│   scenario / outcome_y / population / research_purpose              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Step 0：研究规范化                            │
│   · 解析 ResearchSpec                                               │
│   · 验证 outcome_y 的质量                                           │
│   · 匹配领域模板（Domain Template Matching）                        │
│   · 输出：标准化 ResearchSpec + 领域模板                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 1：维度生成                                │
│   · 以领域模板为骨架                                                │
│   · LLM 根据 scenario + population 特化维度                        │
│   · 人工/自动验证正交性和与 Y 的理论相关性                          │
│   · 输出：D 个 DimensionAxis（含低端/高端锚点描述）                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 2：Persona 生成                            │
│   · Sobol 序列在 D 维空间均匀采样 N 个点                            │
│   · LLM 将每个采样点转化为 persona 叙事文本                         │
│   · 验证：文本内容与维度数值一致性检查                              │
│   · 输出：N 个 Persona（含维度数值 + 叙事文本）                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 3：Y 值评估                                │
│   · LLM 扮演 persona 基于 TPB 框架评估 Y                            │
│   · 输出结构化评分：Attitude / SocialNorm / PBC / Friction          │
│   · 计算最终 Y                                                      │
│   · 验证：Y 的分布检查，必要时定向补充                              │
│   · 输出：每个 persona 的 TPB 评分 + Y 值                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 4：归因分析                                │
│   · 特征重要性：相关分析 + SHAP 值                                  │
│   · 分群分析：K-means 按 Y 区间聚类 persona                         │
│   · 交互效应：关键维度交叉分析                                      │
│   · 临界条件：Y 从低到高的翻转条件识别                              │
│   · 输出：AttributionResult（洞察报告）                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 领域模板系统

领域模板（Domain Template）是本框架的核心设计，它将领域专家知识结构化，避免每次从零让 LLM 生成维度。

**当前支持的领域：**

| 领域 | 典型 scenario | 典型 outcome_y |
|------|--------------|----------------|
| `healthcare` | 医疗 App、慢病管理、用药提醒 | 用药依从性、复诊率、健康行为改变 |
| `consumer_tech` | 新 App、硬件产品、订阅服务 | 首次付费、7日留存、口碑传播 |
| `fintech` | 理财 App、信贷产品、保险 | 首次入金、风险决策行为、续费 |
| `education` | 在线课程、职业培训、K12 | 完课率、付费续费、主动探索 |
| `retail` | 新品上市、品牌进入新市场 | 复购意愿、价格敏感临界、推荐意愿 |
| `policy` | 政策推广、公共卫生干预 | 自愿参与率、行为改变持续性 |

**以 `healthcare` 模板为例：**

```
DomainTemplate(
    domain = "healthcare",

    # 该领域下通常重要的维度骨架（LLM 在此基础上特化）
    dimension_skeleton = [
        "疾病感知严重性",        # 患者认为自己的病有多严重
        "医患信任度",            # 对医生建议的接受和信任程度
        "现有治疗方案依赖深度",  # 对现有用药习惯的依赖
        "家庭支持环境",          # 家人是否督促、提醒、支持
        "数字工具接受度",        # 使用手机 App 的基础意愿和能力
        "副作用/风险感知",       # 对用药副作用的担忧程度
        "经济约束强度",          # 医疗费用对行为的限制
    ],

    # 该领域下 TPB 各分项的典型权重（可被场景覆盖）
    tpb_weights = {
        "attitude": 0.30,
        "social_norm": 0.35,   # 医疗领域家庭/医生社会规范影响大
        "pbc": 0.35,           # 感知能力（能不能坚持用药）影响大
    },

    # 该领域下已知的高价值交互效应（分析时优先检验）
    known_interactions = [
        ("疾病感知严重性", "副作用感知"),   # 两者交叉决定用药态度
        ("家庭支持环境", "数字工具接受度"), # 家人帮助可弥补数字能力不足
    ],

    # outcome_y 的常见类型和对应的 Y 计算方式
    y_templates = {
        "用药依从意愿": "按时按量服药的主观意愿，包含坚持意愿和应对遗忘的主动性",
        "复诊意愿":     "主动预约下次复诊的意愿，而非被动等待",
        "App持续使用":  "下载后 30 天内仍主动打开 App 的意愿",
    }
)
```

### 3.3 维度生成的约束条件

Step 1 生成的每个维度必须满足：

| 约束 | 说明 | 验证方法 |
|------|------|----------|
| **与 Y 理论相关** | 存在合理的因果或相关路径 | LLM 生成时要求给出因果路径说明 |
| **不包含 Y 内容** | 维度描述中不能出现 outcome_y 的关键词 | 关键词过滤 + 人工审查 |
| **低维间相关** | 两两相关系数 < 0.4 | 生成后用 Pearson/Spearman 检验 |
| **可被 persona 化** | 能用具体的人物场景文字表达 | 要求为每个维度写出低端/高端锚点例句 |
| **语义可区分** | 不同值域内的 persona 应有显著不同的描述 | 锚点例句对比检验 |

---

## 4. 核心数据结构

### 4.1 输入层

```python
class ResearchSpec(BaseModel):
    scenario: str
    # 产品/干预的上下文，自然语言
    # 例："一款面向慢病患者的用药管理 App，核心功能包括用药提醒、
    #      复诊预约、健康数据追踪，目标在二三线城市通过医院渠道推广"

    outcome_y: str
    # 目标变量，自然语言，需满足可操作性标准
    # 例："患者在使用 App 后坚持按时按量服药的意愿（用药依从意愿）"

    population: str
    # 目标人群的范围约束
    # 例："中国二三线城市慢病患者，40~65岁，以高血压/糖尿病为主，
    #      中低教育程度，家庭月收入 3000~8000 元"

    research_purpose: Literal[
        "target_identification",   # 找目标人群
        "barrier_analysis",        # 找阻力来源
        "threshold_detection",     # 找临界条件
        "intervention_priority",   # 确定干预优先级
        "segment_comparison",      # 跨分群比较
    ] = "barrier_analysis"

    n_personas: int = 200
    # 生成的 persona 数量，影响分析精度和成本

    domain_hint: Optional[str] = None
    # 领域提示，如果用户知道所属领域可以指定，否则自动匹配
```

### 4.2 领域模板

```python
class DomainTemplate(BaseModel):
    domain: str
    # 领域标识符，如 "healthcare", "consumer_tech"

    dimension_skeleton: list[str]
    # 该领域的通用维度骨架，作为 LLM 生成维度的起点

    tpb_weights: dict[str, float]
    # TPB 三要素在该领域的默认权重
    # {"attitude": float, "social_norm": float, "pbc": float}

    known_interactions: list[tuple[str, str]]
    # 已知的高价值维度交互对，在分析时优先检验

    y_templates: dict[str, str]
    # 常见 outcome_y 类型的标准化描述
```

### 4.3 维度轴

```python
class DimensionAxis(BaseModel):
    name: str
    # 维度名称，简洁，2~6个字
    # 例："疾病感知严重性"

    description: str
    # 维度含义的完整说明
    # 例："患者对自身慢病严重性的主观感知程度，
    #      影响其对治疗行为的紧迫感和投入意愿"

    low_anchor: str
    # 取值接近 0 时的典型 persona 场景描述（1~2句话）
    # 例："觉得自己血压只是稍微高一点，没什么大碍，
    #      没有症状就感觉没有问题，不太在意医生的叮嘱"

    high_anchor: str
    # 取值接近 1 时的典型 persona 场景描述（1~2句话）
    # 例："曾经因血压飙升住院，对此非常恐惧，
    #      把控制血压当作首要生活任务，高度配合医生方案"

    causal_path_to_y: str
    # 该维度影响 outcome_y 的因果路径说明
    # 例："疾病感知严重性越高 → 态度中的感知价值越高（更认为用药有必要）
    #      → TPB Attitude 分项上升 → Y（用药依从意愿）提升"

    domain_source: str
    # 来源：来自领域模板骨架还是场景特化生成
    # "template" | "scenario_specific"
```

### 4.4 合成 Persona

```python
class Persona(BaseModel):
    persona_id: str
    # 唯一标识，格式 "P_{三位数字}"，例 "P_042"

    dimension_values: dict[str, float]
    # 各维度的取值，均为 [0, 1]
    # 例：{"疾病感知严重性": 0.82, "医患信任度": 0.45, ...}

    narrative: str
    # 由 LLM 基于 dimension_values 生成的 persona 叙事文本
    # 包含：基本背景、典型日常场景、对 scenario 的具体态度
    # 长度：150~300字

    tpb_scores: TPBScores
    # TPB 各分项评分

    y_value: float
    # 最终 outcome_y 评分，[0, 1]

    y_reasoning: str
    # LLM 对该 persona Y 值的推理过程摘要（1~3句话）


class TPBScores(BaseModel):
    attitude: float          # 对采纳行为的评价性态度，[0, 1]
    social_norm: float       # 感知到的社会压力/支持，[0, 1]
    pbc: float               # 感知行为控制（能不能做到），[0, 1]
    intention: float         # 加权意向分，[0, 1]
    friction: float          # 摩擦系数（阻力），[0, 1]，越高阻力越大
```

### 4.5 归因结果

```python
class AttributionResult(BaseModel):
    research_spec: ResearchSpec

    # ── 全局特征重要性 ──────────────────────────────────────────────
    feature_importance: list[FeatureImportance]
    # 按重要性降序排列

    # ── 分群画像 ────────────────────────────────────────────────────
    segments: list[PersonaSegment]
    # 通常按 Y 区间分为 3~5 个段

    # ── 交互效应 ────────────────────────────────────────────────────
    interaction_effects: list[InteractionEffect]
    # 重要的维度间交互效应

    # ── 临界条件 ────────────────────────────────────────────────────
    threshold_conditions: list[ThresholdCondition]
    # Y 从低到高的翻转条件

    # ── 可操作建议 ──────────────────────────────────────────────────
    recommendations: list[Recommendation]


class FeatureImportance(BaseModel):
    dimension_name: str
    importance_score: float      # 归一化重要性，[0, 1]
    shap_mean_abs: float         # SHAP 均值绝对值
    correlation_with_y: float    # 与 Y 的 Spearman 相关系数
    direction: Literal["positive", "negative", "nonlinear"]
    interpretation: str          # 1句话解释


class PersonaSegment(BaseModel):
    segment_id: str              # 例 "S_high" / "S_low" / "S_mid"
    y_range: tuple[float, float] # 例 (0.7, 1.0)
    size: int                    # 该段 persona 数量
    label: str                   # 例 "强依从群体"
    centroid: dict[str, float]   # 各维度的中位数值
    key_characteristics: list[str]  # 2~4条文字描述该群体的核心特征
    tpb_bottleneck: str          # 该群体 TPB 的主要限制分项


class InteractionEffect(BaseModel):
    dim_a: str
    dim_b: str
    effect_description: str      # 交互效应的文字描述
    effect_strength: float       # 交互效应强度


class ThresholdCondition(BaseModel):
    description: str             # "当 X 从低变高时，Y 出现显著跳升"
    trigger_dimension: str
    threshold_value: float
    y_change: float              # Y 的变化量


class Recommendation(BaseModel):
    priority: int                # 1 = 最高优先级
    target_segment: str          # 针对哪个分群
    action: str                  # 建议的具体行动
    rationale: str               # 基于哪个归因结论
    expected_y_lift: Optional[float]  # 预期 Y 提升幅度（如可估算）
```

---

## 5. 流转逻辑

### 5.1 主流程

```
输入：ResearchSpec
         │
         ▼
  ┌─────────────────┐
  │  Step 0         │  研究规范化
  │  Normalizer     │
  └────────┬────────┘
           │  · 验证 outcome_y 是否可操作
           │  · 自动补全缺失字段
           │  · 匹配 DomainTemplate
           ▼
  ┌─────────────────┐
  │  Step 1         │  维度生成
  │  DimGenerator   │
  └────────┬────────┘
           │  · 以模板骨架为起点
           │  · LLM 结合 scenario + population 特化维度
           │  · 验证正交性 + 与 Y 的无循环性
           │  输出：List[DimensionAxis]（通常 5~8 个）
           ▼
  ┌─────────────────┐
  │  Step 2         │  Persona 生成
  │  PersonaGen     │
  └────────┬────────┘
           │  · Sobol 序列采样 N 个点
           │  · 每个点 → LLM 生成 narrative
           │  · 一致性检验：narrative 与 dimension_values 对齐
           │  输出：List[Persona]（narrative 已填，Y 待填）
           ▼
  ┌─────────────────┐
  │  Step 3         │  Y 值评估
  │  YEvaluator     │
  └────────┬────────┘
           │  · 每个 persona → LLM 推理 TPB 各分项
           │  · 计算 intention + friction → Y
           │  · Y 分布检验：各区间覆盖是否足够
           │  · 若某区间不足，定向补充 persona
           │  输出：List[Persona]（Y 值已填）
           ▼
  ┌─────────────────┐
  │  Step 4         │  归因分析
  │  Analyzer       │
  └────────┬────────┘
           │  · 特征重要性（相关系数 + SHAP）
           │  · K-means 聚类（按 Y 区间）
           │  · 交互效应检验（优先检验领域模板的 known_interactions）
           │  · 临界条件识别
           │  · 建议生成
           │  输出：AttributionResult
           ▼
输出：洞察报告（可导出为 JSON / Markdown）
```

### 5.2 Y 分布验证与补充逻辑

```
生成 N 个 persona 并评估 Y 后：

将 [0, 1] 分为 K=5 个区间，检查各区间的 persona 数量：

  区间          目标数量（N=200）   实际数量   状态
  [0.0, 0.2)       ≥ 30              12        ← 不足，需补充
  [0.2, 0.4)       ≥ 30              38        ✓
  [0.4, 0.6)       ≥ 30              72        ✓（过多，可下采样）
  [0.6, 0.8)       ≥ 30              55        ✓
  [0.8, 1.0]       ≥ 30              23        ← 不足，需补充

对不足的区间：
  1. 识别哪些维度组合会导致该区间 Y 值（基于已有数据回归）
  2. 在该区间的典型维度区域内额外采样 M 个点
  3. 生成对应 persona，补充到数据集

最终数据集的分析权重按区间均匀化调整，抵消分布不均影响
```

### 5.3 领域模板匹配逻辑

```
输入：scenario 文本

Step 1：关键词匹配
  · "医院"、"用药"、"慢病"、"患者"、"医生" → healthcare
  · "贷款"、"理财"、"保险"、"存款"、"投资" → fintech
  · "课程"、"培训"、"学习"、"教育"、"考试" → education
  ···

Step 2：若关键词不明确，LLM 分类
  输入：scenario 描述
  输出：domain 标签 + 置信度

Step 3：若置信度 < 0.7，使用通用模板（general）
  通用模板包含所有领域共有的基础维度骨架：
  · 痛点意识度
  · 现有方案依赖深度
  · 新产品接受度
  · 外部约束强度
  · 付费心智
  · 问题归因方式
```

### 5.4 示例：医疗 App 场景的完整流转

**输入：**
```
scenario:   "一款用药提醒和复诊管理 App，目标通过二三线城市社区医院推广"
outcome_y:  "患者用药依从意愿（坚持按时按量服药的意愿）"
population: "中国二三线城市慢病患者，40~65岁，高血压/糖尿病为主"
research_purpose: "barrier_analysis"
n_personas: 200
```

**Step 0 输出：**
```
matched_domain: "healthcare"
y_quality_check: PASS
y_clarification: "Y 定义为：患者在知晓 App 功能后，主观表达会坚持使用 App 辅助
                  按时按量服药的意愿强度，Y=1 为明确表示会坚持，Y=0 为明确拒绝"
```

**Step 1 输出（7个维度）：**
```
D1: 疾病感知严重性
    低端锚点："觉得血压只是数字问题，没症状就不在意"
    高端锚点："曾住院，把控制血压视为头等大事"

D2: 医患信任度
    低端锚点："觉得医生只是走程序，处方随便拿，不太信"
    高端锚点："把主治医生的话当圣旨，医生让咋做就咋做"

D3: 现有用药习惯稳定性
    低端锚点："经常忘记吃药，自己觉得没副作用就停药"
    高端锚点："多年来已形成固定用药仪式，不需要任何提醒"

D4: 家庭督促环境
    低端锚点："独居或家人不关心健康，无人提醒"
    高端锚点："子女/配偶高度关注，每天检查是否按时吃药"

D5: 数字工具接受度
    低端锚点："连微信都用得费劲，对手机 App 有强烈抵触"
    高端锚点："经常使用手机 App，对新工具接受度高"

D6: 副作用/用药顾虑
    低端锚点："非常担心长期用药的副作用，经常自行减量"
    高端锚点："完全信任医嘱用药方案，无明显顾虑"

D7: 经济约束强度
    低端锚点："医疗费用占家庭收入比例高，会因费用减少用药频率"
    高端锚点："经济压力小，费用不影响用药决策"
```

**Step 3 评估（示例 3 个 persona）：**

```
P_042（疾病感知高，医患信任低，数字接受低）
  D1=0.85, D2=0.18, D3=0.60, D4=0.30, D5=0.12, D6=0.72, D7=0.45
  → Attitude=0.61  [重视用药，但不需要 App 的额外帮助]
  → SocialNorm=0.28 [家人不怎么管，医生也没特别推荐]
  → PBC=0.19       [觉得自己搞不定 App，会出错]
  → Friction=0.68  [有固定习惯，切换成本高]
  → Y = 0.21       ← 低依从意愿

P_117（疾病感知低，家庭督促高，数字接受高）
  D1=0.22, D2=0.55, D3=0.35, D4=0.92, D5=0.80, D6=0.45, D7=0.60
  → Attitude=0.42  [不太觉得自己病重，动力不足]
  → SocialNorm=0.78 [子女强烈建议用，有压力]
  → PBC=0.85       [会用 App，觉得能做到]
  → Friction=0.25
  → Y = 0.67       ← 中高依从意愿（家庭驱动型）

P_183（疾病感知高，医患信任高，数字接受中）
  D1=0.90, D2=0.88, D3=0.40, D4=0.55, D5=0.55, D6=0.85, D7=0.70
  → Attitude=0.88
  → SocialNorm=0.62
  → PBC=0.58
  → Friction=0.15
  → Y = 0.82       ← 高依从意愿
```

**Step 4 归因输出（摘要）：**

```
特征重要性排序：
  1. 数字工具接受度（SHAP=0.28，正向）   ← 最强阻力来源
  2. 医患信任度    （SHAP=0.22，正向）
  3. 副作用顾虑    （SHAP=0.19，负向）
  4. 家庭督促环境  （SHAP=0.17，正向）
  5. 疾病感知严重性（SHAP=0.11，正向，但有天花板效应）

关键交互效应：
  · 家庭督促高 × 数字接受低 → Y 不降反升
    （家人可代为操作 App，弥补数字能力不足）

临界条件：
  · 数字接受度 > 0.45 时，Y 出现显著跳升（+0.23）

分群画像：
  · 强阻力群（Y<0.3，n=38）：数字接受极低 + 有固定用药习惯 + 无家庭支持
  · 家庭驱动群（Y=0.5~0.7，n=62）：数字接受中等，但家庭督促强
  · 主动依从群（Y>0.7，n=45）：高医患信任 + 高疾病感知 + 无副作用顾虑

可操作建议：
  1. 【最高优先级】针对强阻力群：设计"家人代操作"模式，
     将 App 使用者从患者本人转移到子女/配偶，绕过数字能力障碍
  2. 【次优先级】针对副作用顾虑高的群体：App 内增加
     "权威解答副作用"模块，引用医生声音，降低顾虑
  3. 【渠道建议】医患信任是高杠杆变量：在医院内由医生
     推荐 App，比任何广告渠道效率高 3x（基于 SocialNorm 系数）
```

---

## 6. 目录结构

```
consumer_research/
│
├── docs/
│   └── consumer_research_framework.md    ← 本文件
│
├── src/
│   └── consumer_research/
│       │
│       ├── __init__.py
│       │
│       ├── schemas/                       # 核心数据结构（Pydantic）
│       │   ├── __init__.py
│       │   ├── research_spec.py          # ResearchSpec
│       │   ├── domain_template.py        # DomainTemplate
│       │   ├── dimension.py              # DimensionAxis
│       │   ├── persona.py                # Persona, TPBScores
│       │   └── attribution.py            # AttributionResult 及子类
│       │
│       ├── templates/                     # 领域模板库（JSON/YAML）
│       │   ├── healthcare.yaml
│       │   ├── consumer_tech.yaml
│       │   ├── fintech.yaml
│       │   ├── education.yaml
│       │   ├── retail.yaml
│       │   ├── policy.yaml
│       │   └── general.yaml              # 通用模板（兜底）
│       │
│       ├── pipeline/                      # 主流程各步骤
│       │   ├── __init__.py
│       │   ├── normalizer.py             # Step 0：研究规范化 + 模板匹配
│       │   ├── dim_generator.py          # Step 1：维度生成
│       │   ├── persona_generator.py      # Step 2：Persona 生成 + 采样
│       │   ├── y_evaluator.py            # Step 3：Y 值评估 + 分布验证
│       │   └── analyzer.py               # Step 4：归因分析
│       │
│       ├── llm/                           # LLM 调用封装
│       │   ├── __init__.py
│       │   ├── client.py                 # LangChain / OpenRouter 客户端
│       │   ├── prompts/                   # 各步骤 Prompt 模板
│       │   │   ├── normalize.jinja2
│       │   │   ├── dim_generate.jinja2
│       │   │   ├── persona_narrative.jinja2
│       │   │   └── y_evaluate.jinja2
│       │   └── cache.py                  # 调用缓存（SQLite）
│       │
│       ├── sampling/                      # 采样工具
│       │   ├── __init__.py
│       │   └── sobol.py                  # Sobol 序列 + 定向补充逻辑
│       │
│       └── analysis/                      # 分析工具
│           ├── __init__.py
│           ├── importance.py             # SHAP + 相关系数
│           ├── clustering.py             # K-means 分群
│           ├── interaction.py            # 交互效应检验
│           └── report.py                 # 洞察报告生成（Markdown / JSON）
│
├── experiments/
│   └── healthcare_medication.py          # 医疗 App 示例实验脚本
│
├── data/
│   ├── cache/                            # LLM 调用缓存
│   └── outputs/                          # 实验输出
│       └── {experiment_id}/
│           ├── research_spec.json
│           ├── dimensions.json
│           ├── personas.json
│           └── attribution_report.md
│
└── tests/
    ├── test_schemas.py
    ├── test_normalizer.py
    ├── test_dim_generator.py
    └── test_y_evaluator.py
```

---

## 7. 设计决策记录

| 决策 | 选择 | 理由 | 放弃的选项 |
|------|------|------|-----------|
| 维度生成方式 | 领域模板骨架 + LLM 特化 | 可靠性 + 灵活性平衡，避免纯 LLM 生成的正交性问题 | 纯 LLM 生成（质量不稳定）；纯固定模板（扩展性差） |
| 采样方法 | Sobol 低差异序列 | 在高维空间中比随机采样和网格采样均匀 | 随机采样（覆盖不均）；拉丁超立方（等效但工具链更复杂） |
| Y 的均匀化策略 | 验证后定向补充 | 保留自然分布的信息，同时确保极端段有足够样本 | 强制均匀（失真）；不处理（低 Y 段欠采样） |
| TPB 权重 | 领域模板提供默认值，场景可覆盖 | 医疗与消费品的 TPB 结构显著不同，需要领域先验 | 固定全局权重（忽视领域差异）；完全 LLM 推断（不稳定） |
| 数据结构 | 全 Pydantic | 结构化输出强制约束 + 序列化 + 与 LLM 结构化输出直接对接 | dataclass（无验证）；dict（无类型保证） |

---

## 参考

- Ajzen, I. (1991). Theory of Planned Behavior
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis*
- Owen, A. B. (1998). Scrambling Sobol' and Niederreiter–Xing Points
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions (SHAP)
- 项目内部文档：`syn_persona.md`，`research_design_framework.md`

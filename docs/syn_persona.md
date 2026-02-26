# Persona-Based Startup Idea Validation System

## 设计文档 v0.4

---

## 1. 系统概述

用合成 persona 模拟多样化消费者，基于 TPB（计划行为理论）预测其对产品的接受度，从而在早期快速验证创业点子的风险边界。

**核心设计原则：**

- 输入维度（描述人是谁）与测量框架（TPB）完全解耦，避免循环自证
- 消费者描述维度由产品设计者预定义，不依赖 LLM 自由发挥
- 均匀采样覆盖极端情况，而非模拟真实分布

**技术栈：**

- LLM 调用：LangChain + OpenRouter
- 输出解析 + 内部数据流转：统一使用 Pydantic model（不使用 dataclass）
- 采样：`scipy.stats.qmc.Sobol`

---

## 2. 两套框架的关系

本系统涉及两套独立的评估框架，职责不同，不应混用。

**创始人评估框架**（想验证什么）是创业者关心的5个问题，其中4个可以通过用户模拟间接回答，1个无法通过用户模拟回答：

| 创始人维度           | 能否用用户模拟回答 | 对应消费者维度                  |
| -------------------- | ------------------ | ------------------------------- |
| 外部环境匹配度       | 部分能             | 新产品接受度 + 外部约束强度     |
| 需求真实性           | 能                 | 痛点意识度 + 问题归因方式       |
| 用户迁移成本         | 能                 | 现有方案依赖深度 + 外部约束强度 |
| 商业模式（付费侧）   | 部分能             | 付费心智                        |
| 竞争护城河（用户侧） | 部分能             | 现有方案依赖深度 + 新产品接受度 |
| ~~能不能做出来~~     | **不能**           | 技术/资源问题，超出用户模拟范围 |

**消费者描述框架**（生成谁）是用来构造多样化用户的6个正交维度，由产品设计者预定义，适用于所有创业场景：

| 维度             | 含义                                 |
| ---------------- | ------------------------------------ |
| 痛点意识度       | 用户是否意识到自己有问题需要解决     |
| 现有方案依赖深度 | 用户对当前解决方式的依赖程度         |
| 付费心智         | 用户为此类价值付费的意愿             |
| 新产品接受度     | 用户尝试新产品的积极性               |
| 外部约束强度     | 预算、家庭、时间等外部因素的限制程度 |
| 问题归因方式     | 用户认为问题该自己解决还是等别人解决 |

两套框架的关系：消费者描述框架是**输入**（描述人是谁），TPB 是**测量工具**（测这个人的反应），创始人评估框架是**解读视角**（从结果里读出什么）。三者在数据流中处于不同层次，互不干扰。

---

## 3. 整体数据流

```
┌─────────────────────────────────────────────────────────────────┐
│  输入层                                                          │
│  product_description (自然语言)                                  │
│  diversity_axes (预定义6维度)                                    │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  第一条线：人物生成                                               │
│                                                                  │
│  diversity_axes                                                  │
│       │                                                          │
│       ▼                                                          │
│  语义枚举值生成 (LLM)                                            │
│  { axis: [L1_label, L2_label, L3_label, L4_label] }             │
│       │                                                          │
│       ▼                                                          │
│  均匀采样 (正交/Sobol)                                           │
│  → N 个向量 [(L2, L1, L3, L4, L2, L1), ...]                    │
│       │                                                          │
│       ▼                                                          │
│  人物描述生成 (LLM)                                              │
│  → N 个 PersonaProfile (2~3句自然语言)                          │
└──────────────┬──────────────────────────────────────────────────┘
               │
               │  PersonaProfile ──────────────────────────┐
               │                                            │
               ▼                                            ▼
┌─────────────────────────────┐    ┌────────────────────────────────┐
│  第二条线：问卷生成           │    │  第二条线：LLM 模拟回答         │
│                              │    │                                │
│  product_description         │    │  persona_description           │
│       +                      │    │       +                        │
│  TPB 框架                    │    │  questionnaire                 │
│       +                      │    │       │                        │
│  LLM 推导行为目标             │    │       ▼                        │
│       │                      │    │  LLM 扮演 persona 逐题回答     │
│       ▼                      │    │  → 1~5 分 × 12 题             │
│  Questionnaire               │    │                                │
│  (12题，1~5分)               │    └───────────────┬────────────────┘
└──────────────┬──────────────┘                    │
               └──────────────┬────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  聚合层                                                          │
│                                                                  │
│  TPB 公式聚合                                                    │
│  → AcceptanceScore (0~1) per persona                            │
│                                                                  │
│  EvaluationMatrix                                               │
│  N personas × (6 input dims + 1 acceptance score)              │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  洞察层                                                          │
│                                                                  │
│  - 相关性分析：哪个维度驱动接受度                                  │
│  - 高分/低分 persona 特征对比                                     │
│  - 系统性死区识别                                                 │
│  - 自然语言洞察报告 (LLM)                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心数据结构

所有结构统一用 Pydantic model，单文件 `models/schemas.py`，不引入 dataclass。

```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

# ── 系统常量 ───────────────────────────────────────────────────────

PREDEFINED_AXIS_NAMES = [
    "痛点意识度",
    "现有方案依赖深度",
    "付费心智",
    "新产品接受度",
    "外部约束强度",
    "问题归因方式",
]

TPB_WEIGHTS = {"attitude": 0.4, "subjective_norm": 0.3, "perceived_control": 0.3}


# ── 用户输入 ───────────────────────────────────────────────────────

class ProductInfo(BaseModel):
    name: str
    description: str
    target_market: str


# ── Stage 1 输出 ───────────────────────────────────────────────────

class DiversityAxis(BaseModel):
    name: str
    description: str = Field(description="针对本产品具体化的维度说明，1句话")
    labels: list[str] = Field(min_length=4, max_length=4,
                               description="从低到高的4个语义标签 [L1, L2, L3, L4]")

class AxesOutput(BaseModel):
    axes: list[DiversityAxis] = Field(min_length=6, max_length=6)


# ── Stage 2 输出 ───────────────────────────────────────────────────

class PersonaVector(BaseModel):
    id: str
    axis_indices: list[int]    # 长度6，每个值 0~3
    axis_labels: list[str]     # 对应语义标签

class PersonaProfile(BaseModel):
    id: str
    vector: PersonaVector
    description: str = Field(description="2~3句自然语言人物描述，含姓名、年龄、职业、关键特征")
    behavior_goal: str = Field(description="该产品对应的核心行为目标")


# ── Stage 3 输出 ───────────────────────────────────────────────────

class QuestionnaireItem(BaseModel):
    id: str
    tpb_dimension: Literal["attitude", "subjective_norm", "perceived_control"]
    question_text: str
    reverse_scored: bool = False

class Questionnaire(BaseModel):
    behavior_goal: str
    items: list[QuestionnaireItem] = Field(min_length=12, max_length=12)

class SingleResponse(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

class QuestionnaireResponse(BaseModel):
    persona_id: str
    responses: dict[str, SingleResponse]   # key = question id


# ── Stage 4 输出 ───────────────────────────────────────────────────

class TPBScore(BaseModel):
    attitude: float
    subjective_norm: float
    perceived_control: float
    acceptance_score: float

class PersonaEvaluation(BaseModel):
    persona: PersonaProfile
    response: QuestionnaireResponse
    tpb_score: TPBScore

class EvaluationMatrix(BaseModel):
    product: ProductInfo
    evaluations: list[PersonaEvaluation]

class InsightReport(BaseModel):
    top_correlations: list[dict]
    ideal_persona_description: str
    dead_zone_description: str
    key_risks: list[str] = Field(min_length=3, max_length=5)
    summary: str
```

---

## 5. 项目结构

```
persona_validator/
│
├── main.py
├── config.py                    # OpenRouter API key、model、采样数量
│
├── models/
│   └── schemas.py               # 所有 Pydantic model（唯一数据结构文件）
│
├── pipeline/
│   ├── axis_builder.py          # Stage 1
│   ├── sampler.py               # Stage 2a（纯计算）
│   ├── persona_generator.py     # Stage 2b
│   ├── questionnaire_builder.py # Stage 3a
│   ├── responder.py             # Stage 3b
│   ├── scorer.py                # Stage 4a（纯计算）
│   └── insight_engine.py        # Stage 4b
│
├── prompts/
│   ├── axis_builder.txt
│   ├── persona_generator.txt
│   ├── questionnaire_builder.txt
│   ├── responder.txt
│   └── insight_engine.txt
│
└── utils/
    ├── llm_client.py
    └── export.py
```

---

## 6. 各模块伪代码

### utils/llm_client.py

```python
from langchain_openai import ChatOpenAI
from config import settings

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
        model="anthropic/claude-3.5-sonnet",
        temperature=0.7,
    )
```

---

### Stage 1: axis_builder.py

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import AxesOutput, ProductInfo, PREDEFINED_AXIS_NAMES
from utils.llm_client import get_llm

PROMPT = """
你是一位消费者行为研究专家。
产品：{product_name} - {product_description}，目标市场：{target_market}

预设的6个多样性维度名称：
{axis_names}

对每个维度：
1. 结合本产品将说明具体化为1句话（description）
2. 生成从低到高的4个语义标签（L1~L4），具体、口语化

{format_instructions}
"""

def build_axes(product: ProductInfo) -> AxesOutput:
    parser = PydanticOutputParser(pydantic_object=AxesOutput)
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser
    return chain.invoke({
        "product_name": product.name,
        "product_description": product.description,
        "target_market": product.target_market,
        "axis_names": "\n".join(f"- {n}" for n in PREDEFINED_AXIS_NAMES),
        "format_instructions": parser.get_format_instructions(),
    })
```

---

### Stage 2a: sampler.py

```python
from scipy.stats.qmc import Sobol
from models.schemas import AxesOutput, PersonaVector

def sample_vectors(axes: AxesOutput, n: int = 16) -> list[PersonaVector]:
    samples = Sobol(d=6, scramble=True).random(n)
    return [
        PersonaVector(
            id=f"persona_{i:03d}",
            axis_indices=[int(v * 4) for v in row],
            axis_labels=[axes.axes[j].labels[int(v * 4)] for j, v in enumerate(row)],
        )
        for i, row in enumerate(samples)
    ]
```

---

### Stage 2b: persona_generator.py

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import PersonaProfile, PersonaVector, AxesOutput, ProductInfo
from utils.llm_client import get_llm

PROMPT = """
你是一位消费者研究员。
产品：{product_description}，目标市场：{target_market}

该用户特征：
{axis_summary}

请完成：
1. 2~3句描述这个真实的人（姓名、年龄、职业、关键行为特征）
2. 推导其对此产品最核心的行为目标

不要提及"维度"或"标签"等系统术语。

{format_instructions}
"""

def generate_persona(vector: PersonaVector, axes: AxesOutput, product: ProductInfo) -> PersonaProfile:
    parser = PydanticOutputParser(pydantic_object=PersonaProfile)
    axis_summary = "\n".join(
        f"- {axes.axes[i].name}：{label}" for i, label in enumerate(vector.axis_labels)
    )
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser
    result = chain.invoke({
        "product_description": product.description,
        "target_market": product.target_market,
        "axis_summary": axis_summary,
        "format_instructions": parser.get_format_instructions(),
    })
    return result.model_copy(update={"id": vector.id, "vector": vector})
```

---

### Stage 3a: questionnaire_builder.py

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import Questionnaire, ProductInfo
from utils.llm_client import get_llm

PROMPT = """
你是一位 TPB 问卷设计专家。
产品：{product_description}，目标行为：{behavior_goal}

设计12题（每个 TPB 维度各4题）：
- attitude：对完成行为结果的评价
- subjective_norm：身边重要的人是否支持
- perceived_control：自己是否有能力完成

要求：第一人称、口语化、不提产品名、每维度至少1道反向计分题。

{format_instructions}
"""

def build_questionnaire(product: ProductInfo, behavior_goal: str) -> Questionnaire:
    parser = PydanticOutputParser(pydantic_object=Questionnaire)
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser
    return chain.invoke({
        "product_description": product.description,
        "behavior_goal": behavior_goal,
        "format_instructions": parser.get_format_instructions(),
    })
```

---

### Stage 3b: responder.py

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import PersonaProfile, Questionnaire, QuestionnaireResponse
from utils.llm_client import get_llm

PROMPT = """
请完全代入以下人物，以第一人称回答问卷，不要跳出角色。

人物：{persona_description}
被问及的行为："{behavior_goal}"

对每道题给出1~5分（1=完全不同意，5=完全同意）并说明理由（1句话）：
{questions}

{format_instructions}
"""

def simulate_responses(persona: PersonaProfile, questionnaire: Questionnaire) -> QuestionnaireResponse:
    parser = PydanticOutputParser(pydantic_object=QuestionnaireResponse)
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser
    result = chain.invoke({
        "persona_description": persona.description,
        "behavior_goal": questionnaire.behavior_goal,
        "questions": "\n".join(f"{item.id}. {item.question_text}" for item in questionnaire.items),
        "format_instructions": parser.get_format_instructions(),
    })
    return result.model_copy(update={"persona_id": persona.id})
```

---

### Stage 4a: scorer.py

```python
from models.schemas import QuestionnaireResponse, Questionnaire, TPBScore, TPB_WEIGHTS

def compute_tpb_score(response: QuestionnaireResponse, questionnaire: Questionnaire) -> TPBScore:
    item_map = {item.id: item for item in questionnaire.items}
    dim_scores: dict[str, list[float]] = {"attitude": [], "subjective_norm": [], "perceived_control": []}

    for item_id, resp in response.responses.items():
        item = item_map[item_id]
        score = (6 - resp.score) if item.reverse_scored else resp.score
        dim_scores[item.tpb_dimension].append((score - 1) / 4.0)

    a = sum(dim_scores["attitude"])          / len(dim_scores["attitude"])
    s = sum(dim_scores["subjective_norm"])   / len(dim_scores["subjective_norm"])
    p = sum(dim_scores["perceived_control"]) / len(dim_scores["perceived_control"])

    return TPBScore(
        attitude=round(a, 3),
        subjective_norm=round(s, 3),
        perceived_control=round(p, 3),
        acceptance_score=round(TPB_WEIGHTS["attitude"]*a + TPB_WEIGHTS["subjective_norm"]*s + TPB_WEIGHTS["perceived_control"]*p, 3),
    )
```

---

### Stage 4b: insight_engine.py

```python
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import EvaluationMatrix, InsightReport
from utils.llm_client import get_llm

PROMPT = """
你是一位创业顾问，正在分析用户模拟数据。
产品：{product_description}

数值分析：
- 各维度与接受度相关系数：{correlations}
- 高接受度用户：{high_profiles}
- 低接受度用户：{low_profiles}
- 分布：最高 {max_score}，最低 {min_score}，均值 {mean_score}

重点识别系统性风险，区分"产品价值未被认可"与"用户无法行动"两类问题。

{format_instructions}
"""

def generate_insights(matrix: EvaluationMatrix) -> InsightReport:
    scores = [e.tpb_score.acceptance_score for e in matrix.evaluations]

    correlations = sorted([
        {
            "axis": matrix.evaluations[0].persona.vector.axis_labels[i],
            "correlation": round(float(np.corrcoef(
                [e.persona.vector.axis_indices[i] for e in matrix.evaluations], scores
            )[0, 1]), 3),
        }
        for i in range(6)
    ], key=lambda x: abs(x["correlation"]), reverse=True)

    median = float(np.median(scores))
    parser = PydanticOutputParser(pydantic_object=InsightReport)
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser
    result = chain.invoke({
        "product_description": matrix.product.description,
        "correlations": correlations,
        "high_profiles":  [e.persona.description for e in matrix.evaluations if e.tpb_score.acceptance_score >= median][:5],
        "low_profiles":   [e.persona.description for e in matrix.evaluations if e.tpb_score.acceptance_score < median][:5],
        "max_score":  round(max(scores), 3),
        "min_score":  round(min(scores), 3),
        "mean_score": round(float(np.mean(scores)), 3),
        "format_instructions": parser.get_format_instructions(),
    })
    return result.model_copy(update={"top_correlations": correlations[:3]})
```

---

### main.py

```python
from models.schemas import ProductInfo, PersonaEvaluation, EvaluationMatrix
from pipeline.axis_builder import build_axes
from pipeline.sampler import sample_vectors
from pipeline.persona_generator import generate_persona
from pipeline.questionnaire_builder import build_questionnaire
from pipeline.responder import simulate_responses
from pipeline.scorer import compute_tpb_score
from pipeline.insight_engine import generate_insights

def run_validation(product: ProductInfo, n_personas: int = 16):
    axes          = build_axes(product)
    vectors       = sample_vectors(axes, n=n_personas)
    profiles      = [generate_persona(v, axes, product) for v in vectors]
    questionnaire = build_questionnaire(product, profiles[0].behavior_goal)

    evaluations = []
    for p in profiles:
        resp = simulate_responses(p, questionnaire)
        evaluations.append(PersonaEvaluation(
            persona=p, response=resp, tpb_score=compute_tpb_score(resp, questionnaire)
        ))

    return generate_insights(EvaluationMatrix(product=product, evaluations=evaluations))


if __name__ == "__main__":
    product = ProductInfo(
        name="钓鱼塘连接App",
        description="一款连接钓鱼爱好者和鱼塘的App，用户可以查看实时鱼情、余位，提前预约鱼塘。",
        target_market="中国城市钓鱼爱好者，25~60岁",
    )
    report = run_validation(product, n_personas=16)
    print(report.summary)
    for risk in report.key_risks:
        print(f"  - {risk}")
```

> **LLM 调用次数（n=16）**：Stage 1×1 + Stage 2b×16 + Stage 3a×1 + Stage 3b×16 + Stage 4b×1 = **35次**。Stage 2b 和 Stage 3b 可用 `asyncio.gather` 并行，实际耗时约等于单次调用的 3 倍。

---

## 7. 关键设计决策说明

**为什么消费者描述维度由产品设计者预定义而非 LLM 自由推导**
LLM 自由推导的维度容易偏向通用化，遗漏创始人最关心的具体假设。预定义维度使系统的评估边界可控、可复现，不同时间跑出的结果可以横向比较。同时对用户而言，维度是系统内部语言，最终暴露的只是洞察结论，不需要用户理解维度本身。

**为什么 behavior_goal 由 LLM 推导而不是手动填写**
不同产品的关键行为不同（有的是下载、有的是首次付费、有的是复购），让 LLM 从产品描述中自动推导，使系统对不同创业场景通用，不需要每次手动配置。

**为什么问卷对所有 persona 共用一套**
问卷测的是"行为意向"，题目只和行为目标相关，和人物特征无关。人物特征通过 LLM 扮演时的代入来影响答案，而不是通过不同题目来区分。

**为什么 TPB 和6个维度完全分离**
如果用同一套框架描述人又评估人，LLM 会自动从描述里推断答案，变成循环自证。分离后，LLM 需要真正从人物状态推导出行为反应。

**为什么用 Sobol 序列而不是纯随机**
纯随机采样在16个点的情况下容易在某些区域扎堆，极端组合（最难转化/最理想用户）有可能采不到。Sobol 序列保证每个维度的每个区间都被均匀覆盖。

---

## 8. 与直接使用 LLM 分析的对比

最直接的替代方案是：把产品描述扔给 LLM，让它直接回答"这个点子靠不靠谱"。这个方案的问题和本系统的优势如下。

**问题一：LLM 给出的是平均用户的视角**

直接问 LLM 时，它倾向于描述"典型用户"，也就是训练数据里最常见的用户画像。钓鱼App的典型用户是40岁男性、有固定鱼塘、不太用新App——LLM会基于这个平均画像给出结论，但这个结论对你的早期种子用户（可能是年轻、痛点强烈、愿意尝鲜的用户）并不适用。

本系统通过均匀采样强制覆盖极端情况，包括最理想的用户和最难转化的用户，结论不依赖"平均值"。

**问题二：结论无法溯源**

LLM 直接给出"付费意愿偏低"的结论，你不知道是因为哪类用户拉低了均值，也不知道有没有一类用户愿意付费。本系统的每个分数都来自具体的 persona，可以追溯到"是哪类人、在哪道题上、给了几分、理由是什么"。

**问题三：无法系统性地压力测试**

LLM 单次回答很难覆盖所有反驳角度。你需要反复追问才能让它说出不利于点子的观点，而且每次追问的覆盖面取决于提问方式，不系统。本系统通过设计反向计分题和刻意生成"最不利"的 persona 组合，强制暴露弱点。

**问题四：不同时间、不同问法结果不稳定**

LLM 对同一个问题的回答受 prompt 措辞影响很大，今天问和明天问可能结论不同。本系统的采样向量是固定的，问卷题目是结构化的，结果可复现，不同版本的产品描述可以横向对比。

**本系统的局限**

相比直接问 LLM，本系统的成本更高（多次 LLM 调用、需要设计维度和问卷），结果仍然是模拟数据而非真实用户数据，不能替代真实用户访谈。它的定位是：在访谈之前，用更低成本、更系统的方式找到"最值得验证的假设"，而不是给出最终答案。

---

## 9. 模拟数据的分析方法

拿到 EvaluationMatrix（N personas × 评估结果）之后，围绕创始人的5个可验证问题，分别用以下方式分析。

### 9.1 需求真实性

**核心问题**：用户真的有这个痛点吗，还是伪命题？

```python
# 用"痛点意识度"和"问题归因方式"两个维度切分用户
# 看这两个维度高的用户，attitude 分数是否也高

high_awareness = [
    e for e in matrix.evaluations
    if e.persona.vector.axis_indices[0] >= 2   # 痛点意识度 L3/L4
    and e.persona.vector.axis_indices[5] >= 2  # 内归因 L3/L4
]

low_awareness = [
    e for e in matrix.evaluations
    if e.persona.vector.axis_indices[0] <= 1
]

avg_attitude_high = mean([e.tpb_score.attitude for e in high_awareness])
avg_attitude_low  = mean([e.tpb_score.attitude for e in low_awareness])

# 判断：
# avg_attitude_high 显著高于 avg_attitude_low → 需求是真实的，有意识的用户认可产品价值
# 两者相差不大 → 警惕伪命题，即使痛苦的用户也不认为产品能解决问题
```

**洞察信号**：如果高意识度用户的 attitude 分也低于 0.5，说明产品对痛点的解法本身有问题，不是需求不存在，而是解决方案没被认可。

---

### 9.2 用户迁移成本

**核心问题**：用户愿意放弃现有方案吗？

```python
# "现有方案依赖"和"外部约束"联合分析 perceived_control 分数
# perceived_control 测的是"我有能力完成这个行为"，迁移成本高的人这项分低

dependency_vs_control = [
    {
        "persona_id": e.persona.id,
        "dependency": e.persona.vector.axis_indices[1],   # 现有方案依赖
        "constraint": e.persona.vector.axis_indices[4],   # 外部约束
        "perceived_control": e.tpb_score.perceived_control
    }
    for e in matrix.evaluations
]

# 找出 perceived_control 最低的 persona 群体
dead_zone = [r for r in dependency_vs_control if r["perceived_control"] < 0.3]

# 判断：
# dead_zone 集中在高依赖 + 高约束区域 → 迁移成本是系统性障碍
# dead_zone 分散 → 迁移成本不是主要问题
```

**洞察信号**：如果 perceived_control 整体偏低（均值 < 0.4），说明产品需要大幅降低切换门槛，比如导入旧数据、提供过渡期免费方案等。

---

### 9.3 外部环境匹配度

**核心问题**：现在的时机对吗？用户所在的市场准备好了吗？

```python
# "新产品接受度"对应市场成熟度和用户心智准备程度
# 用 subjective_norm 分数衡量：身边人的支持度反映市场氛围

acceptance_by_innovativeness = {}
for level in range(4):   # L1~L4
    group = [
        e for e in matrix.evaluations
        if e.persona.vector.axis_indices[3] == level  # 新产品接受度
    ]
    if group:
        acceptance_by_innovativeness[f"L{level+1}"] = {
            "subjective_norm": mean([e.tpb_score.subjective_norm for e in group]),
            "acceptance_score": mean([e.tpb_score.acceptance_score for e in group])
        }

# 判断：
# 只有 L4（早期尝鲜者）接受度高，L1~L3 都低 → 市场还太早，只能做极小众
# L2/L3 接受度也较高 → 市场时机相对成熟，主流用户可触达
```

---

### 9.4 商业模式（付费侧）

**核心问题**：用户愿意付钱吗？什么样的人愿意付？

```python
# "付费心智"维度直接对应商业模式可行性
# 结合 attitude 分数（认可产品价值）一起看

payment_analysis = [
    {
        "pay_willingness": e.persona.vector.axis_indices[2],  # 付费心智 L1~L4
        "attitude": e.tpb_score.attitude,
        "acceptance": e.tpb_score.acceptance_score,
        "persona": e.persona.description
    }
    for e in matrix.evaluations
]

# 找出"认可价值但不愿付费"的群体（高attitude + 低pay_willingness）
value_yes_pay_no = [
    r for r in payment_analysis
    if r["attitude"] > 0.6 and r["pay_willingness"] <= 1
]

# 判断：
# value_yes_pay_no 比例高 → 商业模式需要重新设计
#   可能需要：免费增值、B端收费C端免费、按次收费替代订阅
# value_yes_pay_no 比例低 → 付费逻辑基本成立，关注定价区间
```

**洞察信号**：这组分析能区分"不愿意付钱"和"不认可产品价值"两种不同的商业问题，对应完全不同的解法。

---

### 9.5 竞争护城河（用户侧）

**核心问题**：用户会不会因为有竞品而不选你？

```python
# "现有方案依赖"高的用户 = 已经被竞品或替代方案锁定
# 这类用户的 subjective_norm 分数反映"社交环境是否在推他转向你"

locked_users = [
    e for e in matrix.evaluations
    if e.persona.vector.axis_indices[1] >= 2   # 现有方案依赖 L3/L4
]

# 看这批用户的 subjective_norm
# 高 subjective_norm → 虽然有现有方案，但社交环境在推他用新产品 → 可以撬动
# 低 subjective_norm → 现有方案 + 没有社交压力 → 护城河很深，很难抢

moat_risk = mean([e.tpb_score.subjective_norm for e in locked_users])

# 判断：
# moat_risk > 0.5 → 竞品护城河可被口碑/社群突破
# moat_risk < 0.3 → 需要考虑差异化定位，不能正面竞争
```

---

### 9.6 综合风险矩阵

将5个维度的分析结果汇总成一张风险矩阵，给出每个维度的信号颜色：

```python
def generate_risk_matrix(matrix: EvaluationMatrix) -> dict:
    return {
        "需求真实性":   assess_demand_validity(matrix),    # 红/黄/绿
        "迁移成本":     assess_migration_cost(matrix),
        "外部环境":     assess_market_timing(matrix),
        "商业模式":     assess_business_model(matrix),
        "竞争护城河":   assess_moat(matrix),
    }
    # 每项返回 {"signal": "red|yellow|green", "reason": str, "key_finding": str}
```

这张矩阵是最终给创始人看的核心输出，不需要他理解 TPB 或 persona 向量。

---

## 10. 待定问题（下一步）

- [ ] 问卷题目是否需要针对每个 persona 的 behavior_goal 单独生成，还是统一用第一个
- [ ] TPB 三个维度的权重（0.4 / 0.3 / 0.3）是否需要可配置
- [ ] n_personas 建议值：16 够用还是需要 32
- [ ] 洞察层是否需要支持交互效应检验（两个维度的乘积项）
- [ ] 风险矩阵的红/黄/绿阈值如何标定（需要校准数据或专家判断）

# Campaign Risk Simulator — Python 实现设计文档

## 目录

1. [技术栈](#1-技术栈)
2. [项目结构](#2-项目结构)
3. [核心数据模型](#3-核心数据模型)
4. [API 接口](#4-api-接口)
5. [数据流转逻辑](#5-数据流转逻辑)
6. [Pipeline 实现](#6-pipeline-实现)
7. [Report 页面](#7-report-页面)

---

## 1. 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 原生 async，BackgroundTasks 处理 pipeline |
| 数据库 | MongoDB + motor | 异步驱动，分 collection 存储 |
| LLM 调用 | LangChain Python + OpenRouter | 结构化输出，灵活切换模型 |
| 输出验证 | Pydantic v2 | 替代 Zod，与 FastAPI 原生集成 |
| Sobol 采样 | scipy.stats.qmc.Sobol | 论文原始环境，比 JS 实现更成熟 |
| 图片存储 | 七牛云 SDK (qiniu) | 服务端上传，存储 key 写回 campaign |
| 报告渲染 | Jinja2 + Tailwind CDN | 服务端渲染 HTML，零 JS |
| 配置管理 | pydantic-settings | 环境变量统一管理 |

---

## 2. 项目结构

```
campaign-risk-simulator/
├── main.py                        # FastAPI app 入口
├── config.py                      # 环境变量配置（pydantic-settings）
├── pyproject.toml
│
├── api/
│   └── campaigns.py               # 所有路由
│
├── models/
│   ├── campaign.py                # Campaign Pydantic models
│   ├── dimension.py               # Dimension / Segment models
│   ├── persona.py                 # Persona models
│   └── comment.py                 # Comment / RiskSignals models
│
├── db/
│   ├── client.py                  # motor 连接
│   └── repositories/
│       ├── campaign_repo.py
│       ├── dimension_repo.py
│       ├── persona_repo.py
│       └── comment_repo.py
│
├── pipeline/
│   ├── runner.py                  # pipeline 主流程（BackgroundTask 入口）
│   ├── visual.py                  # 图片 → visual_description
│   ├── dimensions.py              # LLM 提取风险维度
│   ├── sampling.py                # Sobol 采样 → segment 映射
│   ├── personas.py                # 并行生成 Persona
│   └── comments.py                # 并行生成 Comment
│
├── chains/
│   ├── client.py                  # LangChain + OpenRouter 配置
│   ├── image_chain.py             # 多模态图片描述
│   ├── dimension_chain.py         # 风险维度提取
│   ├── persona_chain.py           # Persona 生成
│   └── comment_chain.py           # 评论生成
│
├── storage/
│   └── qiniu.py                   # 七牛云上传封装
│
└── templates/
    ├── report.html                # 报告主模板
    └── partials/
        ├── header.html            # Campaign 标题区
        ├── dimensions.html        # 风险维度 section
        ├── risk_summary.html      # 风险总览 section
        └── comments.html          # 评论列表 section
```

---

## 3. 核心数据模型

### 3.1 Pydantic Models（API 层 & 验证层）

```python
# models/campaign.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class CampaignStatus(str, Enum):
    idle = "idle"
    processing_visual = "processing_visual"
    extracting_dimensions = "extracting_dimensions"
    sampling = "sampling"
    generating_personas = "generating_personas"
    generating_comments = "generating_comments"
    completed = "completed"
    error = "error"

class CampaignCreate(BaseModel):
    title: str
    description: str
    n_personas: int = Field(default=8, ge=4, le=16)

class CampaignStatusResponse(BaseModel):
    campaign_id: str
    status: CampaignStatus
    progress: dict
    stages_completed: list[str]
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
```

```python
# models/dimension.py

from pydantic import BaseModel
from typing import Literal

class Segment(BaseModel):
    id: str                        # "seg_1"
    label: str                     # "传统守护者"
    description: str

class Dimension(BaseModel):
    id: str                        # "dim_1"
    name: str                      # "家庭责任观"
    description: str
    relevance_reason: str
    source: Literal["library", "campaign-specific"]
    segments: list[Segment]        # 3-5 个

# LLM 输出结构
class DimensionsOutput(BaseModel):
    dimensions: list[Dimension]
```

```python
# models/persona.py

from pydantic import BaseModel
from typing import Optional

class Demographics(BaseModel):
    age: int
    gender: str
    city: str
    occupation: str
    income_range: str

class DimensionSegmentRef(BaseModel):
    segment_id: str
    label: str
    description: str

class Persona(BaseModel):
    id: str                        # "persona_1"
    identity_summary: str
    demographics: Demographics
    life_context: str
    values_and_worldview: list[str]
    brand_relationship: str
    psychological_trigger: str
    initial_reaction_hint: str
    # 采样数据（存 DB，不传 LLM）
    dimension_scores: dict[str, float]           # dim_id → 0.0-1.0
    dimension_segments: dict[str, DimensionSegmentRef]  # dim_id → segment
    completed: bool = False
```

```python
# models/comment.py

from pydantic import BaseModel
from typing import Literal

class Reasoning(BaseModel):
    situation_reading: str
    emotional_state: str
    action_choice: str

class CommentContent(BaseModel):
    platform: str                  # "微博" | "小红书" | "抖音评论区"
    text: str
    tone: Literal["正面", "负面", "中立", "复杂"]
    length_type: str               # "短评" | "中评" | "长文"

class RiskSignals(BaseModel):
    spread_likelihood: Literal["低", "中", "高"]
    spread_reason: str
    trigger_keywords: list[str]
    escalation_risk: Literal["低", "中", "高"]

class Comment(BaseModel):
    persona_id: str
    reasoning: Reasoning
    comment: CommentContent
    risk_signals: RiskSignals

class RiskSummary(BaseModel):
    high_escalation_count: int
    high_spread_count: int
    top_trigger_keywords: list[str]
    tone_distribution: dict[str, int]
    riskiest_persona_id: str
```

### 3.2 MongoDB Collections

```
campaigns
  _id                 ObjectId
  title               str
  description         str
  image_key           str | null       # 七牛云 key
  image_url           str | null       # 七牛云访问 URL
  visual_description  str | null       # 多模态 LLM 输出
  status              str              # CampaignStatus 枚举值
  n_personas          int
  error_message       str | null
  created_at          datetime
  started_at          datetime | null
  completed_at        datetime | null

dimensions
  _id                 ObjectId
  campaign_id         str              # campaigns._id
  dim_id              str              # "dim_1"
  name                str
  description         str
  relevance_reason    str
  source              str
  segments            list[dict]       # [{id, label, description}]
  order               int

personas
  _id                 ObjectId
  campaign_id         str
  persona_id          str              # "persona_1"
  dimension_scores    dict             # {dim_id: float}
  dimension_segments  dict             # {dim_id: {segment_id, label, description}}
  identity_summary    str | null
  demographics        dict | null
  life_context        str | null
  values_and_worldview list | null
  brand_relationship  str | null
  psychological_trigger str | null
  initial_reaction_hint str | null
  completed           bool

comments
  _id                 ObjectId
  campaign_id         str
  persona_id          str
  situation_reading   str
  emotional_state     str
  action_choice       str
  platform            str
  text                str
  tone                str
  length_type         str
  spread_likelihood   str
  spread_reason       str
  trigger_keywords    list[str]
  escalation_risk     str
```

**索引**：

```python
# dimensions: campaign_id
# personas:   campaign_id, (campaign_id + persona_id)
# comments:   campaign_id, escalation_risk（报告排序用）
```

---

## 4. API 接口

### 接口总览

```
POST  /campaigns                   # 创建 campaign
POST  /campaigns/{id}/run          # 触发 pipeline（异步）
GET   /campaigns/{id}/status       # 轮询状态
GET   /campaigns/{id}/result       # 获取完整结果（JSON）
GET   /campaigns/{id}/report       # 获取渲染后的 HTML 报告页
```

### `POST /campaigns`

```
Request Body:
{
  "title": "新能源汽车春节活动",
  "description": "某国内新能源汽车品牌...",
  "n_personas": 8
}

Response 201:
{
  "campaign_id": "abc123",
  "status": "idle",
  "created_at": "2026-03-13T10:00:00Z"
}
```

### `POST /campaigns/{id}/run`

```
Request Body: 空

Response 202:
{
  "campaign_id": "abc123",
  "status": "processing_visual",
  "message": "pipeline started"
}

Response 409（已在运行）:
{
  "error": "pipeline_already_running",
  "current_status": "generating_personas"
}
```

幂等设计：已 completed 的 campaign 再次调用 run 返回 409，不重复执行。

### `GET /campaigns/{id}/status`

```
Response 200:
{
  "campaign_id": "abc123",
  "status": "generating_personas",
  "progress": {
    "personas_completed": 5,
    "personas_total": 8,
    "comments_completed": 0,
    "comments_total": 8
  },
  "stages_completed": [
    "processing_visual",
    "extracting_dimensions",
    "sampling"
  ],
  "error": null,
  "started_at": "2026-03-13T10:00:05Z",
  "completed_at": null
}
```

Agent 轮询间隔建议 5-10s，预估总耗时 60-120s。

### `GET /campaigns/{id}/result`

```
Response 200（completed）:
{
  "campaign_id": "abc123",
  "campaign": {
    "title": "...",
    "description": "...",
    "visual_description": "..."
  },
  "dimensions": [ ... ],
  "personas": [ ... ],
  "comments": [ ... ],           # 按 escalation_risk 降序
  "risk_summary": {
    "high_escalation_count": 2,
    "high_spread_count": 3,
    "top_trigger_keywords": ["留守儿童", "femvertising", "不孝"],
    "tone_distribution": {"正面": 1, "负面": 3, "中立": 1, "复杂": 3},
    "riskiest_persona_id": "persona_7"
  },
  "report_url": "https://api.example.com/campaigns/abc123/report"
}

Response 425（未完成）:
{
  "error": "result_not_ready",
  "current_status": "generating_comments",
  "hint": "poll /campaigns/{id}/status until status=completed"
}
```

### `GET /campaigns/{id}/report`

返回渲染好的 HTML 页面（Content-Type: text/html）。

---

## 5. 数据流转逻辑

### 状态机

```
idle
  → processing_visual       有 image_key 时执行，无图片跳过
  → extracting_dimensions   LLM 提取风险维度，写入 dimensions collection
  → sampling                Sobol 采样，预写入 N 条 personas（completed=False）
  → generating_personas     asyncio.gather 并行 N 次 LLM，逐条更新 persona
  → generating_comments     asyncio.gather 并行 N 次 LLM，逐条写入 comments
  → completed
  → error                   任意阶段异常均可跳转，写入 error_message
```

### 核心流程伪代码

```python
# pipeline/runner.py

async def run_pipeline(campaign_id: str):
    try:
        campaign = await campaign_repo.get(campaign_id)

        # Step 1: 图片处理（可选）
        if campaign.image_key:
            await update_status(campaign_id, "processing_visual")
            visual_desc = await process_visual(campaign)
            await campaign_repo.update_visual(campaign_id, visual_desc)

        # Step 2: 提取风险维度
        await update_status(campaign_id, "extracting_dimensions")
        dimensions = await extract_dimensions(campaign)
        await dimension_repo.insert_many(campaign_id, dimensions)

        # Step 3: Sobol 采样
        await update_status(campaign_id, "sampling")
        positions = sample_persona_positions(dimensions, campaign.n_personas)
        # 预写入占位记录
        await persona_repo.insert_placeholders(campaign_id, positions)

        # Step 4: 并行生成 Persona
        await update_status(campaign_id, "generating_personas")
        await asyncio.gather(*[
            generate_and_save_persona(campaign, dimensions, pos)
            for pos in positions
        ])

        # Step 5: 并行生成评论
        await update_status(campaign_id, "generating_comments")
        personas = await persona_repo.get_all(campaign_id)
        await asyncio.gather(*[
            generate_and_save_comment(campaign, persona)
            for persona in personas
        ])

        await update_status(campaign_id, "completed")

    except Exception as e:
        await campaign_repo.set_error(campaign_id, str(e))
```

### Sobol 采样 → Segment 映射

```python
# pipeline/sampling.py

from scipy.stats.qmc import Sobol

def sample_persona_positions(dimensions: list[Dimension], n: int) -> list[dict]:
    sampler = Sobol(d=len(dimensions), scramble=True)
    raw = sampler.random(n)                        # shape (n, n_dimensions)

    positions = []
    for i, row in enumerate(raw):
        scores = {}
        segments = {}
        for j, dim in enumerate(dimensions):
            score = float(row[j])
            seg_index = min(int(score * len(dim.segments)), len(dim.segments) - 1)
            seg = dim.segments[seg_index]
            scores[dim.id] = score
            segments[dim.id] = {
                "segment_id": seg.id,
                "label": seg.label,
                "description": seg.description,
            }
        positions.append({
            "id": f"persona_{i + 1}",
            "scores": scores,
            "segments": segments,
        })
    return positions
```

### 并行 LLM 调用

```python
# pipeline/personas.py

async def generate_and_save_persona(campaign, dimensions, position):
    segments_text = "\n".join([
        f"{dim.name}: {position['segments'][dim.id]['label']} — "
        f"{position['segments'][dim.id]['description']}"
        for dim in dimensions
        if dim.id in position['segments']
    ])

    result = await persona_chain.invoke({
        "campaign_description": campaign.description,
        "segments_text": segments_text,
        "persona_id": position["id"],
    })

    parsed = Persona.model_validate(result)
    await persona_repo.update(campaign.id, position["id"], parsed)
```

### progress 字段计算

status 接口的 progress 从 DB 实时计算，不单独存储：

```python
async def get_progress(campaign_id: str) -> dict:
    personas_total = await persona_repo.count(campaign_id)
    personas_completed = await persona_repo.count_completed(campaign_id)
    comments_total = await comment_repo.count(campaign_id)
    return {
        "personas_completed": personas_completed,
        "personas_total": personas_total,
        "comments_completed": comments_total,
        "comments_total": personas_total,
    }
```

---

## 6. Pipeline 实现

### FastAPI BackgroundTasks

```python
# api/campaigns.py

@router.post("/{id}/run", status_code=202)
async def run_campaign(
    id: str,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
):
    campaign = await campaign_repo.get(id)

    if campaign is None:
        raise HTTPException(404)

    # 幂等检查
    if campaign.status not in ("idle", "error"):
        raise HTTPException(409, detail={
            "error": "pipeline_already_running",
            "current_status": campaign.status,
        })

    await campaign_repo.update_status(id, "processing_visual")
    background_tasks.add_task(run_pipeline, id)

    return {"campaign_id": id, "status": "processing_visual", "message": "pipeline started"}
```

### 错误处理

pipeline 任意步骤抛出异常，统一 catch 后写入 `error_message`，status 置为 `error`。

Agent 通过 status 接口检测到 `error` 后，可读取 `error` 字段，决定是否重新触发 run。

---

## 7. Report 页面

### 布局结构

```
GET /campaigns/{id}/report → text/html

┌─────────────────────────────────────┐
│  Campaign Header                    │  标题、描述、visual_description（折叠）
├─────────────────────────────────────┤
│  Section 1: 风险维度                 │  横向卡片，每卡一个维度 + segments 列表
├─────────────────────────────────────┤
│  Section 2: 风险总览                 │  4格：高升级数、高传播数、情绪分布、触发词
├─────────────────────────────────────┤
│  Section 3: 模拟评论区               │  评论列表，默认按 escalation_risk 降序
└─────────────────────────────────────┘
```

### 评论卡片结构

```
┌─────────────────────────────────────────────────────┐
│ [风险标签] 🚨升级:高  🔥传播:高                       │
│                                                     │
│ persona_7 · 女 22岁 · 西安大学生                     │
│ 小红书 · 长文 · 情绪:复杂                             │
│                                                     │
│ "说说我为什么看这个广告不舒服..."                      │
│                                                     │
│ 触发词: [留守儿童] [不孝]                             │
│ 传播原因: 情感共鸣强，易引发集体发声                   │
│                                                     │
│ <details>▼ 查看 Persona 详情</details>               │  ← 原生折叠，零 JS
│   心理触发: ...                                      │
│   价值观: ...                                        │
│   维度定位: 传统守护者 / 审慎支持者 / 全球化融合者      │
└─────────────────────────────────────────────────────┘
```

### 色彩规范

| 场景 | Tailwind class |
|------|---------------|
| 高风险 | `text-red-600 bg-red-50` |
| 中风险 | `text-amber-600 bg-amber-50` |
| 低风险 | `text-green-600 bg-green-50` |
| 负面情绪 | `text-red-500` |
| 复杂情绪 | `text-purple-500` |
| 正面情绪 | `text-green-500` |
| 中立情绪 | `text-gray-500` |
| library 维度 | `bg-blue-50 border-blue-200` |
| campaign-specific 维度 | `bg-orange-50 border-orange-200` |

### Jinja2 渲染

```python
# api/campaigns.py

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

@router.get("/{id}/report", response_class=HTMLResponse)
async def get_report(request: Request, id: str):
    campaign = await campaign_repo.get(id)
    if campaign.status != "completed":
        raise HTTPException(425, detail="result_not_ready")

    dimensions = await dimension_repo.get_by_campaign(id)
    personas = await persona_repo.get_by_campaign(id)
    # 评论按 escalation_risk 降序
    comments = await comment_repo.get_by_campaign(id, sort_by="escalation_risk")
    risk_summary = compute_risk_summary(comments)

    return templates.TemplateResponse("report.html", {
        "request": request,
        "campaign": campaign,
        "dimensions": dimensions,
        "personas": {p.persona_id: p for p in personas},
        "comments": comments,
        "risk_summary": risk_summary,
    })
```

---

_文档版本：v1.0 | 2026-03-13_

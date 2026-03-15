import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from db.client import init_indexes, close_client
from api.studies import router as studies_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_indexes()
    yield
    await close_client()


app = FastAPI(
    title="Consumer Research Framework",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(studies_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/agent-guide", response_class=PlainTextResponse)
async def agent_guide(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return f"""# Consumer Research Framework — Agent Guide

## 这个服务是做什么的

Consumer Research Framework 是一个消费者行为研究仿真服务。
给定任意商业决策场景（营销活动、新品概念、定价方案、广告创意等），
服务会自动生成多个合成消费者 Persona，模拟他们对该场景的真实反应，并输出结构化研究报告。

适用场景举例：
- 评估营销活动的舆论风险
- 测试新产品概念的市场接受度
- 验证定价方案是否合理
- 测试广告创意是否能引发共鸣

## 完整接口文档

{base_url}/openapi.json

读取上方 JSON 可获得所有接口的完整参数定义和响应结构。

## 认证

所有 /studies 接口需要 Bearer Token：
  Authorization: Bearer <api_key>

如服务未配置 API_KEYS，则无需认证。

## 关键业务概念（openapi.json 里读不到的语义）

### study_design 的四个枚举如何组合

study_type      — 研究场景类型，影响维度提取方向
research_objective — 自由文本，直接告诉 LLM "你想了解什么"，越具体越好
response_mode   — 消费者以什么方式反应（评论/问卷/访谈/购买决策/即时情绪）
analysis_framework — 结果用什么框架分析（风险/接受度/细分/决策路径/匹配度）

常用组合：
  营销风险评估：risk_assessment + comment + risk
  新品概念测试：concept_test + survey + acceptance
  定价测试：    pricing_test + purchase_intent + acceptance
  创意测试：    creative_test + reaction + fit_assessment
  公关声明评估：policy_test + comment + risk
  用户旅程研究：user_journey + interview + decision_path

### Pipeline 状态流转

创建任务后异步执行，通过轮询 status 接口跟踪进度：

  extracting_dimensions → sampling → generating_personas → generating_responses → completed
  processing_visual（有图片时为第一步）
  error（任务失败）

建议轮询间隔：8-10 秒

### 结果中最重要的字段

summary.overall_conclusion  — 一句话核心结论
summary.findings            — 核心发现列表，每条含 key/value/evidence/importance
summary.suggested_actions   — 可执行的建议动作
response_summary.stance_distribution — 消费者立场分布（正面/负面/中立/复杂）
report_url                  — 可分享的 HTML 报告页面

## 写 Skill 的最简流程

1. POST /studies 创建任务，传入 title + description + study_design（JSON字符串）
2. 轮询 GET /studies/{{study_id}}/status 直到 status == "completed"
3. GET /studies/{{study_id}}/result 获取完整结果
4. 将 report_url 和 summary.overall_conclusion 返回给用户
"""

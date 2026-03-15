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


_AGENT_GUIDE_STATIC = """\
## 这个服务是做什么的

Consumer Research Framework 是一个消费者行为研究仿真服务。
给定任意商业决策场景（营销活动、新品概念、定价方案、广告创意等），
服务会自动生成多个合成消费者 Persona，模拟他们对该场景的真实反应，并输出结构化研究报告。

适用场景举例：
- 评估营销活动的舆论风险
- 测试新产品概念的市场接受度
- 验证定价方案是否合理
- 测试广告创意是否能引发共鸣

## 认证

所有 /studies 接口需要 Bearer Token：
  Authorization: Bearer <api_key>

如服务未配置 API_KEYS，则无需认证。

## 关键业务概念（openapi.json 里读不到的语义）

### study_design 的四个枚举如何组合

study_type         -- 研究场景类型，影响维度提取方向
research_objective -- 自由文本，直接告诉 LLM "你想了解什么"，越具体越好
response_mode      -- 消费者以什么方式反应（评论/问卷/访谈/购买决策/即时情绪）
analysis_framework -- 结果用什么框架分析（风险/接受度/细分/决策路径/匹配度）

常用组合：
  营销风险评估：risk_assessment + comment + risk
  新品概念测试：concept_test + survey + acceptance
  定价测试：    pricing_test + purchase_intent + acceptance
  创意测试：    creative_test + reaction + fit_assessment
  公关声明评估：policy_test + comment + risk
  用户旅程研究：user_journey + interview + decision_path

### Pipeline 状态流转

创建任务后异步执行，通过轮询 status 接口跟踪进度：

  extracting_dimensions -> sampling -> generating_personas -> generating_responses -> completed
  processing_visual（有图片时为第一步）
  error（任务失败）

建议轮询间隔：8-10 秒

### 结果中最重要的字段

summary.overall_conclusion           -- 一句话核心结论
summary.findings                     -- 核心发现列表，每条含 key/value/evidence/importance
summary.suggested_actions            -- 可执行的建议动作
response_summary.stance_distribution -- 消费者立场分布（正面/负面/中立/复杂）
report_url                           -- 可分享的 HTML 报告页面

## 写 Skill 的最简流程

1. POST /studies 创建任务，传入 title + description + study_design（JSON字符串）
2. 轮询 GET /studies/{study_id}/status 直到 status == "completed"
3. GET /studies/{study_id}/result 获取完整结果
4. 将 report_url 和 summary.overall_conclusion 返回给用户

## Skill 目录结构示例

    ~/.nanobot/workspace/skills/consumer-research/   # skill 根目录，目录名即 skill 名
    |-- SKILL.md       # skill 元数据（name/description/触发词）+ 使用说明，必须存在
    +-- scripts/
        |-- run.py     # 主脚本：创建任务 -> 轮询 -> 返回结果（供 agent 调用）
        |-- status.py  # 辅助脚本：只查询某个 study_id 的状态
        +-- result.py  # 辅助脚本：只拉取某个 study_id 的完整结果

SKILL.md 头部格式（YAML frontmatter）：

    ---
    name: consumer-research
    description: 触发词写在这里，agent 用它判断何时调用本 skill
    ---
    # 正文：使用方法、参数说明、返回字段说明

## 核心脚本示例（带注释）

    #!/usr/bin/env python3
    # run.py -- 创建研究任务并阻塞等待完成，最终输出结果 JSON。
    # agent 直接调用此脚本，无需理解底层 HTTP 细节。
    import argparse, json, time, sys
    import httpx  # pip install httpx

    BASE_URL = "http://localhost:6791"  # 服务地址，可通过 --base_url 覆盖
    POLL_INTERVAL = 8                   # 轮询间隔（秒），官方建议 8-10 秒

    def main():
        parser = argparse.ArgumentParser()
        # 任务基本信息
        parser.add_argument("--title",       required=True)
        parser.add_argument("--description", required=True)
        # study_design 四要素（参考上方常用组合表）
        parser.add_argument("--study_type",          default="risk_assessment")
        parser.add_argument("--research_objective",  default="识别潜在风险")
        parser.add_argument("--response_mode",       default="comment")
        parser.add_argument("--analysis_framework",  default="risk")
        # 可选参数
        parser.add_argument("--n_personas",    type=int, default=8)
        parser.add_argument("--target_market", default="中国大陆")
        parser.add_argument("--api_key",       default="")
        parser.add_argument("--base_url",      default=BASE_URL)
        args = parser.parse_args()

        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"

        # Step 1: 创建研究任务
        # study_design 以 JSON 字符串形式传入（不是嵌套对象）
        payload = {
            "title": args.title,
            "description": args.description,
            "study_design": json.dumps({
                "study_type": args.study_type,
                "research_objective": args.research_objective,
                "response_mode": args.response_mode,
                "analysis_framework": args.analysis_framework,
            }),
            "n_personas": args.n_personas,
            "target_market": args.target_market,
        }
        resp = httpx.post(f"{args.base_url}/studies", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        study_id = resp.json()["study_id"]  # 后续轮询用

        # Step 2: 轮询状态直到完成
        while True:
            st = httpx.get(f"{args.base_url}/studies/{study_id}/status",
                           headers=headers, timeout=10).json()
            status = st["status"]
            print(f"[status] {status}", file=sys.stderr)  # 进度到 stderr，不污染 stdout
            if status == "completed":
                break
            if status == "error":
                print(json.dumps({"error": st.get("error_message")}))
                sys.exit(1)
            time.sleep(POLL_INTERVAL)

        # Step 3: 获取完整结果，输出到 stdout 供 agent 读取
        # 重点字段：overall_conclusion / findings / report_url
        result = httpx.get(f"{args.base_url}/studies/{study_id}/result",
                           headers=headers, timeout=30).json()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if __name__ == "__main__":
        main()
"""


@app.get("/agent-guide", response_class=PlainTextResponse)
async def agent_guide(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return (
        "# Consumer Research Framework -- Agent Guide\n\n"
        + _AGENT_GUIDE_STATIC.replace(
            "## 认证",
            f"## 完整接口文档\n\n{base_url}/openapi.json\n\n"
            "读取上方 JSON 可获得所有接口的完整参数定义和响应结构。\n\n"
            "## 认证",
            1,
        )
    )

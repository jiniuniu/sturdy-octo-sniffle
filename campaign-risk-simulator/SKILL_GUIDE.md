# Consumer Research Framework — Nanobot Skill 开发指南

本文档指导 agent 为 Consumer Research Framework 服务编写 nanobot skill。

---

## Skill 结构

每个 skill 放在 `~/.nanobot/workspace/skills/<skill-name>/` 目录下：

```
~/.nanobot/workspace/skills/consumer-research/
  SKILL.md          ← skill 描述与使用说明（agent 读取此文件决定是否调用）
  scripts/
    run.py          ← 创建并等待研究任务完成
    status.py       ← 查询任务状态
    result.py       ← 获取完整结果
```

---

## 服务接口说明

**Base URL**：`http://localhost:6791`（云端替换为实际地址）

### 1. 创建研究任务

```
POST /studies
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | string | ✓ | 研究标题 |
| `description` | string | ✓ | 研究对象详细描述 |
| `study_design` | string (JSON) | ✓ | 研究设计，见下方说明 |
| `stimulus_type` | string | 否 | 刺激物类型，默认 `campaign` |
| `context` | string | 否 | 背景信息 |
| `n_personas` | int | 否 | persona 数量，4~16，默认 8 |
| `target_market` | string | 否 | 目标市场，默认 `中国大陆` |
| `callback_url` | string | 否 | 完成后回调地址 |
| `image` | file | 否 | 相关图片（jpg/png） |

**study_design JSON 结构**：
```json
{
  "study_type": "risk_assessment",
  "research_objective": "识别该活动可能引发的舆论风险",
  "response_mode": "comment",
  "analysis_framework": "risk"
}
```

**响应（202）**：
```json
{
  "study_id": "abc123",
  "status": "extracting_dimensions",
  "created_at": "2026-03-15T10:00:00Z"
}
```

### 2. 查询状态

```
GET /studies/{study_id}/status
```

**响应**：
```json
{
  "study_id": "abc123",
  "status": "generating_responses",
  "progress": {
    "personas_completed": 6,
    "personas_total": 8,
    "responses_completed": 4,
    "responses_total": 8
  },
  "stages_completed": ["extracting_dimensions", "sampling", "generating_personas"],
  "report_url": null
}
```

`status` 流转：
```
extracting_dimensions → sampling → generating_personas → generating_responses → completed
processing_visual（有图片时为第一步）
error
```

`completed` 时 `report_url` 有值，指向 HTML 报告页面。

### 3. 获取结果

```
GET /studies/{study_id}/result
```

仅 `status=completed` 时可调用，否则返回 `425`。

**响应**：
```json
{
  "study_id": "abc123",
  "study": {
    "title": "...",
    "description": "...",
    "study_design": { "study_type": "risk_assessment", "..." : "..." }
  },
  "summary": {
    "overall_conclusion": "该活动存在中高文化挪用风险...",
    "confidence_level": "高",
    "findings": [
      { "key": "整体风险等级", "value": "高", "evidence": "...", "importance": "高" }
    ],
    "segment_differences": [...],
    "suggested_actions": [...],
    "open_questions": [...]
  },
  "dimensions": [...],
  "personas": [...],
  "responses": [...],
  "response_summary": {
    "stance_distribution": { "负面": 4, "中立": 3, "正面": 1 },
    "signal_summary": { "escalation_risk": 3, "spread_risk": 2 },
    "top_quotes": [...]
  },
  "report_url": "http://localhost:6791/studies/abc123/report"
}
```

### 4. Webhook 回调

创建时传 `callback_url`，pipeline 完成/失败后服务会主动 POST：

```json
// 成功
{"study_id": "abc123", "status": "completed"}

// 失败
{"study_id": "abc123", "status": "error", "error": "..."}
```

---

## SKILL.md 模板

```markdown
---
name: consumer-research
description: 对任意商业决策场景进行消费者反应仿真研究，生成多角色 Persona 反应与研究报告。支持风险评估、概念测试、定价测试、创意测试等场景。触发词：消费者研究、风险分析、概念测试、定价测试、创意测试、用户反应。
---

# Consumer Research Framework

对任意商业内容进行 AI 多 Persona 消费者仿真，输出结构化研究报告。

## 使用方法

### 创建并等待研究完成

```bash
python3 ~/.nanobot/workspace/skills/consumer-research/scripts/run.py \
  --title "研究标题" \
  --description "研究对象详细描述" \
  --study_type "risk_assessment" \
  --research_objective "识别该活动可能引发的舆论风险" \
  --response_mode "comment" \
  --analysis_framework "risk" \
  [--n_personas 8] \
  [--target_market "中国大陆"]
```

### 查询状态

```bash
python3 ~/.nanobot/workspace/skills/consumer-research/scripts/status.py <study_id>
```

### 获取结果

```bash
python3 ~/.nanobot/workspace/skills/consumer-research/scripts/result.py <study_id>
```

## 常用 study_type / response_mode / analysis_framework 组合

| 场景 | study_type | response_mode | analysis_framework |
|------|-----------|--------------|-------------------|
| 营销活动风险评估 | risk_assessment | comment | risk |
| 新品概念测试 | concept_test | survey | acceptance |
| 定价测试 | pricing_test | purchase_intent | acceptance |
| 广告创意测试 | creative_test | reaction | fit_assessment |
| 公关声明评估 | policy_test | comment | risk |
| 用户旅程研究 | user_journey | interview | decision_path |

## 返回说明

- `run.py`：阻塞等待直到完成，输出完整结果 JSON
- `status.py`：输出当前状态与进度
- `result.py`：输出完整结果 JSON，包含 `report_url`

## 注意事项

1. 研究通常需要 2~5 分钟，取决于 persona 数量
2. `report_url` 是可分享的 HTML 报告页面链接
3. `summary.overall_conclusion` 为一句话核心结论
```
```

---

## scripts/run.py 模板

```python
#!/usr/bin/env python3
"""
创建研究任务并等待完成，输出最终结果。
Usage: python3 run.py --title "标题" --description "描述" --study_type "risk_assessment" ...
"""
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse

BASE_URL = "http://localhost:6791"


def create_study(title, description, study_type, research_objective, response_mode, analysis_framework, n_personas, target_market):
    study_design = json.dumps({
        "study_type": study_type,
        "research_objective": research_objective,
        "response_mode": response_mode,
        "analysis_framework": analysis_framework,
    })
    body = urllib.parse.urlencode({
        "title": title,
        "description": description,
        "study_design": study_design,
        "n_personas": str(n_personas),
        "target_market": target_market,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/studies",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def poll_status(study_id, interval=10, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        req = urllib.request.Request(f"{BASE_URL}/studies/{study_id}/status")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        status = data["status"]
        progress = data.get("progress", {})
        print(
            f"[{status}] personas {progress.get('personas_completed', 0)}/{progress.get('personas_total', 0)}"
            f" responses {progress.get('responses_completed', 0)}/{progress.get('responses_total', 0)}",
            file=sys.stderr,
        )
        if status == "completed":
            return True
        if status == "error":
            print(f"Pipeline 失败: {data.get('error')}", file=sys.stderr)
            return False
        time.sleep(interval)
    print("超时", file=sys.stderr)
    return False


def get_result(study_id):
    req = urllib.request.Request(f"{BASE_URL}/studies/{study_id}/result")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--study_type", default="risk_assessment")
    parser.add_argument("--research_objective", default="分析消费者对该内容的反应")
    parser.add_argument("--response_mode", default="comment")
    parser.add_argument("--analysis_framework", default="risk")
    parser.add_argument("--n_personas", type=int, default=8)
    parser.add_argument("--target_market", default="中国大陆")
    args = parser.parse_args()

    created = create_study(
        args.title, args.description,
        args.study_type, args.research_objective,
        args.response_mode, args.analysis_framework,
        args.n_personas, args.target_market,
    )
    study_id = created["study_id"]
    print(f"study_id: {study_id}", file=sys.stderr)

    if poll_status(study_id):
        result = get_result(study_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        sys.exit(1)
```

---

## 目录创建命令

agent 执行以下命令完成 skill 安装：

```bash
mkdir -p ~/.nanobot/workspace/skills/consumer-research/scripts
# 然后写入 SKILL.md 和 scripts/run.py、status.py、result.py
```

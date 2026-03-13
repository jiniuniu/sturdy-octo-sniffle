# Campaign Risk Simulator — Nanobot Skill 开发指南

本文档指导 agent 为 Campaign Risk Simulator 服务编写 nanobot skill。

---

## Skill 结构

每个 skill 放在 `~/.nanobot/workspace/skills/<skill-name>/` 目录下：

```
~/.nanobot/workspace/skills/campaign-risk/
  SKILL.md          ← skill 描述与使用说明（agent 读取此文件决定是否调用）
  scripts/
    run.py          ← 创建并触发分析任务
    status.py       ← 查询任务状态
    result.py       ← 获取完整结果
```

---

## 服务接口说明

**Base URL**：`http://localhost:6791`（云端替换为实际地址）

### 1. 创建任务

```
POST /campaigns
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | string | ✓ | 活动名称 |
| `description` | string | ✓ | 活动详细描述 |
| `n_personas` | int | 否 | persona 数量，4~16，默认 8 |
| `target_market` | string | 否 | 目标市场，默认"中国大陆" |
| `callback_url` | string | 否 | 完成后回调地址 |
| `image` | file | 否 | 活动图片（jpg/png） |

**响应（202）**：
```json
{
  "campaign_id": "abc123",
  "status": "extracting_dimensions",
  "created_at": "2026-03-13T10:00:00Z"
}
```

### 2. 查询状态

```
GET /campaigns/{campaign_id}/status
```

**响应**：
```json
{
  "campaign_id": "abc123",
  "status": "generating_comments",
  "progress": {
    "personas_completed": 6,
    "personas_total": 8,
    "comments_completed": 4,
    "comments_total": 8
  },
  "stages_completed": ["extracting_dimensions", "sampling", "generating_personas"],
  "report_url": null
}
```

`status` 流转：
```
extracting_dimensions → sampling → generating_personas → generating_comments → completed
processing_visual（有图片时为第一步）
error（可重新创建任务）
```

`completed` 时 `report_url` 有值，指向 HTML 报告页面。

### 3. 获取结果

```
GET /campaigns/{campaign_id}/result
```

仅 `status=completed` 时可调用，否则返回 `425`。

**响应**：
```json
{
  "campaign_id": "abc123",
  "campaign": { "title": "...", "description": "..." },
  "summary": {
    "overall_risk_level": "高",
    "key_risk_summary": "...",
    "top_risks": [...],
    "suggested_actions": [...]
  },
  "dimensions": [...],
  "personas": [...],
  "comments": [...],
  "risk_summary": {
    "high_escalation_count": 3,
    "high_spread_count": 2,
    "top_trigger_keywords": ["价格", "质量"],
    "tone_distribution": {"负面": 4, "中性": 3, "正面": 1},
    "riskiest_persona_id": "persona_3"
  },
  "report_url": "http://localhost:6791/campaigns/abc123/report"
}
```

### 4. Webhook 回调

创建时传 `callback_url`，pipeline 完成/失败后服务会主动 POST：

```json
// 成功
{"campaign_id": "abc123", "status": "completed"}

// 失败
{"campaign_id": "abc123", "status": "error", "error": "..."}
```

---

## SKILL.md 模板

```markdown
---
name: campaign-risk
description: 对品牌营销活动进行风险模拟分析，生成多角色消费者反应与风险评估报告。当用户需要评估活动风险、分析消费者反应、检验营销方案时使用。触发词：风险分析、活动评估、消费者反应、campaign risk。
---

# Campaign Risk Simulator

对营销活动进行 AI 多 Persona 风险仿真，输出结构化风险报告。

## 使用方法

### 创建并等待分析完成

```bash
python3 ~/.nanobot/workspace/skills/campaign-risk/scripts/run.py \
  --title "活动名称" \
  --description "活动详细描述" \
  [--n_personas 8] \
  [--target_market "中国大陆"]
```

### 查询状态

```bash
python3 ~/.nanobot/workspace/skills/campaign-risk/scripts/status.py <campaign_id>
```

### 获取结果

```bash
python3 ~/.nanobot/workspace/skills/campaign-risk/scripts/result.py <campaign_id>
```

## 返回说明

- `run.py`：阻塞等待直到完成，输出完整结果 JSON
- `status.py`：输出当前状态与进度
- `result.py`：输出完整结果 JSON，包含 `report_url`

## 注意事项

1. 分析通常需要 2~5 分钟，取决于 persona 数量
2. `report_url` 是可分享的 HTML 报告页面链接
3. `summary.overall_risk_level` 为最终风险等级：低 / 中 / 高
```
```

---

## scripts/run.py 模板

```python
#!/usr/bin/env python3
"""
创建 campaign 并等待完成，输出最终结果。
Usage: python3 run.py --title "活动名" --description "描述" [--n_personas 8] [--target_market "中国大陆"]
"""
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse

BASE_URL = "http://localhost:6791"


def create_campaign(title, description, n_personas, target_market):
    body = urllib.parse.urlencode({
        "title": title,
        "description": description,
        "n_personas": str(n_personas),
        "target_market": target_market,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/campaigns",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def poll_status(campaign_id, interval=10, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        req = urllib.request.Request(f"{BASE_URL}/campaigns/{campaign_id}/status")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        status = data["status"]
        progress = data.get("progress", {})
        print(
            f"[{status}] personas {progress.get('personas_completed', 0)}/{progress.get('personas_total', 0)}"
            f" comments {progress.get('comments_completed', 0)}/{progress.get('comments_total', 0)}",
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


def get_result(campaign_id):
    req = urllib.request.Request(f"{BASE_URL}/campaigns/{campaign_id}/result")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--n_personas", type=int, default=8)
    parser.add_argument("--target_market", default="中国大陆")
    args = parser.parse_args()

    created = create_campaign(args.title, args.description, args.n_personas, args.target_market)
    campaign_id = created["campaign_id"]
    print(f"campaign_id: {campaign_id}", file=sys.stderr)

    if poll_status(campaign_id):
        result = get_result(campaign_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        sys.exit(1)
```

---

## 目录创建命令

agent 执行以下命令完成 skill 安装：

```bash
mkdir -p ~/.nanobot/workspace/skills/campaign-risk/scripts
# 然后写入 SKILL.md 和 scripts/run.py、status.py、result.py
```

"""
本地服务端到端测试脚本
用法:
  python test_service.py                        # 默认运行 risk_assessment 场景
  python test_service.py --scenario concept     # 运行概念测试场景
  python test_service.py --scenario pricing     # 运行定价测试场景
  python test_service.py --n_personas 4         # 减少 persona 数量加快测试
  python test_service.py --study_id abc123      # 跳过创建，直接查询已有任务
"""
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
import urllib.parse

BASE_URL = "http://localhost:6791"

# ── 预设测试场景 ──────────────────────────────────────────────

SCENARIOS = {
    "risk": {
        "title": "新能源汽车春节返乡活动",
        "description": (
            "某国内新能源汽车品牌发起春节营销活动，主题为「回家，从不将就」。"
            "活动内容：推出限时优惠，并发布一支广告片，展示一位在大城市打拼的年轻人"
            "拒绝乘坐父母安排的传统燃油车，坚持开自己的新能源车回家，"
            "广告中父母神情失落，最终在孩子的感染下认可了新能源车。"
        ),
        "study_design": {
            "study_type": "risk_assessment",
            "research_objective": "识别该广告可能引发的舆论风险和文化敏感点",
            "response_mode": "comment",
            "analysis_framework": "risk",
        },
    },
    "concept": {
        "title": "AI 健康管理订阅服务概念测试",
        "description": (
            "我们计划推出一款 AI 健康管理应用，每月通过问卷和可穿戴设备数据"
            "生成个性化健康建议，并配备专属 AI 健康顾问随时解答问题。"
            "目标用户为 25-45 岁、关注健康的城市白领。"
        ),
        "study_design": {
            "study_type": "concept_test",
            "research_objective": "了解目标用户对该产品概念的接受程度和核心顾虑",
            "response_mode": "survey",
            "analysis_framework": "acceptance",
        },
    },
    "pricing": {
        "title": "咖啡会员卡定价测试",
        "description": (
            "某连锁咖啡品牌计划推出月度会员卡，每月 38 元，可享受每天一杯指定饮品（门店价 28-35 元）、"
            "生日当月免费升杯、每周一次买一赠一。会员卡不可转让，未使用权益不累积。"
        ),
        "study_design": {
            "study_type": "pricing_test",
            "research_objective": "评估该会员卡定价和权益设计对目标用户的吸引力",
            "response_mode": "purchase_intent",
            "analysis_framework": "acceptance",
        },
    },
}

# ── HTTP 工具 ─────────────────────────────────────────────────

def _request(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    body = None
    headers = {}
    if data:
        body = urllib.parse.urlencode(data).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body}")
        raise


def check_health():
    try:
        resp = _request("GET", "/health")
        return resp.get("status") == "ok"
    except Exception:
        return False


def create_study(scenario: dict, n_personas: int) -> str:
    data = {
        "title": scenario["title"],
        "description": scenario["description"],
        "study_design": json.dumps(scenario["study_design"], ensure_ascii=False),
        "n_personas": str(n_personas),
        "target_market": "中国大陆",
    }
    resp = _request("POST", "/studies", data)
    return resp["study_id"]


def poll_status(study_id: str, interval: int = 8, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    last_status = ""
    while time.time() < deadline:
        resp = _request("GET", f"/studies/{study_id}/status")
        status = resp["status"]
        progress = resp.get("progress", {})

        if status != last_status:
            print(f"\n  ▶ {status}", end="", flush=True)
            last_status = status
        else:
            p = progress.get("personas_completed", 0)
            pt = progress.get("personas_total", 0)
            r = progress.get("responses_completed", 0)
            rt = progress.get("responses_total", 0)
            print(f"\r  ▶ {status}  personas {p}/{pt}  responses {r}/{rt}   ", end="", flush=True)

        if status == "completed":
            print()
            return True
        if status == "error":
            print(f"\n  ✗ 失败: {resp.get('error')}")
            return False

        time.sleep(interval)

    print("\n  ✗ 超时")
    return False


def print_result(result: dict):
    study = result.get("study", {})
    design = study.get("study_design", {})
    summary = result.get("summary", {})
    resp_summary = result.get("response_summary", {})

    print("\n" + "=" * 60)
    print(f"  {study.get('title', '')}")
    print(f"  study_type: {design.get('study_type')}  |  framework: {design.get('analysis_framework')}")
    print("=" * 60)

    # 总结
    if summary:
        print(f"\n【总结】")
        print(f"  结论: {summary.get('overall_conclusion', '')}")
        print(f"  可信度: {summary.get('confidence_level', '')}")
        print()
        for f in summary.get("findings", []):
            importance_mark = "🔴" if f["importance"] == "高" else ("🟡" if f["importance"] == "中" else "⚪")
            print(f"  {importance_mark} {f['key']}: {f['value']}")
            print(f"     → {f['evidence'][:80]}{'...' if len(f['evidence']) > 80 else ''}")
        print()
        print("  建议动作:")
        for i, action in enumerate(summary.get("suggested_actions", []), 1):
            print(f"    {i}. {action}")
        if summary.get("open_questions"):
            print("\n  待深入研究:")
            for q in summary.get("open_questions", []):
                print(f"    · {q}")

    # 反应分布
    if resp_summary:
        print(f"\n【反应分布】")
        dist = resp_summary.get("stance_distribution", {})
        total = sum(dist.values()) or 1
        for stance, count in sorted(dist.items(), key=lambda x: -x[1]):
            bar = "█" * count + "░" * (max(0, 8 - count))
            print(f"  {stance:4s}  {bar}  {count}/{total}")

        signals = resp_summary.get("signal_summary", {})
        if signals:
            print(f"\n  高风险信号: " + "  ".join(f"{k}×{v}" for k, v in signals.items()))

        quotes = resp_summary.get("top_quotes", [])
        if quotes:
            print(f"\n  代表性观点:")
            for q in quotes[:3]:
                print(f"  [{q['stance']}] {q['quote'][:60]}{'...' if len(q['quote']) > 60 else ''}")

    # 人群差异
    seg_diffs = (summary or {}).get("segment_differences", [])
    if seg_diffs:
        print(f"\n【人群差异】")
        for s in seg_diffs:
            print(f"  · {s['segment_description']}（{s.get('size_estimate', '')}）")
            print(f"    {s['stance']}")

    print(f"\n  报告地址: {result.get('report_url', '')}")
    print("=" * 60)


# ── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consumer Research Framework 本地测试")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="risk",
                        help="测试场景 (默认: risk)")
    parser.add_argument("--n_personas", type=int, default=4,
                        help="Persona 数量，4-16 (默认: 4，加快测试)")
    parser.add_argument("--study_id", default="",
                        help="跳过创建，直接查询已有 study_id")
    parser.add_argument("--base_url", default=BASE_URL,
                        help=f"服务地址 (默认: {BASE_URL})")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    print(f"Consumer Research Framework 本地测试")
    print(f"服务地址: {BASE_URL}")
    print("-" * 40)

    # 健康检查
    print("健康检查...", end=" ")
    if not check_health():
        print("✗ 服务未启动，请先运行: uvicorn main:app --reload --port 6791")
        sys.exit(1)
    print("✓")

    # 创建 or 复用
    if args.study_id:
        study_id = args.study_id
        print(f"复用已有任务: {study_id}")
    else:
        scenario = SCENARIOS[args.scenario]
        print(f"\n场景: {args.scenario}  |  n_personas: {args.n_personas}")
        print(f"标题: {scenario['title']}")
        print(f"研究目标: {scenario['study_design']['research_objective']}")
        print(f"\n创建任务...", end=" ")
        t0 = time.time()
        study_id = create_study(scenario, args.n_personas)
        print(f"✓  study_id: {study_id}")

    # 等待完成
    print(f"\n等待 pipeline 完成（每 8 秒刷新）...")
    t0 = time.time()
    if not poll_status(study_id):
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"  完成，耗时 {elapsed:.0f}s")

    # 打印结果
    result = _request("GET", f"/studies/{study_id}/result")
    print_result(result)


if __name__ == "__main__":
    main()

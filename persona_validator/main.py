# main.py
import asyncio
from datetime import datetime
from typing import Callable

from config import settings
from models.schemas import (
    EvaluationMatrix,
    InsightReport,
    PersonaEvaluation,
    PersonaProfile,
    PersonaVector,
    ProductInfo,
)
from pipeline.axis_builder import build_axes
from pipeline.insight_engine import generate_insights
from pipeline.persona_generator import generate_persona
from pipeline.questionnaire_builder import build_questionnaire
from pipeline.responder import simulate_responses
from pipeline.sampler import sample_vectors
from pipeline.scorer import compute_tpb_score
from storage.job_repo import get_job_repo
from utils.export import generate_report

# ── 并行执行工具 ──────────────────────────────────────────────────────────────


async def _run_in_executor(fn: Callable, *args):
    """将同步函数包装为异步，使用默认线程池执行。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def _generate_personas_async(
    vectors: list[PersonaVector],
    axes,
    product: ProductInfo,
) -> list[PersonaProfile]:
    """并行生成所有 persona 描述（Stage 2b）。"""
    tasks = [_run_in_executor(generate_persona, v, axes, product) for v in vectors]
    return await asyncio.gather(*tasks)


async def _simulate_all_async(
    profiles: list[PersonaProfile],
    questionnaire,
) -> list[PersonaEvaluation]:
    """并行执行所有 persona 的问卷回答 + 打分（Stage 3b + 4a）。"""

    async def _evaluate_one(profile: PersonaProfile) -> PersonaEvaluation:
        response = await _run_in_executor(simulate_responses, profile, questionnaire)
        tpb_score = compute_tpb_score(response, questionnaire)
        return PersonaEvaluation(
            persona=profile, response=response, tpb_score=tpb_score
        )

    tasks = [_evaluate_one(p) for p in profiles]
    return await asyncio.gather(*tasks)


# ── 主流程 ────────────────────────────────────────────────────────────────────


async def run_validation_async(
    product: ProductInfo,
    n_personas: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    job_id: str | None = None,
) -> tuple[EvaluationMatrix, InsightReport]:
    """
    完整验证流程（异步版本，Stage 2b 和 Stage 3b 并行执行）。

    job_id: 若提供，则在每个阶段完成后将中间数据持久化到 MongoDB。

    Returns:
        (matrix, report) — matrix 用于生成报告，report 包含 LLM 洞察文字
    """
    n = n_personas or settings.N_PERSONAS
    repo = get_job_repo() if job_id else None

    def log(msg: str):
        if on_progress:
            on_progress(msg)
        else:
            print(msg)

    async def persist(fields: dict) -> None:
        if repo and job_id:
            await repo.patch(job_id, fields)

    # Stage 1：具体化维度 + 语义标签（1次 LLM 调用）
    log(f"[1/5] 生成维度语义标签...")
    axes = build_axes(product)
    await persist({"axes": axes.model_dump()})

    # Stage 2a：均匀采样（无 LLM）
    log(f"[2/5] Sobol 采样 {n} 个向量...")
    vectors = sample_vectors(axes, n=n)
    await persist({"vectors": [v.model_dump() for v in vectors]})

    # Stage 2b：并行生成人物描述（n 次 LLM 调用）
    log(f"[3/5] 并行生成 {n} 个 persona 描述...")
    profiles = await _generate_personas_async(vectors, axes, product)
    await persist({"personas": [p.model_dump() for p in profiles]})

    # Stage 3a：生成问卷（1次 LLM 调用，所有 persona 共用）
    log(f"[4/5] 生成 TPB 问卷...")
    questionnaire = build_questionnaire(product, profiles[0].behavior_goal)
    await persist({"questionnaire": questionnaire.model_dump()})

    # Stage 3b + 4a：并行问卷回答 + 打分（n 次 LLM 调用）
    log(f"[5/5] 并行模拟 {n} 个 persona 回答问卷并打分...")
    evaluations = await _simulate_all_async(profiles, questionnaire)
    await persist({"evaluations": [e.model_dump() for e in evaluations]})

    # Stage 4b：生成洞察报告（1次 LLM 调用）
    log("生成洞察报告...")
    matrix = EvaluationMatrix(product=product, evaluations=evaluations)
    report = generate_insights(matrix)
    await persist({"insights": report.model_dump()})

    return matrix, report


def run_validation(
    product: ProductInfo,
    n_personas: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[EvaluationMatrix, InsightReport]:
    """同步入口，内部调用异步版本。"""
    return asyncio.run(run_validation_async(product, n_personas, on_progress))


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    product = ProductInfo(
        name="钓鱼塘连接App",
        description="一款连接钓鱼爱好者和鱼塘的App，用户可以查看实时鱼情、余位，提前预约鱼塘。",
        target_market="中国城市钓鱼爱好者，25~60岁",
    )

    matrix, report = run_validation(product)

    # ── 终端摘要输出 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("理想用户")
    print("=" * 60)
    print(report.ideal_persona_description)

    print("\n" + "=" * 60)
    print("死区用户")
    print("=" * 60)
    print(report.dead_zone_description)

    print("\n" + "=" * 60)
    print("关键风险")
    print("=" * 60)
    for i, risk in enumerate(report.key_risks, 1):
        print(f"{i}. {risk}")

    print("\n" + "=" * 60)
    print("整体摘要")
    print("=" * 60)
    print(report.summary)

    print("\n" + "=" * 60)
    print("相关性 Top 3")
    print("=" * 60)
    for c in report.top_correlations:
        print(f"  {c['axis']}：{c['correlation']:+.3f}")

    # ── 保存完整报告 ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    print(f"\n[导出] 生成完整报告...")
    report_path = generate_report(matrix, report, filename=filename)
    print(f"[导出] 完成：{report_path}")

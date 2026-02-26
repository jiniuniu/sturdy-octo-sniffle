# pipeline/insight_engine.py
import math

import numpy as np
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from scipy.stats import spearmanr

from models.schemas import (
    BUSINESS_DIMENSIONS,
    DimensionInsight,
    EvaluationMatrix,
    InsightReport,
)
from utils.llm_client import get_llm

# ── LLM 输出内部 Schema（只含文字部分，统计字段由计算注入）────────────────────


class _DimFinding(BaseModel):
    finding: str
    recommendation: str


class _InsightLLMOutput(BaseModel):
    """LLM 只负责生成文字，统计字段在外部计算后合并。"""

    dimension_findings: list[_DimFinding]
    ideal_persona_description: str
    dead_zone_description: str
    key_risks: list[str]
    summary: str


# ── 统计计算 ──────────────────────────────────────────────────────────────────


def _dim_axis_values(evaluations, axis_indices: list[int]) -> list[float]:
    """计算某维度的轴值（多轴取均值）。"""
    return [
        sum(e.persona.vector.axis_indices[i] for i in axis_indices) / len(axis_indices)
        for e in evaluations
    ]


def _compute_label_means(
    evaluations, axis_values: list[float], scores: list[float]
) -> dict[str, float]:
    """
    将连续轴值分入 L1~L4 四个桶，计算每桶的接受度均值。
    多轴合并后轴值为浮点数，用 floor 归桶（上限 3）。
    """
    buckets: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    for val, score in zip(axis_values, scores):
        bucket = min(3, math.floor(val))
        buckets[bucket].append(score)

    label_means: dict[str, float] = {}
    for k, label in enumerate(["L1", "L2", "L3", "L4"]):
        vals = buckets[k]
        label_means[label] = round(float(np.mean(vals)), 3) if vals else float("nan")
    return label_means


def _compute_signal(correlation: float, pvalue: float) -> str:
    if pvalue > 0.1:
        return "数据噪声"
    abs_corr = abs(correlation)
    if abs_corr >= 0.4:
        return "验证通过"
    if abs_corr >= 0.25:
        return "需关注"
    return "风险较高"


def _compute_dim_stats(matrix: EvaluationMatrix, dim: dict) -> dict:
    """计算单个业务维度的所有统计量。"""
    scores = [e.tpb_score.acceptance_score for e in matrix.evaluations]
    axis_values = _dim_axis_values(matrix.evaluations, dim["axis_indices"])

    # 多轴：分别计算相关系数后取均值（两轴通常高度相关，直接均值即可）
    corrs, pvals = [], []
    for ax_idx in dim["axis_indices"]:
        raw_vals = [e.persona.vector.axis_indices[ax_idx] for e in matrix.evaluations]
        c, p = spearmanr(raw_vals, scores)
        corrs.append(float(c) if not np.isnan(c) else 0.0)
        pvals.append(float(p) if not np.isnan(p) else 1.0)

    correlation = round(float(np.mean(corrs)), 3)
    pvalue = round(float(np.mean(pvals)), 3)

    return {
        "correlation": correlation,
        "pvalue": pvalue,
        "label_means": _compute_label_means(matrix.evaluations, axis_values, scores),
        "signal": _compute_signal(correlation, pvalue),
    }


# ── Prompt 组装 ───────────────────────────────────────────────────────────────

_PROMPT = """\
你是一位专注于早期创业验证的产品策略顾问，擅长从合成用户数据中提炼可执行的业务洞察。

## 产品信息
- 产品名称：{product_name}
- 产品描述：{product_description}
- 目标市场：{product_target_market}

## 测试概况
- 模拟用户数：{n_personas} 位
- 整体接受度：均值 {mean_score}，标准差 {std_score}（越大说明用户分化越明显）
- TPB 障碍：态度 {mean_attitude} ｜ 主观规范 {mean_subjective_norm} ｜ 感知行为控制 {mean_perceived_control}

## 高接受度用户（前50%）的典型描述
{high_score_profiles}

## 低接受度用户（后50%）的典型描述
{low_score_profiles}

## 五大业务维度数据
{dimensions_context}

---

## 你的任务

### 第一部分：五大维度逐一分析（严格按上方维度顺序，共 {n_dims} 条）

对每个维度输出：
- **finding**（2句话）
  - 第1句：引用具体数字说明现象，例如"L4用户接受度均值（{example_l4}）是L1用户（{example_l1}）的X倍"
  - 第2句：给出业务判断，回答该维度的业务问题
- **recommendation**（1句话）：创始人下一步最该做的一个具体、可执行的验证动作

### 第二部分：用户分类与整体总结

- **ideal_persona_description**：2~3句，综合高接受度用户共同特征，要有画面感（职业、年龄、生活状态）
- **dead_zone_description**：2~3句，综合低接受度用户共同特征，明确说明他们卡在哪个 TPB 环节
- **key_risks**：3~5条系统性风险，每条需包含：风险性质（价值未被认可 or 用户无法行动 or 缺乏社会支持）+ 数据依据 + 一句针对性建议
- **summary**：3~5句整体判断，结合五大维度信号，给出最值得优先验证的方向

## 输出格式
{format_instructions}
"""


def _build_dimensions_context(matrix: EvaluationMatrix, dim_stats_list: list[dict]) -> str:
    """将五大维度的数值数据格式化为 prompt 中的可读文本。"""
    scores = np.array([e.tpb_score.acceptance_score for e in matrix.evaluations])
    median = float(np.median(scores))

    lines = []
    for dim, stats in zip(BUSINESS_DIMENSIONS, dim_stats_list):
        corr = stats["correlation"]
        pval = stats["pvalue"]
        lm = stats["label_means"]
        sig_marker = " *" if pval < 0.05 else "（趋势）"
        direction = "正相关↑" if corr >= 0 else "负相关↓"

        lm_str = " | ".join(
            f"L{i+1}={lm.get(f'L{i+1}', float('nan')):.2f}" for i in range(4)
        )

        # 取该维度高/低接受度代表用户各2个
        high_reps = [
            e.persona.description
            for e in sorted(matrix.evaluations, key=lambda e: e.tpb_score.acceptance_score, reverse=True)
            if e.tpb_score.acceptance_score >= median
        ][:2]
        low_reps = [
            e.persona.description
            for e in sorted(matrix.evaluations, key=lambda e: e.tpb_score.acceptance_score)
            if e.tpb_score.acceptance_score < median
        ][:2]

        lines += [
            f"### 维度：{dim['name']}",
            f"业务问题：{dim['question']}",
            f"关键 TPB 障碍：{dim['key_tpb']}",
            f"相关系数：{corr:+.3f}（{direction}{sig_marker}）｜ 信号：{stats['signal']}",
            f"各层级接受度均值：{lm_str}",
            f"高接受度代表用户：{' / '.join(high_reps)}",
            f"低接受度代表用户：{' / '.join(low_reps)}",
            "",
        ]
    return "\n".join(lines)


# ── 主函数 ────────────────────────────────────────────────────────────────────


def generate_insights(matrix: EvaluationMatrix) -> InsightReport:
    scores = np.array([e.tpb_score.acceptance_score for e in matrix.evaluations])
    median_val = float(np.median(scores))

    # 1. 纯计算：五大维度统计量
    dim_stats_list = [_compute_dim_stats(matrix, dim) for dim in BUSINESS_DIMENSIONS]

    # 2. 高/低分组 persona 描述
    high_profiles = [
        e.persona.description
        for e in sorted(matrix.evaluations, key=lambda e: e.tpb_score.acceptance_score, reverse=True)
        if e.tpb_score.acceptance_score >= median_val
    ]
    low_profiles = [
        e.persona.description
        for e in sorted(matrix.evaluations, key=lambda e: e.tpb_score.acceptance_score)
        if e.tpb_score.acceptance_score < median_val
    ]

    dimensions_context = _build_dimensions_context(matrix, dim_stats_list)

    # 3. LLM 生成文字部分
    parser = PydanticOutputParser(pydantic_object=_InsightLLMOutput)
    chain = ChatPromptTemplate.from_template(_PROMPT) | get_llm() | parser

    # 取第一个维度的 L1/L4 示例值，用于 prompt 中的示例占位
    first_lm = dim_stats_list[0]["label_means"]
    result: _InsightLLMOutput = chain.invoke(
        {
            "product_name": matrix.product.name,
            "product_description": matrix.product.description,
            "product_target_market": matrix.product.target_market,
            "n_personas": len(matrix.evaluations),
            "mean_score": round(float(np.mean(scores)), 3),
            "std_score": round(float(np.std(scores)), 3),
            "mean_attitude": round(float(np.mean([e.tpb_score.attitude for e in matrix.evaluations])), 3),
            "mean_subjective_norm": round(float(np.mean([e.tpb_score.subjective_norm for e in matrix.evaluations])), 3),
            "mean_perceived_control": round(float(np.mean([e.tpb_score.perceived_control for e in matrix.evaluations])), 3),
            "high_score_profiles": "\n".join(f"  - {p}" for p in high_profiles[:5]),
            "low_score_profiles": "\n".join(f"  - {p}" for p in low_profiles[:5]),
            "dimensions_context": dimensions_context,
            "n_dims": len(BUSINESS_DIMENSIONS),
            "example_l4": first_lm.get("L4", "N/A"),
            "example_l1": first_lm.get("L1", "N/A"),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # 4. 合并计算结果 + LLM 文字 → DimensionInsight 列表
    #    LLM 输出的 dimension_findings 可能比 BUSINESS_DIMENSIONS 少（截断保护）
    dim_insights: list[DimensionInsight] = []
    for i, (dim, stats) in enumerate(zip(BUSINESS_DIMENSIONS, dim_stats_list)):
        llm_finding = (
            result.dimension_findings[i]
            if i < len(result.dimension_findings)
            else _DimFinding(finding="（数据待补充）", recommendation="（待补充）")
        )
        dim_insights.append(
            DimensionInsight(
                dimension_name=dim["name"],
                business_question=dim["question"],
                axis_names=dim["axis_names"],
                correlation=stats["correlation"],
                pvalue=stats["pvalue"],
                label_means=stats["label_means"],
                key_tpb_barrier=dim["key_tpb"],
                signal=stats["signal"],
                finding=llm_finding.finding,
                recommendation=llm_finding.recommendation,
            )
        )

    return InsightReport(
        dimension_insights=dim_insights,
        ideal_persona_description=result.ideal_persona_description,
        dead_zone_description=result.dead_zone_description,
        key_risks=result.key_risks,
        summary=result.summary,
    )


# ── 命令行测试 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models.schemas import (
        PersonaEvaluation,
        PersonaProfile,
        PersonaVector,
        ProductInfo,
        QuestionnaireResponse,
        SingleResponse,
        TPBScore,
    )

    product = ProductInfo(
        name="钓鱼塘连接App",
        description="一款连接钓鱼爱好者和鱼塘的App，用户可以查看实时鱼情、余位，提前预约鱼塘。",
        target_market="中国城市钓鱼爱好者，25~60岁",
    )

    high_configs = [
        ([3, 0, 3, 3, 0, 3], "李明，32岁，互联网从业者，经常找不到合适鱼塘，愿意为便利付费。", TPBScore(attitude=0.85, subjective_norm=0.70, perceived_control=0.90, acceptance_score=0.818)),
        ([3, 1, 2, 3, 1, 3], "王芳，28岁，自由职业者，把钓鱼当减压方式，愿意用App简化流程。", TPBScore(attitude=0.80, subjective_norm=0.65, perceived_control=0.85, acceptance_score=0.768)),
        ([2, 0, 3, 2, 0, 2], "陈强，35岁，销售经理，希望能快速查到有鱼的地方。", TPBScore(attitude=0.75, subjective_norm=0.75, perceived_control=0.80, acceptance_score=0.763)),
        ([3, 1, 2, 3, 0, 3], "张伟，40岁，私企老板，时间有限，希望提前确认鱼情余位。", TPBScore(attitude=0.78, subjective_norm=0.60, perceived_control=0.88, acceptance_score=0.754)),
        ([3, 0, 3, 3, 1, 2], "刘洋，25岁，把钓鱼当周末消遣，乐于尝试新工具。", TPBScore(attitude=0.82, subjective_norm=0.72, perceived_control=0.92, acceptance_score=0.822)),
        ([2, 1, 2, 2, 1, 3], "赵磊，38岁，工程师，觉得有实时鱼情数据是刚需。", TPBScore(attitude=0.73, subjective_norm=0.58, perceived_control=0.80, acceptance_score=0.714)),
        ([3, 0, 3, 3, 0, 3], "孙浩，30岁，设计师，热衷新工具。", TPBScore(attitude=0.88, subjective_norm=0.68, perceived_control=0.93, acceptance_score=0.834)),
        ([2, 1, 2, 3, 1, 2], "周鑫，33岁，教师，觉得现有方式效率太低。", TPBScore(attitude=0.76, subjective_norm=0.62, perceived_control=0.82, acceptance_score=0.734)),
    ]
    low_configs = [
        ([0, 3, 0, 0, 3, 0], "老王，60岁，退休工人，有固定鱼塘二十年，不需要任何App。", TPBScore(attitude=0.25, subjective_norm=0.20, perceived_control=0.30, acceptance_score=0.245)),
        ([1, 2, 0, 0, 2, 0], "李大爷，58岁，觉得钓鱼就是图个安静，不想被手机打扰。", TPBScore(attitude=0.20, subjective_norm=0.15, perceived_control=0.25, acceptance_score=0.198)),
        ([0, 3, 1, 0, 3, 0], "张叔，55岁，国企职工，觉得App多此一举。", TPBScore(attitude=0.22, subjective_norm=0.18, perceived_control=0.28, acceptance_score=0.222)),
        ([1, 3, 0, 0, 2, 1], "刘师傅，57岁，从没想过用手机预约。", TPBScore(attitude=0.18, subjective_norm=0.22, perceived_control=0.20, acceptance_score=0.198)),
        ([0, 2, 0, 0, 3, 0], "陈老，62岁，觉得信息化对钓鱼没什么意义。", TPBScore(attitude=0.15, subjective_norm=0.12, perceived_control=0.18, acceptance_score=0.148)),
        ([1, 2, 1, 0, 3, 1], "赵大哥，52岁，不想为此学新软件。", TPBScore(attitude=0.28, subjective_norm=0.25, perceived_control=0.22, acceptance_score=0.254)),
        ([0, 3, 0, 0, 2, 0], "吴师傅，59岁，消息靠口耳相传，不用App。", TPBScore(attitude=0.20, subjective_norm=0.17, perceived_control=0.15, acceptance_score=0.175)),
        ([1, 2, 0, 0, 3, 0], "郑叔，56岁，觉得预约App解决不了根本问题。", TPBScore(attitude=0.23, subjective_norm=0.19, perceived_control=0.21, acceptance_score=0.211)),
    ]

    mock_evaluations = []
    for i, (indices, desc, tpb) in enumerate(high_configs + low_configs):
        labels = [str(idx) for idx in indices]
        mock_evaluations.append(PersonaEvaluation(
            persona=PersonaProfile(
                id=f"persona_{i:03d}",
                vector=PersonaVector(id=f"persona_{i:03d}", axis_indices=indices, axis_labels=labels),
                description=desc,
                behavior_goal="下载App后查看附近鱼塘的实时余位并完成首次预约",
            ),
            response=QuestionnaireResponse(
                persona_id=f"persona_{i:03d}",
                responses={"q01": SingleResponse(score=3, reasoning="mock")},
            ),
            tpb_score=tpb,
        ))

    matrix = EvaluationMatrix(product=product, evaluations=mock_evaluations)
    report = generate_insights(matrix)

    print("=" * 60)
    for di in report.dimension_insights:
        print(f"\n【{di.dimension_name}】{di.signal}")
        print(f"  {di.business_question}")
        print(f"  相关系数：{di.correlation:+.3f}  label_means: {di.label_means}")
        print(f"  Finding: {di.finding}")
        print(f"  建议: {di.recommendation}")

    print("\n" + "=" * 60)
    print("整体摘要")
    print("=" * 60)
    print(report.summary)

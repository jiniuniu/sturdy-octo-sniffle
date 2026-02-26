# pipeline/insight_engine.py
import numpy as np
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import PREDEFINED_AXIS_NAMES, EvaluationMatrix, InsightReport
from utils.llm_client import get_llm

PROMPT = """\
你是一位专注于早期创业验证的顾问，擅长从用户模拟数据中识别产品风险和机会。

## 任务背景

我们用合成 persona 模拟了 {n_personas} 位不同类型的用户，评估他们对某产品的接受程度（0~1分）。
你需要根据以下数值分析结果，生成一份有洞察力的产品验证报告。

## 产品信息

- 产品名称：{product_name}
- 产品描述：{product_description}

## 数值分析结果

**接受度分布：**
- 均值：{mean_score}
- 中位数：{median_score}
- 最高分：{max_score}（最理想用户）
- 最低分：{min_score}（最难转化用户）
- 标准差：{std_score}（越大说明用户分化越严重）

**各维度与接受度的相关性（Spearman，值域 -1~1）：**
{correlations_text}

正相关：该维度越高的用户，接受度越高
负相关：该维度越高的用户，接受度越低
绝对值越大：该维度对接受度的影响越强

**高接受度用户（前50%）的典型描述：**
{high_score_profiles}

**低接受度用户（后50%）的典型描述：**
{low_score_profiles}

**TPB 三维度均值（态度/主观规范/感知行为控制）：**
- 态度（attitude）均值：{mean_attitude}
- 主观规范（subjective_norm）均值：{mean_subjective_norm}
- 感知行为控制（perceived_control）均值：{mean_perceived_control}

## 你的任务

请生成一份洞察报告，包含以下内容：

**1. ideal_persona_description（理想用户描述）**
综合高接受度用户的共同特征，用2~3句话描述最可能成为早期用户的人群。
要具体，不要只说"痛点强、付费意愿高"，而要说清楚这类人的生活状态和行为特征。

**2. dead_zone_description（死区描述）**
综合低接受度用户的共同特征，描述哪类人群几乎不可能转化。
说明他们卡在哪里——是不认可产品价值（attitude低），还是没有外部支持（subjective_norm低），\
还是觉得自己做不到（perceived_control低）。

**3. key_risks（关键风险）**
列出 3~5 条系统性风险（不是个别用户的问题，而是大多数用户都面临的障碍）。
每条风险需要：
- 指出具体是哪个维度驱动的
- 区分是"产品价值未被认可"还是"用户无法或不愿行动"这两类不同性质的问题
- 给出 1 句针对性的建议方向

**4. summary（整体摘要）**
用 3~5 句话总结本次验证的核心发现：
- 产品对哪类用户有吸引力，对哪类用户没有
- 目前最大的风险是什么
- 建议创始人下一步优先验证什么

## 输出要求

{format_instructions}
"""


def _build_correlations_text(matrix: EvaluationMatrix) -> tuple[list[dict], str]:
    """计算每个输入维度与 acceptance_score 的 Spearman 相关系数，返回结构化数据和可读文本。"""
    scores = np.array([e.tpb_score.acceptance_score for e in matrix.evaluations])

    correlations = []
    for i, axis_name in enumerate(PREDEFINED_AXIS_NAMES):
        axis_values = np.array(
            [e.persona.vector.axis_indices[i] for e in matrix.evaluations]
        )

        # 使用 Spearman 相关（基于排名），对序数型的 label index 更合适
        from scipy.stats import spearmanr

        corr, pvalue = spearmanr(axis_values, scores)
        corr = float(corr) if not np.isnan(corr) else 0.0

        correlations.append(
            {
                "axis": axis_name,
                "correlation": round(corr, 3),
                "pvalue": round(float(pvalue), 3) if not np.isnan(pvalue) else 1.0,
            }
        )

    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    lines = []
    for c in correlations:
        direction = "正相关 ↑" if c["correlation"] > 0 else "负相关 ↓"
        sig = " *" if c["pvalue"] < 0.05 else ""
        lines.append(f"  - {c['axis']}：{c['correlation']:+.3f} ({direction}){sig}")

    return correlations, "\n".join(lines)


def _split_by_median(matrix: EvaluationMatrix) -> tuple[list[str], list[str]]:
    """按接受度中位数将 persona 分为高低两组，返回各组描述列表。"""
    scores = [e.tpb_score.acceptance_score for e in matrix.evaluations]
    median = float(np.median(scores))

    high = [
        e.persona.description
        for e in matrix.evaluations
        if e.tpb_score.acceptance_score >= median
    ]
    low = [
        e.persona.description
        for e in matrix.evaluations
        if e.tpb_score.acceptance_score < median
    ]

    return high, low


def generate_insights(matrix: EvaluationMatrix) -> InsightReport:
    scores = np.array([e.tpb_score.acceptance_score for e in matrix.evaluations])

    attitudes = np.array([e.tpb_score.attitude for e in matrix.evaluations])
    subjective_norms = np.array(
        [e.tpb_score.subjective_norm for e in matrix.evaluations]
    )
    perceived_controls = np.array(
        [e.tpb_score.perceived_control for e in matrix.evaluations]
    )

    correlations, correlations_text = _build_correlations_text(matrix)
    high_profiles, low_profiles = _split_by_median(matrix)

    parser = PydanticOutputParser(pydantic_object=InsightReport)
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser

    result: InsightReport = chain.invoke(
        {
            "n_personas": len(matrix.evaluations),
            "product_name": matrix.product.name,
            "product_description": matrix.product.description,
            "mean_score": round(float(np.mean(scores)), 3),
            "median_score": round(float(np.median(scores)), 3),
            "max_score": round(float(np.max(scores)), 3),
            "min_score": round(float(np.min(scores)), 3),
            "std_score": round(float(np.std(scores)), 3),
            "correlations_text": correlations_text,
            "high_score_profiles": "\n".join(f"  - {p}" for p in high_profiles[:6]),
            "low_score_profiles": "\n".join(f"  - {p}" for p in low_profiles[:6]),
            "mean_attitude": round(float(np.mean(attitudes)), 3),
            "mean_subjective_norm": round(float(np.mean(subjective_norms)), 3),
            "mean_perceived_control": round(float(np.mean(perceived_controls)), 3),
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return result.model_copy(update={"top_correlations": correlations[:3]})


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

    # 构造16个 mock PersonaEvaluation，模拟两极分化的结果
    # 前8个：高接受度（年轻、痛点强、付费意愿高）
    # 后8个：低接受度（年龄大、依赖现有方案、对新App抵触）
    mock_evaluations = []

    high_configs = [
        (
            [3, 0, 3, 3, 0, 3],
            "李明，32岁，互联网从业者，钓鱼两年，经常找不到合适鱼塘，愿意为便利付费。",
            TPBScore(
                attitude=0.85,
                subjective_norm=0.70,
                perceived_control=0.90,
                acceptance_score=0.818,
            ),
        ),
        (
            [3, 1, 2, 3, 1, 3],
            "王芳，28岁，自由职业者，把钓鱼当减压方式，愿意用App简化预约流程。",
            TPBScore(
                attitude=0.80,
                subjective_norm=0.65,
                perceived_control=0.85,
                acceptance_score=0.768,
            ),
        ),
        (
            [2, 0, 3, 2, 0, 2],
            "陈强，35岁，销售经理，钓鱼是社交工具，希望能快速查到有鱼的地方。",
            TPBScore(
                attitude=0.75,
                subjective_norm=0.75,
                perceived_control=0.80,
                acceptance_score=0.763,
            ),
        ),
        (
            [3, 1, 2, 3, 0, 3],
            "张伟，40岁，私企老板，时间有限，希望每次出行都能提前确认鱼情余位。",
            TPBScore(
                attitude=0.78,
                subjective_norm=0.60,
                perceived_control=0.88,
                acceptance_score=0.754,
            ),
        ),
        (
            [3, 0, 3, 3, 1, 2],
            "刘洋，25岁，刚工作两年，把钓鱼当周末消遣，乐于尝试新工具。",
            TPBScore(
                attitude=0.82,
                subjective_norm=0.72,
                perceived_control=0.92,
                acceptance_score=0.822,
            ),
        ),
        (
            [2, 1, 2, 2, 1, 3],
            "赵磊，38岁，工程师，逻辑思维强，觉得有实时鱼情数据是刚需。",
            TPBScore(
                attitude=0.73,
                subjective_norm=0.58,
                perceived_control=0.80,
                acceptance_score=0.714,
            ),
        ),
        (
            [3, 0, 3, 3, 0, 3],
            "孙浩，30岁，设计师，把找鱼塘当作一个可以被优化的问题，热衷新工具。",
            TPBScore(
                attitude=0.88,
                subjective_norm=0.68,
                perceived_control=0.93,
                acceptance_score=0.834,
            ),
        ),
        (
            [2, 1, 2, 3, 1, 2],
            "周鑫，33岁，教师，钓鱼频率高，觉得现有方式效率太低，想改进。",
            TPBScore(
                attitude=0.76,
                subjective_norm=0.62,
                perceived_control=0.82,
                acceptance_score=0.734,
            ),
        ),
    ]

    low_configs = [
        (
            [0, 3, 0, 0, 3, 0],
            "老王，60岁，退休工人，有固定鱼塘二十年，和塘主是老朋友，不需要任何App。",
            TPBScore(
                attitude=0.25,
                subjective_norm=0.20,
                perceived_control=0.30,
                acceptance_score=0.245,
            ),
        ),
        (
            [1, 2, 0, 0, 2, 0],
            "李大爷，58岁，退休干部，觉得钓鱼就是图个安静，不想被手机打扰。",
            TPBScore(
                attitude=0.20,
                subjective_norm=0.15,
                perceived_control=0.25,
                acceptance_score=0.198,
            ),
        ),
        (
            [0, 3, 1, 0, 3, 0],
            "张叔，55岁，国企职工，钓鱼只在固定地点，觉得App多此一举。",
            TPBScore(
                attitude=0.22,
                subjective_norm=0.18,
                perceived_control=0.28,
                acceptance_score=0.222,
            ),
        ),
        (
            [1, 3, 0, 0, 2, 1],
            "刘师傅，57岁，司机，周末固定去附近水库，从没想过用手机预约。",
            TPBScore(
                attitude=0.18,
                subjective_norm=0.22,
                perceived_control=0.20,
                acceptance_score=0.198,
            ),
        ),
        (
            [0, 2, 0, 0, 3, 0],
            "陈老，62岁，退休教师，钓鱼三十年，觉得信息化对钓鱼来说没什么意义。",
            TPBScore(
                attitude=0.15,
                subjective_norm=0.12,
                perceived_control=0.18,
                acceptance_score=0.148,
            ),
        ),
        (
            [1, 2, 1, 0, 3, 1],
            "赵大哥，52岁，厂长，钓鱼是放松方式，不想为此学新软件。",
            TPBScore(
                attitude=0.28,
                subjective_norm=0.25,
                perceived_control=0.22,
                acceptance_score=0.254,
            ),
        ),
        (
            [0, 3, 0, 0, 2, 0],
            "吴师傅，59岁，水电工，有稳定钓友圈子，消息都靠口耳相传，不用App。",
            TPBScore(
                attitude=0.20,
                subjective_norm=0.17,
                perceived_control=0.15,
                acceptance_score=0.175,
            ),
        ),
        (
            [1, 2, 0, 0, 3, 0],
            "郑叔，56岁，工厂班长，钓鱼是唯一爱好但时间受限，觉得预约App解决不了根本问题。",
            TPBScore(
                attitude=0.23,
                subjective_norm=0.19,
                perceived_control=0.21,
                acceptance_score=0.211,
            ),
        ),
    ]

    for i, (indices, desc, tpb) in enumerate(high_configs + low_configs):
        labels = [
            [
                "从没觉得找鱼塘有什么问题",
                "偶尔觉得麻烦但不在意",
                "经常为找不到合适鱼塘烦恼",
                "强烈感受到找鱼塘费时费力",
            ][idx]
            for idx in indices
        ]
        mock_evaluations.append(
            PersonaEvaluation(
                persona=PersonaProfile(
                    id=f"persona_{i:03d}",
                    vector=PersonaVector(
                        id=f"persona_{i:03d}", axis_indices=indices, axis_labels=labels
                    ),
                    description=desc,
                    behavior_goal="下载App后查看附近鱼塘的实时余位并完成首次预约",
                ),
                response=QuestionnaireResponse(
                    persona_id=f"persona_{i:03d}",
                    responses={"q01": SingleResponse(score=3, reasoning="mock")},
                ),
                tpb_score=tpb,
            )
        )

    matrix = EvaluationMatrix(product=product, evaluations=mock_evaluations)
    report = generate_insights(matrix)

    print("=" * 60)
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

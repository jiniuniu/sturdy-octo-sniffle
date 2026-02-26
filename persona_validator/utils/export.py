# utils/export.py
import base64
import io
import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from models.schemas import PREDEFINED_AXIS_NAMES, EvaluationMatrix, InsightReport
from scipy.stats import spearmanr

matplotlib.use("Agg")  # 非交互后端，适合服务端生成图片

# ── 字体配置（支持中文）────────────────────────────────────────────────────────
rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "./outputs"

# 配色
COLOR_HIGH = "#4CAF50"  # 绿：高接受度
COLOR_LOW = "#F44336"  # 红：低接受度
COLOR_MEDIAN = "#9E9E9E"  # 灰：中位线
COLOR_POS = "#2196F3"  # 蓝：正相关
COLOR_NEG = "#FF9800"  # 橙：负相关
COLOR_BG = "#FAFAFA"


# ── 统计数据结构 ───────────────────────────────────────────────────────────────


@dataclass
class ReportStats:
    n_personas: int
    mean_score: float
    median_score: float
    max_score: float
    min_score: float
    std_score: float
    mean_attitude: float
    mean_subjective_norm: float
    mean_perceived_control: float
    correlations: list[dict]  # [{axis, correlation, pvalue}, ...]
    scores: list[float]  # 每个 persona 的 acceptance_score
    persona_labels: list[str]  # 每个 persona 的简短标签（姓名/id）
    axis_score_groups: list[dict]  # 每个维度下 L1~L4 各自的 acceptance_score 列表


def build_stats(matrix: EvaluationMatrix) -> ReportStats:
    scores = [e.tpb_score.acceptance_score for e in matrix.evaluations]
    np_scores = np.array(scores)

    # persona 简短标签：取 description 第一个逗号前的内容，fallback 到 id
    labels = []
    for e in matrix.evaluations:
        desc = e.persona.description
        label = desc.split("，")[0].split(",")[0].strip()
        labels.append(label if label else e.persona.id)

    # Spearman 相关性
    correlations = []
    for i, axis_name in enumerate(PREDEFINED_AXIS_NAMES):
        axis_vals = [e.persona.vector.axis_indices[i] for e in matrix.evaluations]
        corr, pval = spearmanr(axis_vals, scores)
        correlations.append(
            {
                "axis": axis_name,
                "correlation": round(float(corr) if not np.isnan(corr) else 0.0, 3),
                "pvalue": round(float(pval) if not np.isnan(pval) else 1.0, 3),
            }
        )
    correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # 分维度 L1~L4 的接受度分组
    # 需要从 matrix 中取 axes 的 labels——从第一个 persona 的 vector 重建
    # axis_score_groups[i] = {"axis": name, "labels": [L1,L2,L3,L4], "scores": [[...],[...],[...],[...]]}
    axis_score_groups = []
    for i, axis_name in enumerate(PREDEFINED_AXIS_NAMES):
        # 收集每个 label index 对应的分数
        groups: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
        for e in matrix.evaluations:
            idx = e.persona.vector.axis_indices[i]
            groups[idx].append(e.tpb_score.acceptance_score)

        # 取语义标签（从第一个有该 index 的 persona 的 axis_labels 取）
        semantic_labels = ["L1", "L2", "L3", "L4"]
        for e in matrix.evaluations:
            if len(e.persona.vector.axis_labels) > i:
                # 每个 index 对应的标签是固定的，取任意一个 persona 的即可
                # 但我们只有 axis_labels（按采样顺序），需要反查
                # 简化处理：直接用 index 作为 x 轴，label 用维度名+L1~L4
                pass
        # axis_labels 存的是该 persona 在该维度上的标签值，不是全部4个标签
        # 所以这里用 L1~L4 作为 x 轴标签
        axis_score_groups.append(
            {
                "axis": axis_name,
                "labels": ["L1（最低）", "L2", "L3", "L4（最高）"],
                "scores": [groups[j] for j in range(4)],
            }
        )

    return ReportStats(
        n_personas=len(matrix.evaluations),
        mean_score=round(float(np.mean(np_scores)), 3),
        median_score=round(float(np.median(np_scores)), 3),
        max_score=round(float(np.max(np_scores)), 3),
        min_score=round(float(np.min(np_scores)), 3),
        std_score=round(float(np.std(np_scores)), 3),
        mean_attitude=round(
            float(np.mean([e.tpb_score.attitude for e in matrix.evaluations])), 3
        ),
        mean_subjective_norm=round(
            float(np.mean([e.tpb_score.subjective_norm for e in matrix.evaluations])), 3
        ),
        mean_perceived_control=round(
            float(np.mean([e.tpb_score.perceived_control for e in matrix.evaluations])),
            3,
        ),
        correlations=correlations,
        scores=scores,
        persona_labels=labels,
        axis_score_groups=axis_score_groups,
    )


# ── 图表工具 ───────────────────────────────────────────────────────────────────


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _fig_to_file(fig: plt.Figure, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    return path


# ── 图表一：接受度分布（横向点状条形图）─────────────────────────────────────


def chart_distribution(stats: ReportStats, save_file: bool = True) -> str:
    n = stats.n_personas
    scores = stats.scores
    labels = stats.persona_labels
    median = stats.median_score

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.45)))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # 按分数排序
    order = sorted(range(n), key=lambda i: scores[i])
    sorted_scores = [scores[i] for i in order]
    sorted_labels = [labels[i] for i in order]

    colors = [COLOR_HIGH if s >= median else COLOR_LOW for s in sorted_scores]

    ax.barh(range(n), sorted_scores, color=colors, height=0.6, alpha=0.85)
    ax.axvline(
        median,
        color=COLOR_MEDIAN,
        linestyle="--",
        linewidth=1.2,
        label=f"中位数 {median:.2f}",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("接受度分数（0~1）", fontsize=10)
    ax.set_title("每位模拟用户的接受度分数", fontsize=12, fontweight="bold", pad=12)

    high_patch = mpatches.Patch(color=COLOR_HIGH, label=f"高接受度（≥{median:.2f}）")
    low_patch = mpatches.Patch(color=COLOR_LOW, label=f"低接受度（<{median:.2f}）")
    ax.legend(handles=[high_patch, low_patch], fontsize=8, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_file:
        return _fig_to_file(fig, "chart_distribution.png")
    return _fig_to_base64(fig)


# ── 图表二：维度相关性条形图 ──────────────────────────────────────────────────


def chart_correlations(stats: ReportStats, save_file: bool = True) -> str:
    corrs = stats.correlations
    axes_names = [c["axis"] for c in corrs]
    values = [c["correlation"] for c in corrs]
    pvalues = [c["pvalue"] for c in corrs]

    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in values]
    # 显著性用透明度区分
    alphas = [0.9 if p < 0.05 else 0.45 for p in pvalues]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    bars = ax.barh(
        range(len(axes_names)), values, color=colors, alpha=0.85, height=0.55
    )

    # 透明度：不显著的降低
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # 标注数值和显著性
    for i, (v, p) in enumerate(zip(values, pvalues)):
        sig = " *" if p < 0.05 else ""
        offset = 0.02 if v >= 0 else -0.02
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.2f}{sig}", va="center", ha=ha, fontsize=8)

    ax.set_yticks(range(len(axes_names)))
    ax.set_yticklabels(axes_names, fontsize=9)
    ax.axvline(0, color="#BDBDBD", linewidth=0.8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("与接受度的相关系数（* 表示统计显著）", fontsize=9)
    ax.set_title("哪类用户特征与接受度最相关？", fontsize=12, fontweight="bold", pad=12)

    pos_patch = mpatches.Patch(color=COLOR_POS, label="正相关：该特征越高，接受度越高")
    neg_patch = mpatches.Patch(color=COLOR_NEG, label="负相关：该特征越高，接受度越低")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_file:
        return _fig_to_file(fig, "chart_correlations.png")
    return _fig_to_base64(fig)


# ── 图表三：三大障碍进度条 ────────────────────────────────────────────────────


def chart_tpb_barriers(stats: ReportStats, save_file: bool = True) -> str:
    labels = [
        "认可这件事的价值\n（态度）",
        "身边人是否支持\n（社会影响）",
        "觉得自己能做到\n（行动信心）",
    ]
    values = [
        stats.mean_attitude,
        stats.mean_subjective_norm,
        stats.mean_perceived_control,
    ]

    # 颜色按分值：>0.6 绿，0.4~0.6 橙，<0.4 红
    def bar_color(v):
        if v >= 0.6:
            return "#4CAF50"
        if v >= 0.4:
            return "#FF9800"
        return "#F44336"

    colors = [bar_color(v) for v in values]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle("用户在三个层面的整体表现", fontsize=12, fontweight="bold", y=1.02)

    for ax, label, val, color in zip(axes, labels, values, colors):
        ax.set_facecolor(COLOR_BG)
        # 背景灰色进度条
        ax.barh([0], [1], color="#E0E0E0", height=0.5)
        # 实际值
        ax.barh([0], [val], color=color, height=0.5, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1.0"], fontsize=8)
        ax.set_title(label, fontsize=9, pad=8)
        ax.text(
            val / 2,
            0,
            f"{val:.2f}",
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="white",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.tight_layout()

    if save_file:
        return _fig_to_file(fig, "chart_tpb_barriers.png")
    return _fig_to_base64(fig)


# ── 图表四：分维度接受度箱线图 ────────────────────────────────────────────────


def chart_axis_breakdown(stats: ReportStats, save_file: bool = True) -> str:
    groups = stats.axis_score_groups
    n_axes = len(groups)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(
        "不同类型用户的接受度分布（按维度细分）", fontsize=12, fontweight="bold", y=1.01
    )

    for idx, (ax, group) in enumerate(zip(axes.flat, groups)):
        ax.set_facecolor(COLOR_BG)
        data = [s for s in group["scores"] if s]  # 过滤空组
        labels = [group["labels"][j] for j, s in enumerate(group["scores"]) if s]
        positions = list(range(len(data)))

        if not data:
            ax.text(
                0.5, 0.5, "数据不足", ha="center", va="center", transform=ax.transAxes
            )
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#757575"),
            capprops=dict(color="#757575"),
            flierprops=dict(marker="o", markersize=4, color="#BDBDBD"),
        )

        # 渐变色：从红到绿
        gradient = ["#F44336", "#FF9800", "#8BC34A", "#4CAF50"]
        for patch, color in zip(bp["boxes"], gradient[: len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("接受度" if idx % 3 == 0 else "", fontsize=8)
        ax.set_title(group["axis"], fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.5, color="#BDBDBD", linestyle="--", linewidth=0.8)

    plt.tight_layout()

    if save_file:
        return _fig_to_file(fig, "chart_axis_breakdown.png")
    return _fig_to_base64(fig)


# ── Markdown 报告渲染 ─────────────────────────────────────────────────────────


def _score_to_label(score: float) -> str:
    if score >= 0.7:
        return "🟢 较高"
    if score >= 0.5:
        return "🟡 中等"
    if score >= 0.3:
        return "🟠 偏低"
    return "🔴 较低"


def _barrier_interpretation(label: str, value: float) -> str:
    hints = {
        "mean_attitude": {
            "high": "用户普遍认可这件事的价值，产品方向得到认同。",
            "mid": "部分用户认可价值，但还有相当比例的人觉得没必要，需要强化价值传达。",
            "low": "大多数用户不认为这件事值得做，产品核心价值主张需要重新审视。",
        },
        "mean_subjective_norm": {
            "high": "用户身边的人普遍支持，口碑传播条件良好。",
            "mid": "社会支持一般，依靠口碑增长会比较慢，可以考虑建立社群。",
            "low": "用户缺乏来自身边人的支持，这会显著阻碍行动，需要创造社交认同感。",
        },
        "mean_perceived_control": {
            "high": "用户普遍觉得自己能做到，产品上手门槛不是主要障碍。",
            "mid": "部分用户对自己能否完成操作有顾虑，需要优化引导流程。",
            "low": "很多用户觉得自己做不到，产品复杂度或使用门槛需要大幅降低。",
        },
    }
    level = "high" if value >= 0.6 else ("mid" if value >= 0.4 else "low")
    return hints.get(label, {}).get(level, "")


def render_report(
    matrix: EvaluationMatrix,
    report: InsightReport,
    stats: ReportStats,
    chart_paths: dict[str, str],
) -> str:
    """
    将统计数据、图表路径和 LLM 洞察组合成完整的 Markdown 报告字符串。
    chart_paths: {"distribution": path, "correlations": path, "barriers": path, "breakdown": path}
    """
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    p = matrix.product

    def img(path: str) -> str:
        return f"![chart]({path})"

    lines = []

    # ── 封面 ──────────────────────────────────────────────────────────────────
    lines += [
        f"# {p.name} · 用户接受度验证报告",
        f"",
        f"> 生成时间：{now} ｜ 模拟用户数：{stats.n_personas} 位",
        f"",
        "---",
        "",
    ]

    # ── 模块一：概览 ──────────────────────────────────────────────────────────
    lines += [
        "## 一、产品与测试概览",
        "",
        f"**产品描述：** {p.description}",
        f"",
        f"**目标市场：** {p.target_market}",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 整体接受度均值 | **{stats.mean_score:.2f}** | {_score_to_label(stats.mean_score)} |",
        f"| 中位数 | {stats.median_score:.2f} | 一半用户高于此分数 |",
        f"| 最高分 | {stats.max_score:.2f} | 最理想用户的接受度 |",
        f"| 最低分 | {stats.min_score:.2f} | 最难转化用户的接受度 |",
        f"| 标准差 | {stats.std_score:.2f} | {'分化明显，存在清晰的用户细分' if stats.std_score > 0.2 else '分布较集中'} |",
        "",
        f"> **一句话结论：** {report.summary}",
        "",
        "---",
        "",
    ]

    # ── 模块二：接受度分布图 ──────────────────────────────────────────────────
    lines += [
        "## 二、每位用户的接受度分布",
        "",
        "绿色代表高于中位数的用户，红色代表低于中位数的用户，虚线为中位数。",
        "",
        img(chart_paths["distribution"]),
        "",
        "---",
        "",
    ]

    # ── 模块三：维度相关性 ────────────────────────────────────────────────────
    lines += [
        "## 三、哪类用户特征与接受度最相关？",
        "",
        "条形越长，影响越大；蓝色为正相关（该特征越高接受度越高），橙色为负相关。`*` 表示统计显著。",
        "",
        img(chart_paths["correlations"]),
        "",
        "**Top 3 驱动因素：**",
        "",
    ]
    for c in stats.correlations[:3]:
        direction = (
            "越高的用户，接受度越高"
            if c["correlation"] > 0
            else "越高的用户，接受度越低"
        )
        sig = "（显著）" if c["pvalue"] < 0.05 else "（趋势，样本量有限）"
        lines.append(
            f"- **{c['axis']}**（{c['correlation']:+.2f}）{sig}：{c['axis']}{direction}"
        )
    lines += ["", "---", ""]

    # ── 模块四：三大障碍 ──────────────────────────────────────────────────────
    lines += [
        "## 四、用户卡在哪里了？",
        "",
        "从三个维度分析用户整体的行动障碍：",
        "",
        img(chart_paths["barriers"]),
        "",
        "| 维度 | 均值 | 评级 | 解读 |",
        "|---|---|---|---|",
        f"| 认可这件事的价值 | {stats.mean_attitude:.2f} | {_score_to_label(stats.mean_attitude)} | {_barrier_interpretation('mean_attitude', stats.mean_attitude)} |",
        f"| 身边人是否支持 | {stats.mean_subjective_norm:.2f} | {_score_to_label(stats.mean_subjective_norm)} | {_barrier_interpretation('mean_subjective_norm', stats.mean_subjective_norm)} |",
        f"| 觉得自己能做到 | {stats.mean_perceived_control:.2f} | {_score_to_label(stats.mean_perceived_control)} | {_barrier_interpretation('mean_perceived_control', stats.mean_perceived_control)} |",
        "",
        "---",
        "",
    ]

    # ── 模块五：理想用户 vs 死区用户 ─────────────────────────────────────────
    median = stats.median_score
    high_evals = sorted(
        [e for e in matrix.evaluations if e.tpb_score.acceptance_score >= median],
        key=lambda e: e.tpb_score.acceptance_score,
        reverse=True,
    )
    low_evals = sorted(
        [e for e in matrix.evaluations if e.tpb_score.acceptance_score < median],
        key=lambda e: e.tpb_score.acceptance_score,
    )

    lines += [
        "## 五、最可能买单的用户 vs 几乎不会转化的用户",
        "",
        "### ✅ 理想用户画像",
        "",
        f"{report.ideal_persona_description}",
        "",
        "**代表性用户：**",
        "",
    ]
    for e in high_evals[:3]:
        lines.append(
            f"> 「{e.persona.description}」 — 接受度 **{e.tpb_score.acceptance_score:.2f}**"
        )
        lines.append("")

    lines += [
        "### ❌ 几乎不会转化的用户",
        "",
        f"{report.dead_zone_description}",
        "",
        "**代表性用户：**",
        "",
    ]
    for e in low_evals[:3]:
        lines.append(
            f"> 「{e.persona.description}」 — 接受度 **{e.tpb_score.acceptance_score:.2f}**"
        )
        lines.append("")

    lines += ["---", ""]

    # ── 模块六：分维度箱线图 ──────────────────────────────────────────────────
    lines += [
        "## 六、不同类型用户的接受度细分",
        "",
        "每张小图展示某个维度上，从 L1（最低）到 L4（最高）的用户，接受度分别是多少。",
        "如果从左到右分数明显递增，说明这个维度强烈影响接受度；如果差不多平，说明这个维度影响不大。",
        "",
        img(chart_paths["breakdown"]),
        "",
        "---",
        "",
    ]

    # ── 模块七：风险清单 ──────────────────────────────────────────────────────
    lines += [
        "## 七、需要关注的风险",
        "",
    ]
    risk_icons = ["⚠️", "⚠️", "⚠️", "⚠️", "⚠️"]
    for i, risk in enumerate(report.key_risks):
        lines.append(f"{risk_icons[i % len(risk_icons)]} {risk}")
        lines.append("")
    lines += ["---", ""]

    # ── 模块八：下一步建议 ────────────────────────────────────────────────────
    # 根据最弱的障碍维度自动生成建议方向
    barrier_values = {
        "attitude": stats.mean_attitude,
        "subjective_norm": stats.mean_subjective_norm,
        "perceived_control": stats.mean_perceived_control,
    }
    weakest = min(barrier_values, key=barrier_values.get)

    suggestions_map = {
        "attitude": (
            "价值主张验证",
            "整体态度分偏低，说明用户对'这件事值不值得做'还有疑问",
            "找 5~10 位目标用户做深度访谈，重点问：你现在是怎么解决这个问题的？如果有一个更好的方式，你会在意吗？",
        ),
        "subjective_norm": (
            "社会影响力验证",
            "主观规范分偏低，说明用户身边缺乏使用类似产品的氛围",
            "建立种子用户社群，测试口碑传播路径；或找意见领袖（KOL/KOC）背书，观察能否带动社交扩散",
        ),
        "perceived_control": (
            "上手门槛验证",
            "感知行为控制分偏低，说明很多用户觉得自己做不到",
            "做一次可用性测试，观察真实用户在哪一步卡住；重点优化首次使用的引导流程",
        ),
    }

    top_corr = stats.correlations[0] if stats.correlations else None

    lines += [
        "## 八、建议下一步验证什么",
        "",
        f"**优先建议 1：{suggestions_map[weakest][0]}**",
        f"",
        f"背景：{suggestions_map[weakest][1]}，是目前三个障碍中得分最低的（{barrier_values[weakest]:.2f}）。",
        f"",
        f"怎么做：{suggestions_map[weakest][2]}",
        "",
    ]

    if top_corr:
        direction_text = (
            "寻找痛点强的用户"
            if top_corr["correlation"] > 0
            else "了解依赖现有方案的用户为何难以迁移"
        )
        lines += [
            f"**优先建议 2：聚焦核心驱动维度——{top_corr['axis']}**",
            f"",
            f"背景：这是与接受度相关性最强的维度（{top_corr['correlation']:+.2f}），意味着早期用户获取策略应该优先{direction_text}。",
            f"",
            f"怎么做：在推广渠道和文案中明确针对「{top_corr['axis']}」高的用户群体，而非泛泛推广。",
            "",
        ]

    lines += [
        f"**优先建议 3：用真实用户数据校验模拟结论**",
        f"",
        f"背景：本报告基于 {stats.n_personas} 位合成用户的模拟数据，结论方向性可信，但需要真实数据验证。",
        f"",
        f"怎么做：选取本报告中理想用户画像对应的真实人群，做 5~10 次深度访谈，重点验证接受度最高的用户是否真的符合预测特征。",
        "",
        "---",
        "",
        "*本报告由 Persona Validation System 自动生成，数据来源为 LLM 合成用户模拟，仅供创业早期方向性参考，不替代真实用户研究。*",
    ]

    return "\n".join(lines)


# ── 主入口 ────────────────────────────────────────────────────────────────────


def generate_report(
    matrix: EvaluationMatrix,
    report: InsightReport,
    filename: str = "report.md",
) -> str:
    """
    一次性生成所有图表和 Markdown 报告，保存到 OUTPUT_DIR。
    返回报告文件路径。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats = build_stats(matrix)

    chart_paths = {
        "distribution": chart_distribution(stats, save_file=True),
        "correlations": chart_correlations(stats, save_file=True),
        "barriers": chart_tpb_barriers(stats, save_file=True),
        "breakdown": chart_axis_breakdown(stats, save_file=True),
    }

    md = render_report(matrix, report, stats, chart_paths)

    report_path = os.path.join(OUTPUT_DIR, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"报告已生成：{report_path}")
    for k, v in chart_paths.items():
        print(f"  图表 [{k}]：{v}")

    return report_path


# ── 测试 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models.schemas import (
        AxesOutput,
        DiversityAxis,
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

    mock_evaluations = []
    for i, (indices, desc, tpb) in enumerate(high_configs + low_configs):
        mock_evaluations.append(
            PersonaEvaluation(
                persona=PersonaProfile(
                    id=f"persona_{i:03d}",
                    vector=PersonaVector(
                        id=f"persona_{i:03d}",
                        axis_indices=indices,
                        axis_labels=[str(idx) for idx in indices],
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

    mock_report = InsightReport(
        top_correlations=[],
        ideal_persona_description="理想用户是25~40岁、有一定互联网使用习惯的城市钓鱼爱好者，他们意识到找鱼塘麻烦、愿意为便利付费，并且对新工具持开放态度。",
        dead_zone_description="几乎不会转化的用户是50岁以上、有固定鱼塘和塘主关系的老钓友，他们不觉得现有方式有问题，也不想学新App。",
        key_risks=[
            "【价值未被认可】50岁以上用户的态度分普遍偏低（均值<0.25），说明产品对这类用户的核心价值主张无法触达，建议将初期目标用户聚焦在35岁以下。",
            "【用户无法行动】感知行为控制整体均值偏低，部分用户担心自己学不会，建议大幅简化首次使用流程，减少注册和操作步骤。",
            "【缺乏社会支持】主观规范均值偏低，钓鱼社群中尚未形成使用App预约的氛围，建议优先打通头部钓鱼KOL渠道。",
        ],
        summary="产品对年轻城市钓鱼爱好者有清晰的吸引力，但对有固定鱼塘的老钓友几乎没有转化可能。目前最大的风险是用户分化严重，不适合泛渠道推广，建议聚焦35岁以下、痛点意识强的早期种子用户。",
    )

    generate_report(matrix, mock_report)

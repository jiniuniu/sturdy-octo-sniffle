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
from models.schemas import EvaluationMatrix, InsightReport

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
    scores: list[float]       # 每个 persona 的 acceptance_score
    persona_labels: list[str]  # 每个 persona 的简短标签（姓名/id）


def build_stats(matrix: EvaluationMatrix) -> ReportStats:
    scores = [e.tpb_score.acceptance_score for e in matrix.evaluations]
    np_scores = np.array(scores)

    # persona 简短标签：取 description 第一个逗号前的内容，fallback 到 id
    labels = []
    for e in matrix.evaluations:
        desc = e.persona.description
        label = desc.split("，")[0].split(",")[0].strip()
        labels.append(label if label else e.persona.id)

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
            float(np.mean([e.tpb_score.perceived_control for e in matrix.evaluations])), 3
        ),
        scores=scores,
        persona_labels=labels,
    )


# ── 图表工具 ───────────────────────────────────────────────────────────────────


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _fig_to_base64(fig: plt.Figure) -> str:
    return base64.b64encode(_fig_to_bytes(fig)).decode("utf-8")


def _fig_to_file(fig: plt.Figure, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    return path


# ── 图表一：接受度分布（横向点状条形图）─────────────────────────────────────


def chart_distribution(stats: ReportStats, save_file: bool = True) -> str | bytes:
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
    return _fig_to_bytes(fig)


# ── 图表二：五大业务维度信号矩阵 ─────────────────────────────────────────────


_SIGNAL_COLORS = {
    "验证通过": "#4CAF50",
    "需关注":   "#FF9800",
    "风险较高": "#F44336",
    "数据噪声": "#9E9E9E",
}

_TPB_LABELS = {
    "attitude":           "态度",
    "subjective_norm":    "主观规范",
    "perceived_control":  "感知控制",
}


def chart_dimension_matrix(report: "InsightReport", save_file: bool = True) -> str | bytes:
    """
    五大业务维度的信号矩阵：每行一个维度，展示信号、相关系数、关键 TPB 障碍。
    """
    from models.schemas import InsightReport  # 避免循环导入

    dims = report.dimension_insights
    n = len(dims)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    col_x = [0.05, 1.5, 2.6, 3.3]   # 维度名 | 信号 | 相关系数 | TPB障碍
    col_headers = ["业务维度", "信号", "相关系数", "关键障碍"]
    header_y = n - 0.05

    for x, h in zip(col_x, col_headers):
        ax.text(x, header_y, h, fontsize=9, fontweight="bold", va="bottom", color="#424242")

    ax.axhline(n - 0.25, color="#BDBDBD", linewidth=0.8, xmin=0, xmax=1)

    for i, di in enumerate(reversed(dims)):  # 从上到下显示
        y = i
        row_bg = "#F5F5F5" if i % 2 == 0 else COLOR_BG
        ax.add_patch(plt.Rectangle((0, y - 0.45), 4, 0.9, color=row_bg, zorder=0))

        # 维度名 + 业务问题
        ax.text(col_x[0], y + 0.1, di.dimension_name, fontsize=9, fontweight="bold", va="center")
        ax.text(col_x[0], y - 0.18, di.business_question, fontsize=7, va="center", color="#757575")

        # 信号徽章
        sig_color = _SIGNAL_COLORS.get(di.signal, "#9E9E9E")
        ax.add_patch(plt.FancyBboxPatch(
            (col_x[1] - 0.02, y - 0.22), 0.88, 0.44,
            boxstyle="round,pad=0.05", color=sig_color, alpha=0.15, zorder=1
        ))
        ax.text(col_x[1] + 0.42, y, di.signal, fontsize=8.5, va="center", ha="center",
                color=sig_color, fontweight="bold", zorder=2)

        # 相关系数
        corr_color = COLOR_POS if di.correlation >= 0 else COLOR_NEG
        sig_marker = " *" if di.pvalue < 0.05 else ""
        ax.text(col_x[2] + 0.3, y, f"{di.correlation:+.2f}{sig_marker}",
                fontsize=9, va="center", ha="center", color=corr_color, fontweight="bold")

        # 关键 TPB 障碍
        ax.text(col_x[3] + 0.3, y, _TPB_LABELS.get(di.key_tpb_barrier, di.key_tpb_barrier),
                fontsize=8.5, va="center", ha="center", color="#616161")

    ax.set_title("五大业务维度验证结果总览", fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()

    if save_file:
        return _fig_to_file(fig, "chart_dimension_matrix.png")
    return _fig_to_bytes(fig)


# ── 图表三：三大障碍进度条 ────────────────────────────────────────────────────


def chart_tpb_barriers(stats: ReportStats, save_file: bool = True) -> str | bytes:
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
    return _fig_to_bytes(fig)


# ── 图表四：五大维度 L1→L4 接受度梯度图 ─────────────────────────────────────


def chart_label_means(report: "InsightReport", save_file: bool = True) -> str | bytes:
    """
    2×3 子图（最后一格留空），每格展示一个业务维度的 L1→L4 接受度均值折线。
    梯度越陡峭，说明该维度对接受度的驱动越强。
    """
    dims = report.dimension_insights
    n_dims = len(dims)
    labels_x = ["L1\n(最低)", "L2", "L3", "L4\n(最高)"]
    gradient_colors = ["#F44336", "#FF9800", "#8BC34A", "#4CAF50"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(
        "各业务维度：L1→L4 用户的接受度均值（梯度越陡=该维度驱动越强）",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for idx in range(6):
        ax = axes.flat[idx]
        ax.set_facecolor(COLOR_BG)

        if idx >= n_dims:
            ax.axis("off")
            continue

        di = dims[idx]
        y_vals = [di.label_means.get(f"L{k+1}", float("nan")) for k in range(4)]
        valid = [(x, y) for x, y in enumerate(y_vals) if not (isinstance(y, float) and y != y)]
        x_valid = [v[0] for v in valid]
        y_valid = [v[1] for v in valid]

        # 折线
        ax.plot(x_valid, y_valid, color="#455A64", linewidth=1.5, zorder=2)
        # 各点彩色标记
        for xi, yi, c in zip(range(4), y_vals, gradient_colors):
            if not (isinstance(yi, float) and yi != yi):
                ax.scatter(xi, yi, color=c, s=70, zorder=3)
                ax.text(xi, yi + 0.04, f"{yi:.2f}", ha="center", fontsize=7.5, color="#424242")

        ax.set_xlim(-0.4, 3.4)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels_x, fontsize=7.5)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=7)
        ax.set_title(
            f"{di.dimension_name}（{di.signal}）",
            fontsize=9, fontweight="bold",
            color=_SIGNAL_COLORS.get(di.signal, "#424242"),
        )
        ax.axhline(0.5, color="#BDBDBD", linestyle="--", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_file:
        return _fig_to_file(fig, "chart_label_means.png")
    return _fig_to_bytes(fig)


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
    chart_paths 期望的 key：distribution / dimension_matrix / barriers / label_means
    """
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    p = matrix.product

    def img(key: str) -> str:
        path = chart_paths.get(key, "")
        return f"![chart]({path})" if path else ""

    lines = []

    # ── 封面 ──────────────────────────────────────────────────────────────────
    lines += [
        f"# {p.name} · 产品验证报告",
        "",
        f"> 生成时间：{now} ｜ 模拟用户数：{stats.n_personas} 位",
        "",
        "---",
        "",
    ]

    # ── 一、概览 ───────────────────────────────────────────────────────────────
    lines += [
        "## 一、产品与测试概览",
        "",
        f"**产品描述：** {p.description}",
        "",
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

    # ── 二：接受度分布 ─────────────────────────────────────────────────────────
    lines += [
        "## 二、每位用户的接受度分布",
        "",
        "绿色代表高于中位数的用户，红色代表低于中位数的用户，虚线为中位数。",
        "",
        img("distribution"),
        "",
        "---",
        "",
    ]

    # ── 三：五大业务维度总览 ───────────────────────────────────────────────────
    lines += [
        "## 三、五大业务维度验证结果",
        "",
        "每行展示一个业务维度的核心信号、相关系数（`*` 显著）和关键 TPB 障碍。",
        "",
        img("dimension_matrix"),
        "",
    ]
    # 每个维度的文字摘要
    for di in report.dimension_insights:
        sig_icon = {"验证通过": "✅", "需关注": "⚠️", "风险较高": "❌", "数据噪声": "❓"}.get(di.signal, "")
        lines += [
            f"### {sig_icon} {di.dimension_name}",
            "",
            f"> *{di.business_question}*",
            "",
            f"{di.finding}",
            "",
            f"**建议下一步：** {di.recommendation}",
            "",
        ]
    lines += ["---", ""]

    # ── 四：TPB 障碍 ───────────────────────────────────────────────────────────
    lines += [
        "## 四、用户行动的三大障碍",
        "",
        "从态度、社会影响、行动能力三个层面分析整体阻力：",
        "",
        img("barriers"),
        "",
        "| 维度 | 均值 | 评级 | 解读 |",
        "|---|---|---|---|",
        f"| 认可这件事的价值（态度） | {stats.mean_attitude:.2f} | {_score_to_label(stats.mean_attitude)} | {_barrier_interpretation('mean_attitude', stats.mean_attitude)} |",
        f"| 身边人是否支持（主观规范） | {stats.mean_subjective_norm:.2f} | {_score_to_label(stats.mean_subjective_norm)} | {_barrier_interpretation('mean_subjective_norm', stats.mean_subjective_norm)} |",
        f"| 觉得自己能做到（感知控制） | {stats.mean_perceived_control:.2f} | {_score_to_label(stats.mean_perceived_control)} | {_barrier_interpretation('mean_perceived_control', stats.mean_perceived_control)} |",
        "",
        "---",
        "",
    ]

    # ── 五：维度梯度图 ─────────────────────────────────────────────────────────
    lines += [
        "## 五、各维度用户类型的接受度梯度",
        "",
        "每张子图展示某维度下从 L1（最低）到 L4（最高）的用户接受度均值。"
        "梯度越陡，说明该维度对产品接受度的驱动越强。",
        "",
        img("label_means"),
        "",
        "---",
        "",
    ]

    # ── 六：理想用户 vs 死区 ───────────────────────────────────────────────────
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
        "## 六、最可能买单的用户 vs 几乎不会转化的用户",
        "",
        "### ✅ 理想用户画像",
        "",
        f"{report.ideal_persona_description}",
        "",
        "**代表性用户：**",
        "",
    ]
    for e in high_evals[:3]:
        lines.append(f"> 「{e.persona.description}」 — 接受度 **{e.tpb_score.acceptance_score:.2f}**")
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
        lines.append(f"> 「{e.persona.description}」 — 接受度 **{e.tpb_score.acceptance_score:.2f}**")
        lines.append("")
    lines += ["---", ""]

    # ── 七：风险清单 ───────────────────────────────────────────────────────────
    lines += ["## 七、需要关注的风险", ""]
    for risk in report.key_risks:
        lines.append(f"⚠️ {risk}")
        lines.append("")
    lines += ["---", ""]

    # ── 八：下一步建议 ─────────────────────────────────────────────────────────
    # 取信号最差的维度 + 最弱的 TPB 障碍两条建议
    signal_priority = {"风险较高": 0, "需关注": 1, "数据噪声": 2, "验证通过": 3}
    worst_dim = min(report.dimension_insights, key=lambda d: signal_priority.get(d.signal, 3))

    barrier_values = {
        "attitude": stats.mean_attitude,
        "subjective_norm": stats.mean_subjective_norm,
        "perceived_control": stats.mean_perceived_control,
    }
    weakest = min(barrier_values, key=barrier_values.get)
    suggestions_map = {
        "attitude": (
            "价值主张验证",
            "整体态度得分偏低，用户对「这件事值不值得做」仍有疑问",
            "找 5~10 位目标用户做深度访谈，聚焦：你现在怎么解决这个问题？如果有更好的方式，你会在意吗？",
        ),
        "subjective_norm": (
            "社会影响验证",
            "主观规范得分偏低，用户身边缺乏使用类似产品的氛围",
            "建立种子用户社群，测试口碑传播路径；或找意见领袖背书，观察能否带动社交扩散。",
        ),
        "perceived_control": (
            "上手门槛验证",
            "感知行为控制得分偏低，很多用户觉得自己做不到",
            "做一次可用性测试，观察真实用户在哪一步卡住；重点优化首次使用的引导流程。",
        ),
    }

    lines += [
        "## 八、建议下一步验证什么",
        "",
        f"**优先建议 1：{worst_dim.dimension_name}（{worst_dim.signal}）**",
        "",
        f"背景：这是五大维度中信号最弱的一项。",
        "",
        f"怎么做：{worst_dim.recommendation}",
        "",
        f"**优先建议 2：{suggestions_map[weakest][0]}**",
        "",
        f"背景：{suggestions_map[weakest][1]}，是目前三个 TPB 障碍中得分最低的（{barrier_values[weakest]:.2f}）。",
        "",
        f"怎么做：{suggestions_map[weakest][2]}",
        "",
        "**优先建议 3：用真实用户数据校验模拟结论**",
        "",
        f"背景：本报告基于 {stats.n_personas} 位合成用户，结论方向性可信，但需真实数据验证。",
        "",
        "怎么做：选取理想用户画像对应的真实人群，做 5~10 次深度访谈，重点验证接受度最高的用户是否真的符合预测特征。",
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
        "distribution":     chart_distribution(stats, save_file=True),
        "dimension_matrix": chart_dimension_matrix(report, save_file=True),
        "barriers":         chart_tpb_barriers(stats, save_file=True),
        "label_means":      chart_label_means(report, save_file=True),
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

    from models.schemas import DimensionInsight

    mock_dim_insights = [
        DimensionInsight(dimension_name="验证外部环境", business_question="现在进入市场时机成熟吗？", axis_names=["外部时机成熟度"], correlation=0.62, pvalue=0.01, label_means={"L1": 0.22, "L2": 0.40, "L3": 0.62, "L4": 0.81}, key_tpb_barrier="attitude", signal="验证通过", finding="L4用户接受度均值（0.81）是L1用户（0.22）的3.7倍，梯度非常清晰。市场时机感知强的用户已做好准备，但整体仍属少数。", recommendation="在年轻钓鱼社群（抖音/微信群）中做一次小型调研，验证「主动寻找新鱼塘」的用户比例是否在增长。"),
        DimensionInsight(dimension_name="需求真实性", business_question="这是真实痛点还是伪需求？", axis_names=["问题真实度（伪需求风险）"], correlation=0.58, pvalue=0.02, label_means={"L1": 0.19, "L2": 0.35, "L3": 0.61, "L4": 0.79}, key_tpb_barrier="attitude", signal="验证通过", finding="痛点强烈的用户（L3/L4）接受度显著高于感知不到痛点的用户，相关系数0.58（显著）。需求是真实存在的，但仅在特定人群中有足够强度。", recommendation="筛选出「经常因找不到好鱼塘而烦恼」的用户做深度访谈，验证现有替代方案的具体缺陷。"),
        DimensionInsight(dimension_name="迁移成本", business_question="用户从现有方案切换过来容易吗？", axis_names=["用户切换成本"], correlation=-0.31, pvalue=0.06, label_means={"L1": 0.65, "L2": 0.52, "L3": 0.41, "L4": 0.28}, key_tpb_barrier="perceived_control", signal="需关注", finding="切换成本越高的用户接受度越低（负相关-0.31），但统计显著性略不足。有固定鱼塘关系网络的老用户迁移意愿极低。", recommendation="设计「保留旧鱼塘联系方式」功能，降低迁移感知成本，测试是否能提升该群体的接受度。"),
        DimensionInsight(dimension_name="商业模式", business_question="用户愿意为这个价值付费吗？", axis_names=["变现可行性"], correlation=0.44, pvalue=0.04, label_means={"L1": 0.28, "L2": 0.45, "L3": 0.63, "L4": 0.77}, key_tpb_barrier="attitude", signal="需关注", finding="变现可行性高的用户接受度更高（0.44，显著），但整体均值偏低说明大多数用户付费意愿仍不确定。", recommendation="对接受度>0.7的用户做付费意愿测试（呈现具体价格方案），验证转化漏斗。"),
        DimensionInsight(dimension_name="竞争护城河", business_question="产品有没有持续的竞争优势？", axis_names=["竞争防御能力", "网络效应潜力"], correlation=0.18, pvalue=0.24, label_means={"L1": 0.45, "L2": 0.49, "L3": 0.51, "L4": 0.55}, key_tpb_barrier="subjective_norm", signal="数据噪声", finding="竞争防御和网络效应维度与接受度相关性极低（0.18，不显著），说明用户对竞争格局的感知在早期不构成决策障碍。这类判断需要更多样本才能验证。", recommendation="暂不将护城河纳入早期验证重点，等产品有稳定用户后再评估网络效应是否真实存在。"),
    ]

    mock_report = InsightReport(
        dimension_insights=mock_dim_insights,
        ideal_persona_description="理想用户是25~40岁、有一定互联网使用习惯的城市钓鱼爱好者，他们意识到找鱼塘麻烦、愿意为便利付费，并且对新工具持开放态度。",
        dead_zone_description="几乎不会转化的用户是50岁以上、有固定鱼塘和塘主关系的老钓友，他们不觉得现有方式有问题，也不想学新App。主要卡点在态度（不认可价值）和感知控制（不觉得自己能学会）两个层面。",
        key_risks=[
            "【价值未被认可】50岁以上用户的态度分普遍偏低（均值<0.25），说明产品对这类用户的核心价值主张无法触达，建议将初期目标用户聚焦在35岁以下。",
            "【用户无法行动】感知行为控制整体均值偏低，部分用户担心自己学不会，建议大幅简化首次使用流程，减少注册和操作步骤。",
            "【缺乏社会支持】主观规范均值偏低，钓鱼社群中尚未形成使用App预约的氛围，建议优先打通头部钓鱼KOL渠道。",
        ],
        summary="产品对年轻城市钓鱼爱好者有清晰的吸引力，但对有固定鱼塘的老钓友几乎没有转化可能。目前最大的风险是用户分化严重，不适合泛渠道推广，建议聚焦35岁以下、痛点意识强的早期种子用户。",
    )

    generate_report(matrix, mock_report)

from scipy.stats.qmc import Sobol


def sample_persona_positions(dimensions: list[dict], n: int) -> list[dict]:
    """
    Sobol 准随机采样 → segment 映射
    所有维度（含 hidden）都参与采样，保证地域和职业均匀覆盖。
    返回每个 persona 的 scores（数值，存 DB）和 segments（语义，传 LLM）。
    """
    n_dims = len(dimensions)
    sampler = Sobol(d=n_dims, scramble=True)
    raw = sampler.random(n)  # shape (n, n_dims)

    positions = []
    for i, row in enumerate(raw):
        scores: dict[str, float] = {}
        segments: dict[str, dict] = {}

        for j, dim in enumerate(dimensions):
            score = float(row[j])
            segs = dim["segments"]
            seg_index = min(int(score * len(segs)), len(segs) - 1)
            seg = segs[seg_index]

            scores[dim["dim_id"]] = score
            segments[dim["dim_id"]] = {
                "segment_id": seg["id"],
                "label": seg["label"],
                "description": seg["description"],
            }

        positions.append({
            "id": f"persona_{i + 1}",
            "scores": scores,
            "segments": segments,
        })

    return positions


def format_segments_text(position: dict, dimensions: list[dict]) -> str:
    """
    将 segments 格式化为自然语言传给 persona_chain。
    可见维度和 hidden 维度都传入，hidden 维度用于约束城市和职业。
    """
    visible_lines = []
    hidden_lines = []

    for dim in dimensions:
        dim_id = dim["dim_id"]
        if dim_id not in position["segments"]:
            continue
        seg = position["segments"][dim_id]
        line = f"{dim['name']}：{seg['label']} — {seg['description']}"

        if dim.get("source") == "hidden":
            hidden_lines.append(line)
        else:
            visible_lines.append(line)

    parts = []
    if visible_lines:
        parts.append("\n".join(visible_lines))
    if hidden_lines:
        parts.append("【地域与职业约束（必须严格遵守）】\n" + "\n".join(hidden_lines))

    return "\n\n".join(parts)

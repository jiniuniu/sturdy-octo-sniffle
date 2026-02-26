# pipeline/sampler.py
from models.schemas import AxesOutput, PersonaVector
from scipy.stats.qmc import Sobol


def sample_vectors(axes: AxesOutput, n: int = 16) -> list[PersonaVector]:
    """
    用 Sobol 序列在6维空间均匀采样，生成 n 个 PersonaVector。

    Sobol 序列是一种低差异序列（low-discrepancy sequence），相比纯随机采样，
    能更均匀地覆盖多维空间，确保极端组合（最理想/最难转化的用户）也被采到。

    n 建议为 2 的幂次（16、32、64），Sobol 序列在这些点数下均匀性最佳。
    """
    if n & (n - 1) != 0:
        import warnings

        warnings.warn(
            f"n={n} 不是2的幂次，Sobol序列均匀性会下降，建议使用 16、32 或 64。",
            UserWarning,
            stacklevel=2,
        )

    n_dims = len(axes.axes)  # 应为 6
    sampler = Sobol(d=n_dims, scramble=True)
    samples = sampler.random(n)  # shape: (n, 6)，值域 [0, 1)

    vectors = []
    for i, row in enumerate(samples):
        indices = [int(v * 4) for v in row]  # 映射到 0~3
        labels = [axes.axes[j].labels[idx] for j, idx in enumerate(indices)]
        vectors.append(
            PersonaVector(
                id=f"persona_{i:03d}",
                axis_indices=indices,
                axis_labels=labels,
            )
        )

    return vectors


if __name__ == "__main__":
    # 用 mock AxesOutput 测试，不依赖 LLM
    from models.schemas import DiversityAxis

    mock_axes = AxesOutput(
        axes=[
            DiversityAxis(
                name="痛点意识度",
                description="用户是否意识到找鱼塘是个值得解决的麻烦",
                labels=[
                    "从没觉得找鱼塘有什么问题",
                    "偶尔觉得麻烦但不在意",
                    "经常为找不到合适鱼塘烦恼",
                    "强烈感受到找鱼塘费时费力，一直想解决",
                ],
            ),
            DiversityAxis(
                name="现有方案依赖深度",
                description="用户对当前找鱼塘方式的依赖程度",
                labels=[
                    "没有固定方式，随便找",
                    "偶尔用微信群或朋友推荐",
                    "有固定的一两个鱼塘老板，习惯电话预约",
                    "高度依赖固定渠道，轻易不会换",
                ],
            ),
            DiversityAxis(
                name="付费心智",
                description="用户为找鱼塘类服务付费的意愿",
                labels=[
                    "认为这类服务不该收费",
                    "愿意付少量费用但很敏感",
                    "觉得便利值得付费，价格合理就行",
                    "愿意为高质量鱼情信息和预约便利支付溢价",
                ],
            ),
            DiversityAxis(
                name="新产品接受度",
                description="用户尝试新App解决钓鱼相关问题的积极性",
                labels=[
                    "对新App没兴趣，觉得麻烦",
                    "朋友强烈推荐才会试试",
                    "愿意尝试，但需要看到明显好处",
                    "主动寻找新工具，乐于尝鲜",
                ],
            ),
            DiversityAxis(
                name="外部约束强度",
                description="预算、家庭、时间等外部因素对钓鱼频率和消费的限制程度",
                labels=[
                    "几乎无约束，随时可以去钓鱼",
                    "偶尔受时间或预算限制",
                    "经常受家庭或工作影响，钓鱼计划容易泡汤",
                    "外部约束很强，钓鱼机会非常有限",
                ],
            ),
            DiversityAxis(
                name="问题归因方式",
                description="用户认为找鱼塘麻烦是自己要主动解决还是等别人提供解决方案",
                labels=[
                    "觉得这是正常的，不需要谁来解决",
                    "有时希望有更好的方式，但不会主动找",
                    "认为这是个可以被解决的问题，但在等合适的工具出现",
                    "强烈认为这个问题应该被解决，会主动寻找和推广好工具",
                ],
            ),
        ]
    )

    vectors = sample_vectors(mock_axes, n=16)

    print(f"采样数量：{len(vectors)}\n")
    for v in vectors:
        print(f"{v.id}  indices={v.axis_indices}")
        for axis, label in zip(mock_axes.axes, v.axis_labels):
            print(f"  {axis.name}：{label}")
        print()

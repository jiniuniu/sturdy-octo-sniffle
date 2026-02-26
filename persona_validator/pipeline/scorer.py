# pipeline/scorer.py
from config import settings
from models.schemas import Questionnaire, QuestionnaireResponse, TPBScore


def compute_tpb_score(
    response: QuestionnaireResponse,
    questionnaire: Questionnaire,
) -> TPBScore:
    """
    根据问卷回答计算 TPB 三维度分数和最终接受度分数。

    计算步骤：
    1. 反向计分处理：reverse_scored=True 的题目，score = 6 - score
    2. 按 TPB 维度分组求均值
    3. 标准化到 0~1：(mean - 1) / 4
    4. 三维度加权聚合为 acceptance_score
    """
    item_map = {item.id: item for item in questionnaire.items}

    dim_scores: dict[str, list[float]] = {
        "attitude": [],
        "subjective_norm": [],
        "perceived_control": [],
    }

    for item_id, resp in response.responses.items():
        item = item_map.get(item_id)
        if item is None:
            continue

        score = (6 - resp.score) if item.reverse_scored else resp.score
        normalized = (score - 1) / 4.0
        dim_scores[item.tpb_dimension].append(normalized)

    def safe_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    attitude = safe_mean(dim_scores["attitude"])
    subjective_norm = safe_mean(dim_scores["subjective_norm"])
    perceived_control = safe_mean(dim_scores["perceived_control"])

    acceptance_score = (
        settings.TPB_WEIGHT_ATTITUDE * attitude
        + settings.TPB_WEIGHT_SUBJECTIVE_NORM * subjective_norm
        + settings.TPB_WEIGHT_PERCEIVED_CONTROL * perceived_control
    )

    return TPBScore(
        attitude=round(attitude, 3),
        subjective_norm=round(subjective_norm, 3),
        perceived_control=round(perceived_control, 3),
        acceptance_score=round(acceptance_score, 3),
    )


if __name__ == "__main__":
    from models.schemas import QuestionnaireItem, SingleResponse

    # mock 问卷，和 responder.py 测试保持一致
    mock_questionnaire = Questionnaire(
        behavior_goal="下载App后查看附近鱼塘的实时余位并完成首次预约",
        items=[
            QuestionnaireItem(
                id="q01",
                tpb_dimension="attitude",
                question_text="我觉得用App预约鱼塘会让钓鱼体验变得更方便",
            ),
            QuestionnaireItem(
                id="q02",
                tpb_dimension="attitude",
                question_text="我觉得提前在网上查好鱼情再去是个明智的做法",
            ),
            QuestionnaireItem(
                id="q03",
                tpb_dimension="attitude",
                question_text="我觉得用App预约鱼塘是多此一举，没什么必要",
                reverse_scored=True,
            ),
            QuestionnaireItem(
                id="q04",
                tpb_dimension="attitude",
                question_text="如果能提前知道鱼塘的余位和鱼情，我觉得这很有价值",
            ),
            QuestionnaireItem(
                id="q05",
                tpb_dimension="subjective_norm",
                question_text="我身边的钓友大多支持用新方式来预约鱼塘",
            ),
            QuestionnaireItem(
                id="q06",
                tpb_dimension="subjective_norm",
                question_text="如果我开始用App预约鱼塘，身边的人会觉得我赶时髦没必要",
                reverse_scored=True,
            ),
            QuestionnaireItem(
                id="q07",
                tpb_dimension="subjective_norm",
                question_text="我的家人或朋友会鼓励我尝试更便捷的钓鱼预约方式",
            ),
            QuestionnaireItem(
                id="q08",
                tpb_dimension="subjective_norm",
                question_text="我认识的钓友里，已经有人在用类似的方式预约鱼塘",
            ),
            QuestionnaireItem(
                id="q09",
                tpb_dimension="perceived_control",
                question_text="我觉得自己能学会用App查鱼情和预约鱼塘，不会有什么困难",
            ),
            QuestionnaireItem(
                id="q10",
                tpb_dimension="perceived_control",
                question_text="对我来说，下载一个新App并完成首次预约是件容易的事",
            ),
            QuestionnaireItem(
                id="q11",
                tpb_dimension="perceived_control",
                question_text="我不太确定自己能搞清楚怎么用App预约鱼塘",
                reverse_scored=True,
            ),
            QuestionnaireItem(
                id="q12",
                tpb_dimension="perceived_control",
                question_text="就算遇到操作问题，我也有办法找到帮助或自己解决",
            ),
        ],
    )

    # 模拟老张的回答：认可价值但行动力弱
    # attitude 偏高（认为有价值），subjective_norm 和 perceived_control 偏低
    mock_response = QuestionnaireResponse(
        persona_id="persona_test",
        responses={
            "q01": SingleResponse(
                score=4, reasoning="确实方便，但我现在的方式也不麻烦"
            ),
            "q02": SingleResponse(score=4, reasoning="提前了解鱼情是有意义的"),
            "q03": SingleResponse(
                score=2, reasoning="我觉得还是有点必要的，但不是非用不可"
            ),  # 反向：score=2 → 实际=4
            "q04": SingleResponse(
                score=5, reasoning="提前知道余位太重要了，省得白跑一趟"
            ),
            "q05": SingleResponse(score=2, reasoning="我那帮老钓友都不用这玩意儿"),
            "q06": SingleResponse(
                score=4, reasoning="身边确实会有人觉得没必要搞这么复杂"
            ),  # 反向：score=4 → 实际=2
            "q07": SingleResponse(score=2, reasoning="家里人对我钓鱼的事不怎么上心"),
            "q08": SingleResponse(
                score=1, reasoning="我认识的人里没见过谁用App预约鱼塘"
            ),
            "q09": SingleResponse(
                score=2, reasoning="我对这种App不太熟，搞不好弄不明白"
            ),
            "q10": SingleResponse(score=2, reasoning="下载倒是会，但怎么用我不确定"),
            "q11": SingleResponse(
                score=4, reasoning="确实不太确定，这类软件我用得少"
            ),  # 反向：score=4 → 实际=2
            "q12": SingleResponse(
                score=2, reasoning="出了问题我一般就放弃了，不会去找帮助"
            ),
        },
    )

    score = compute_tpb_score(mock_response, mock_questionnaire)

    print("── TPB 分数 ──────────────────────────────")
    print(f"  态度          (attitude)          : {score.attitude:.3f}")
    print(f"  主观规范      (subjective_norm)    : {score.subjective_norm:.3f}")
    print(f"  感知行为控制  (perceived_control)  : {score.perceived_control:.3f}")
    print(f"  ─────────────────────────────────────")
    print(f"  接受度        (acceptance_score)   : {score.acceptance_score:.3f}")
    print()
    print(
        f"权重配置：attitude={settings.TPB_WEIGHT_ATTITUDE} | "
        f"subjective_norm={settings.TPB_WEIGHT_SUBJECTIVE_NORM} | "
        f"perceived_control={settings.TPB_WEIGHT_PERCEIVED_CONTROL}"
    )

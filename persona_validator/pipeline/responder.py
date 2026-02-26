# pipeline/responder.py
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import PersonaProfile, Questionnaire, QuestionnaireResponse
from utils.llm_client import get_llm

PROMPT = """\
你是一位擅长角色扮演的研究助手。你需要完全代入以下人物，以第一人称回答一份问卷。

## 重要规则

- 你就是这个人，用他/她的视角、性格和生活经验来回答
- 不要跳出角色，不要说"作为AI"或"根据人物描述"之类的话
- 回答要符合这个人的性格和生活状态，有时候答案会不那么"理性"或"积极"
- 如果这个人对某件事漠不关心，就如实打低分，不要为了"帮助研究"而打高分

## 你现在是这个人

{persona_description}

## 你被问及的行为

"{behavior_goal}"

## 问卷说明

以下每道题请给出 1~5 分：
- 1 = 完全不同意
- 2 = 不太同意
- 3 = 不确定 / 一般
- 4 = 比较同意
- 5 = 完全同意

请对每道题给出分数，并用一句话说明你（作为这个人物）这样打分的原因。
理由要真实反映这个人的想法，可以带有个人情绪或口语化表达。

## 题目

{questions}

## 输出要求

{format_instructions}
"""


def simulate_responses(
    persona: PersonaProfile,
    questionnaire: Questionnaire,
) -> QuestionnaireResponse:
    parser = PydanticOutputParser(pydantic_object=QuestionnaireResponse)

    questions_text = "\n".join(
        f"{item.id}. {item.question_text}" for item in questionnaire.items
    )

    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser

    result: QuestionnaireResponse = chain.invoke(
        {
            "persona_description": persona.description,
            "behavior_goal": questionnaire.behavior_goal,
            "questions": questions_text,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return result.model_copy(update={"persona_id": persona.id})


if __name__ == "__main__":
    from models.schemas import (
        AxesOutput,
        DiversityAxis,
        PersonaVector,
        QuestionnaireItem,
    )

    # mock persona：高痛点 + 高依赖现有方案 + 对新App没兴趣
    mock_persona = PersonaProfile(
        id="persona_test",
        vector=PersonaVector(
            id="persona_test",
            axis_indices=[3, 2, 1, 0, 1, 3],
            axis_labels=[
                "强烈感受到找鱼塘费时费力，一直想解决",
                "有固定的一两个鱼塘老板，习惯电话预约",
                "愿意付少量费用但很敏感",
                "对新App没兴趣，觉得麻烦",
                "偶尔受时间或预算限制",
                "强烈认为这个问题应该被解决，会主动寻找和推广好工具",
            ],
        ),
        description=(
            "老张，58岁，退休工厂主管，钓鱼二十多年。"
            "每周末固定去城郊同一个鱼塘，和塘主早就混熟了，提前一天打个电话就能订位。"
            "虽然偶尔抱怨鱼塘信息不透明，但对手机App一向没什么耐心，觉得捣鼓这些不如直接打电话省事。"
        ),
        behavior_goal="下载App后查看附近鱼塘的实时余位并完成首次预约",
    )

    # mock 问卷（节选6题，覆盖三个维度各2题含1道反向题）
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

    response = simulate_responses(mock_persona, mock_questionnaire)

    print(f"Persona：{response.persona_id}\n")
    for dim in ["attitude", "subjective_norm", "perceived_control"]:
        items = [i for i in mock_questionnaire.items if i.tpb_dimension == dim]
        print(f"── {dim} ──────────────────────")
        for item in items:
            resp = response.responses.get(item.id)
            if resp:
                reverse_tag = " [反向]" if item.reverse_scored else ""
                print(
                    f"  {item.id}{reverse_tag} [{resp.score}/5]：{item.question_text}"
                )
                print(f"    → {resp.reasoning}")
        print()

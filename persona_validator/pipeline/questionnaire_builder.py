# pipeline/questionnaire_builder.py
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ProductInfo, Questionnaire
from utils.llm_client import get_llm

PROMPT = """\
你是一位熟悉计划行为理论（Theory of Planned Behavior, TPB）的问卷设计专家，\
擅长为消费者行为研究设计高质量的测量工具。

## 任务背景

我们需要评估不同类型的用户对某产品的行为意向，具体是评估他们"完成某个目标行为"的可能性。
你需要根据产品信息和目标行为，设计一套 TPB 问卷。

## 产品信息

- 产品名称：{product_name}
- 产品描述：{product_description}

## 目标行为

{behavior_goal}

## TPB 三个维度说明

**attitude（态度）**
测量用户对"完成目标行为"这件事本身的评价——他们认为这样做是好是坏、有没有价值、值不值得。
注意：测的是对行为结果的评价，不是对产品功能的评价。

**subjective_norm（主观规范）**
测量用户感知到的社会压力——身边重要的人（家人、朋友、同事）是否支持、期望或鼓励他完成这个行为。
注意：测的是感知到的他人期望，不是用户自己的意愿。

**perceived_control（感知行为控制）**
测量用户对自己"能否完成这个行为"的信心——他们是否觉得自己有能力、有条件、有机会完成这个行为。
注意：测的是自我效能感和外部条件感知，不是意愿。

## 问卷设计要求

请设计 **12 道题目**，每个 TPB 维度各 4 题，具体要求如下：

**题目内容要求：**
- 所有题目使用第一人称（"我"），口语化，读起来自然
- 题目针对"完成目标行为"这件事本身，不能直接提及产品名称
- 每道题聚焦一个单一概念，避免在一题中混合两个问题
- 题目要有区分度，同一维度的4道题应从不同角度测量，避免重复

**反向计分题要求：**
- 每个维度至少包含 1 道反向计分题（reverse_scored: true）
- 反向计分题的表述应该是负向的，即得分越高代表该维度越低
- 例如：attitude 的反向题可以是"我觉得用这种方式预约鱼塘是在浪费时间"

**题目编号：**
- 使用 q01 ~ q12 编号
- attitude 对应 q01~q04，subjective_norm 对应 q05~q08，perceived_control 对应 q09~q12

## 输出要求

{format_instructions}
"""


def build_questionnaire(product: ProductInfo, behavior_goal: str) -> Questionnaire:
    parser = PydanticOutputParser(pydantic_object=Questionnaire)

    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser

    return chain.invoke(
        {
            "product_name": product.name,
            "product_description": product.description,
            "behavior_goal": behavior_goal,
            "format_instructions": parser.get_format_instructions(),
        }
    )


if __name__ == "__main__":
    product = ProductInfo(
        name="钓鱼塘连接App",
        description="一款连接钓鱼爱好者和鱼塘的App，用户可以查看实时鱼情、余位，提前预约鱼塘。",
        target_market="中国城市钓鱼爱好者，25~60岁",
    )

    behavior_goal = "下载App后查看附近鱼塘的实时余位并完成首次预约"

    questionnaire = build_questionnaire(product, behavior_goal)

    print(f"行为目标：{questionnaire.behavior_goal}\n")
    print(f"共 {len(questionnaire.items)} 道题\n")

    for dim in ["attitude", "subjective_norm", "perceived_control"]:
        items = [item for item in questionnaire.items if item.tpb_dimension == dim]
        print(f"── {dim} ({len(items)}题) ──────────────────────")
        for item in items:
            reverse_tag = " [反向]" if item.reverse_scored else ""
            print(f"  {item.id}{reverse_tag}：{item.question_text}")
        print()

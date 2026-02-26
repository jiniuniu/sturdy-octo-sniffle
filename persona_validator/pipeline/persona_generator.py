# pipeline/persona_generator.py
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import AxesOutput, PersonaProfile, PersonaVector, ProductInfo
from utils.llm_client import get_llm

PROMPT = """\
你是一位经验丰富的用户研究员，擅长将结构化的用户特征转化为生动、真实的人物描述。

## 任务背景

我们正在为一个创业产品构建虚拟用户，用于模拟不同类型用户对产品的接受程度。
你的任务是根据给定的用户特征，创作出一个真实存在感强的人物描述。

## 产品信息

- 产品名称：{product_name}
- 产品描述：{product_description}
- 目标市场：{target_market}

## 该用户在各维度上的特征

{axis_summary}

## 你的任务

**1. 人物描述（description）**
用2~3句话描述这个真实的人。需要包含：
- 基本信息：姓名、年龄、职业
- 与产品场景相关的生活状态或行为习惯
- 能体现上述维度特征的具体细节，但不要逐条罗列维度，而是自然融入描述中

要求：
- 描述要让人觉得这是一个真实存在的人，而不是一份用户画像报告
- 使用具体细节而非抽象概括，如"他每周末都要开车四十分钟去固定鱼塘"比"他有固定鱼塘"更好
- 不要在描述中提及"维度"、"标签"、"L1~L4"等系统术语
- 不要提及产品名称

**2. 行为目标（behavior_goal）**
推导出该用户对于此产品最核心的行为目标，即：如果这个人要使用这个产品，他/她最关键的第一步行为是什么。
- 写成一句具体的行为描述，如"下载App后查看附近鱼塘的实时余位并完成首次预约"
- 行为目标要符合该人物的性格和使用习惯
- 不要写得过于宏观（如"成为忠实用户"），也不要过于微观（如"点击注册按钮"）

## 输出要求

{format_instructions}
"""


def generate_persona(
    vector: PersonaVector,
    axes: AxesOutput,
    product: ProductInfo,
) -> PersonaProfile:
    parser = PydanticOutputParser(pydantic_object=PersonaProfile)

    axis_summary = "\n".join(
        f"- {axes.axes[i].name}（{axes.axes[i].description}）：{label}"
        for i, label in enumerate(vector.axis_labels)
    )

    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser

    result: PersonaProfile = chain.invoke(
        {
            "product_name": product.name,
            "product_description": product.description,
            "target_market": product.target_market,
            "axis_summary": axis_summary,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return result.model_copy(update={"id": vector.id, "vector": vector})


if __name__ == "__main__":
    from models.schemas import DiversityAxis

    product = ProductInfo(
        name="钓鱼塘连接App",
        description="一款连接钓鱼爱好者和鱼塘的App，用户可以查看实时鱼情、余位，提前预约鱼塘。",
        target_market="中国城市钓鱼爱好者，25~60岁",
    )

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

    # 手动构造一个向量，对应：高痛点 + 高依赖 + 中等付费 + 低新品接受度 + 中等约束 + 主动归因
    test_vector = PersonaVector(
        id="persona_test",
        axis_indices=[3, 2, 1, 0, 1, 3],
        axis_labels=[
            mock_axes.axes[i].labels[idx] for i, idx in enumerate([3, 2, 1, 0, 1, 3])
        ],
    )

    print("向量：")
    for axis, label in zip(mock_axes.axes, test_vector.axis_labels):
        print(f"  {axis.name}：{label}")
    print()

    profile = generate_persona(test_vector, mock_axes, product)

    print(f"ID：{profile.id}")
    print(f"描述：{profile.description}")
    print(f"行为目标：{profile.behavior_goal}")

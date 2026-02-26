from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import PREDEFINED_AXIS_NAMES, AxesOutput, ProductInfo
from utils.llm_client import get_llm

PROMPT = """\
你是一位专注于消费者行为研究的分析师，擅长将抽象的用户分析维度转化为具体、可识别的用户特征。

## 任务背景

我们正在为一个创业产品构建多样化的虚拟用户群体，目的是系统性地评估不同类型用户对该产品的接受程度。
为此，需要将6个通用的用户多样性维度，结合具体产品场景进行具体化。

## 产品信息

- 产品名称：{product_name}
- 产品描述：{product_description}
- 目标市场：{target_market}

## 需要具体化的6个维度

{axis_names}

## 你的任务

对上述每个维度，完成以下两项工作：

**1. 具体化维度说明（description）**
将通用维度定义结合本产品场景改写为1句具体说明。
要求：
- 紧扣产品的使用场景和目标市场
- 说明这个维度在该产品场景下具体指什么
- 语言简洁，1句话以内
- 示例：维度"痛点意识度"对于一款找停车位的App，可具体化为"用户是否意识到找停车位耗时是个值得解决的问题"

**2. 生成语义标签（labels）**
为每个维度生成从低到高排列的4个语义标签（L1~L4），代表该维度上4种典型的用户状态。
要求：
- L1 代表该维度最低程度（如：完全没有痛点意识）
- L4 代表该维度最高程度（如：强烈意识到痛点且主动寻求解决方案）
- 每个标签是1句话的用户状态描述，具体、口语化，像是在描述真实存在的人
- 标签之间要有明显梯度，不要过于相似
- 使用第三人称描述，如"他/她从不…"、"偶尔会…"、"经常…"、"强烈感受到…"
- 不要使用"L1/L2/L3/L4"或"低/中/高"等抽象标记作为标签内容本身
- 不要在标签中提及产品名称

## 输出要求

{format_instructions}
"""


def build_axes(product: ProductInfo) -> AxesOutput:
    parser = PydanticOutputParser(pydantic_object=AxesOutput)
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm() | parser
    return chain.invoke(
        {
            "product_name": product.name,
            "product_description": product.description,
            "target_market": product.target_market,
            "axis_names": "\n".join(f"- {n}" for n in PREDEFINED_AXIS_NAMES),
            "format_instructions": parser.get_format_instructions(),
        }
    )


if __name__ == "__main__":
    product = ProductInfo(
        name="钓鱼塘连接App",
        description="一款连接钓鱼爱好者和鱼塘的App，用户可以查看实时鱼情、余位，提前预约鱼塘。",
        target_market="中国城市钓鱼爱好者，25~60岁",
    )

    axes = build_axes(product)
    print(axes)

from langchain_core.messages import HumanMessage
from chains.client import get_vision_llm


async def describe_image(base64_data: str, mime_type: str, campaign_description: str) -> str:
    llm = get_vision_llm()
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            },
            {
                "type": "text",
                "text": f"""你正在分析一个品牌营销活动的视觉素材。
活动描述为："{campaign_description}"

请从品牌风险分析的角度详细描述这张图片：
- 视觉构成与风格（摄影、插画、配色方案）
- 画面中的人物（年龄、性别、外貌、服装、动作）
- 出现的文化符号、意象或标志性元素
- 传递的情绪基调
- 可见的文字、口号或 logo
- 视觉内容中潜在的文化敏感点或歧义

请保持客观具体。此描述将用于识别消费者反应模拟的风险维度。""",
            },
        ]
    )
    response = await llm.ainvoke([message])
    return str(response.content)

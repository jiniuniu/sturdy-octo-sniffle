"""
人群类型数据模型
"""

from pydantic import BaseModel, Field


class PersonaType(BaseModel):
    """人群类型"""

    type_name: str = Field(description="人群名称")

    description: str = Field(
        description="人群完整描述，包含特征、态度、敏感点、行为倾向等"
    )

    proportion: float = Field(ge=0, le=100, description="在目标受众中的占比（百分比）")

    class Config:
        json_schema_extra = {
            "example": {
                "type_name": "品牌忠粉",
                "description": """
                    对品牌有较强认同感的核心用户群。通常是较早期的消费者，
                    有过多次愉快的消费体验。对品牌有感情基础，愿意为品牌说话，
                    但也期待品牌持续进步。面对负面信息时倾向于先观望或为品牌辩护。
                """,
                "proportion": 15,
            }
        }


class PersonaTypeList(BaseModel):
    """人群类型列表（用于 LLM 结构化输出）"""

    personas: list[PersonaType] = Field(description="人群类型列表")

"""
品牌上下文数据模型
"""

from pydantic import BaseModel, Field


class BrandContext(BaseModel):
    """品牌背景配置"""

    brand_name: str = Field(description="品牌名称")

    description: str = Field(description="品牌背景描述，包含行业、调性、历史事件等")

    target_audience: str = Field(description="目标受众描述")

    sensitivity_topics: list[str] = Field(
        default_factory=list, description="需要关注的敏感议题列表"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "brand_name": "XX饮料",
                "description": """
                    XX饮料是一家主打健康概念的饮品品牌，目标客群为18-35岁注重生活品质的年轻人。
                    品牌调性偏年轻、活力、健康。
                    历史上曾在2024年因代糖成分问题引发争议，后官方澄清但部分消费者仍有疑虑。
                """,
                "target_audience": """
                    主要是一二线城市的年轻白领和大学生，关注健康、热爱社交媒体，
                    对食品安全和成分比较敏感，价格敏感度中等。
                """,
                "sensitivity_topics": ["食品安全", "添加剂/代糖", "价格", "虚假宣传"],
            }
        }

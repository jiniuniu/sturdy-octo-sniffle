"""
人群类型生成器
"""

from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext, PersonaType
from sentisim.models.persona import PersonaTypeList


class PersonaGenerator:
    """人群类型生成器"""

    def __init__(self, llm: SentiSimLLM):
        self.llm = llm

    async def generate(
        self,
        brand_context: BrandContext,
        count: int = 6,
    ) -> list[PersonaType]:
        """
        生成人群类型列表

        Args:
            brand_context: 品牌上下文
            count: 生成数量（6-8）

        Returns:
            人群类型列表
        """
        prompt = f"""
【品牌背景】
{brand_context.description}

【目标受众】
{brand_context.target_audience}

【敏感议题】
{', '.join(brand_context.sensitivity_topics)}

请为该品牌生成 {count} 种典型的受众人群分类。

要求：
1. 覆盖对品牌态度从正面到负面的不同人群
2. 各人群 proportion 之和为 100
3. 每个人群用一段话完整描述其特征、态度、敏感点、行为倾向（50-100字）
4. 不需要考虑影响力层级（KOL/普通用户等），这会在后续根据网络结构分配

人群示例类型（仅供参考，请根据品牌特点调整）：
- 品牌忠粉：对品牌有感情，愿意维护
- 理性消费者：看重性价比，不盲从
- 成分党：关注产品成分和健康
- 价格敏感者：对价格变动敏感
- 潜在批评者：对品牌有负面印象
- 路人观望者：没有明确态度
"""

        result = await self.llm.generate_structured(prompt, PersonaTypeList)

        # # 验证比例之和
        # total = sum(p.proportion for p in result.personas)
        # if abs(total - 100) > 5:  # 允许5%误差
        #     # 归一化
        #     for p in result.personas:
        #         p.proportion = round(p.proportion * 100 / total, 1)

        return result.personas

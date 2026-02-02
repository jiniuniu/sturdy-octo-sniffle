"""
LLM 客户端测试

运行方式:
    # 先设置环境变量
    export OPENROUTER_API_KEY="your-api-key"

    # 运行测试
    cd /path/to/sentisim
    python -m pytest tests/test_llm.py -v

    # 或直接运行
    python tests/test_llm.py
"""

import asyncio

from pydantic import BaseModel, Field
from sentisim.llm import SentiSimLLM
from sentisim.models import ActionType, PersonaType, UserResponse


# 用于测试的简单模型
class SimpleResponse(BaseModel):
    """简单测试响应"""

    answer: str = Field(description="回答")
    confidence: float = Field(description="置信度，0-1之间")


async def test_persona_generation():
    """测试人群类型生成（模拟真实场景）"""
    print("\n=== 测试人群类型生成 ===")

    llm = SentiSimLLM()

    # 模拟生成单个人群类型
    prompt = """
    请为一个饮料品牌生成一个典型的受众人群类型。
    
    要求：
    - type_name: 人群名称（如"品牌忠粉"、"价格敏感者"等）
    - description: 完整描述其特征、态度、敏感点、行为倾向（50-100字）
    - proportion: 在目标受众中的占比（数字，如15表示15%）
    """

    result = await llm.generate_structured(prompt, PersonaType)

    print(f"生成的人群类型:")
    print(f"  - 名称: {result.type_name}")
    print(f"  - 描述: {result.description[:100]}...")
    print(f"  - 占比: {result.proportion}%")

    assert isinstance(result, PersonaType)
    assert len(result.type_name) > 0
    assert 0 <= result.proportion <= 100
    print("✓ 测试通过")


async def test_user_response_generation():
    """测试用户反应生成（核心功能）"""
    print("\n=== 测试用户反应生成 ===")

    llm = SentiSimLLM()

    prompt = """
    【你的身份】
    25岁女生，杭州互联网公司运营，日常靠奶茶续命，最近在尝试戒糖。
    对食品添加剂比较敏感，价格敏感度中等。
    
    【你对该品牌的记忆】
    - 去年代糖事件时有点担心，后来看官方澄清继续买了
    
    【你刚刚看到的内容】
    发布者：品牌官方账号
    内容："【重磅升级】全新XX饮料，0糖0卡，健康新选择！现在购买享8折优惠！"
    
    请完全代入这个用户的身份，描述你的反应：
    - first_reaction: 第一反应（一句话）
    - emotion: 情绪感受
    - impression_change: 对品牌印象的变化（如果没变化就说"没有变化"）
    - action: 你会怎么做（ignore/like/forward/forward_comment）
    - content: 如果选择 forward_comment，写什么评论（否则留空）
    """

    result = await llm.generate_structured(prompt, UserResponse)

    print(f"用户反应:")
    print(f"  - 第一反应: {result.first_reaction}")
    print(f"  - 情绪: {result.emotion}")
    print(f"  - 印象变化: {result.impression_change}")
    print(f"  - 行动: {result.action.value}")
    print(f"  - 内容: {result.content or '(无)'}")
    print(f"  - 是否传播: {result.will_spread}")
    print(f"  - 是否负面: {result.is_negative}")

    assert isinstance(result, UserResponse)
    assert isinstance(result.action, ActionType)
    print("✓ 测试通过")


async def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("SentiSim LLM 客户端测试")
    print("=" * 50)

    try:
        await test_persona_generation()
        await test_user_response_generation()

        print("\n" + "=" * 50)
        print("所有测试通过 ✓")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

"""
阶段3测试：模拟引擎

运行方式:
    export OPENROUTER_API_KEY="your-api-key"
    cd sentisim
    python tests/test_simulation.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from sentisim.generators import MemoryGenerator, PersonaGenerator, UserGenerator
from sentisim.llm import SentiSimLLM
from sentisim.models import BrandContext, InfluenceLevel, Post, PostType, User
from sentisim.network import InfluenceAssigner, NetworkBuilder
from sentisim.simulation import MemoryManager, ResponseSimulator, SimulationEngine

# 测试用的品牌上下文
TEST_BRAND = BrandContext(
    brand_name="XX饮料",
    description="""
        XX饮料是一家主打健康概念的饮品品牌，目标客群为18-35岁注重生活品质的年轻人。
        品牌调性偏年轻、活力、健康。
        历史上曾在2024年因代糖成分问题引发争议，后官方澄清但部分消费者仍有疑虑。
        去年有过一次涨价，引发部分老用户不满。
    """,
    target_audience="""
        主要是一二线城市的年轻白领和大学生，关注健康、热爱社交媒体，
        对食品安全和成分比较敏感，价格敏感度中等，容易被KOL种草也容易被负面新闻劝退。
    """,
    sensitivity_topics=["食品安全", "添加剂/代糖", "价格", "虚假宣传"],
)

TEST_CONTENT = "【重磅升级】全新XX饮料，0糖0卡，健康新选择！现在购买享8折优惠！"


async def test_response_simulator():
    """测试用户反应模拟"""
    print("\n=== 测试用户反应模拟 ===")

    llm = SentiSimLLM()
    simulator = ResponseSimulator(llm)

    # 创建测试用户
    test_user = User(
        user_id="test_user_001",
        persona_type="成分党",
        profile="""
            28岁女性，上海互联网公司产品经理。对食品成分非常敏感，
            会仔细研究配料表。之前因为代糖事件对XX饮料有些不信任，
            但还是会偶尔购买。表达风格理性，喜欢用数据说话。
        """,
        influence_level=InfluenceLevel.ACTIVE,
        memory=["去年代糖事件让我对这个品牌有点失望"],
        follower_ids=[],
    )

    # 创建测试帖子
    test_post = Post(
        post_id="post_001",
        author_id="brand_official",
        content=TEST_CONTENT,
        post_type=PostType.BRAND_ORIGINAL,
        timestamp=0,
    )

    # 模拟反应
    response = await simulator.simulate(
        user=test_user,
        post=test_post,
        is_brand_post=True,
    )

    print(f"用户: {test_user.persona_type}")
    print(f"帖子: {test_post.content[:50]}...")
    print(f"反应:")
    print(f"  - 第一反应: {response.first_reaction}")
    print(f"  - 情绪: {response.emotion}")
    print(f"  - 印象变化: {response.impression_change}")
    print(f"  - 行动: {response.action.value}")
    print(f"  - 内容: {response.content or '(无)'}")
    print(f"  - 是否传播: {response.will_spread}")
    print(f"  - 是否负面: {response.is_negative}")

    print("✓ 测试通过")
    return response


def test_memory_manager():
    """测试记忆管理器"""
    print("\n=== 测试记忆管理器 ===")

    from sentisim.models import ActionType, UserResponse

    manager = MemoryManager(max_memory_size=5)

    # 创建测试用户
    user = User(
        user_id="test_user",
        persona_type="测试用户",
        profile="测试用户画像",
        influence_level=InfluenceLevel.NORMAL,
        memory=["初始记忆1", "初始记忆2"],
        follower_ids=[],
    )

    # 测试1：印象没变化，不应更新
    response1 = UserResponse(
        first_reaction="还行吧",
        emotion="无感",
        impression_change="没有变化",
        action=ActionType.IGNORE,
        content=None,
    )
    updated = manager.update(user, response1)
    print(f"印象无变化时: 更新={updated}, 记忆数={len(user.memory)}")
    assert not updated
    assert len(user.memory) == 2

    # 测试2：印象有变化，应更新
    response2 = UserResponse(
        first_reaction="又是0糖0卡...",
        emotion="怀疑",
        impression_change="感觉品牌还是在回避代糖的具体成分问题",
        action=ActionType.IGNORE,
        content=None,
    )
    updated = manager.update(user, response2)
    print(f"印象有变化时: 更新={updated}, 记忆数={len(user.memory)}")
    assert updated
    assert len(user.memory) == 3
    assert "代糖" in user.memory[-1]

    # 测试3：超过最大记忆数
    for i in range(5):
        response = UserResponse(
            first_reaction="测试",
            emotion="测试",
            impression_change=f"新印象{i}",
            action=ActionType.IGNORE,
            content=None,
        )
        manager.update(user, response)

    print(f"多次更新后: 记忆数={len(user.memory)}")
    assert len(user.memory) <= 5

    print("✓ 测试通过")


async def test_simulation_engine():
    """测试完整模拟引擎（小规模）"""
    print("\n=== 测试模拟引擎 ===")

    llm = SentiSimLLM()

    # 1. 构建小规模网络
    print("构建网络拓扑...")
    builder = NetworkBuilder()
    topology = builder.build(user_count=20)  # 小规模测试

    # 2. 分配影响力
    print("分配影响力...")
    assigner = InfluenceAssigner()
    influence_levels = assigner.assign(topology)

    # 3. 生成人群类型
    print("生成人群类型...")
    persona_gen = PersonaGenerator(llm)
    personas = await persona_gen.generate(TEST_BRAND, count=4)  # 少量人群
    print(f"  生成了 {len(personas)} 种人群")

    # 4. 生成用户
    print("生成用户画像...")
    user_gen = UserGenerator(llm)
    users = await user_gen.generate_users(
        user_ids=topology.user_ids,
        persona_types=personas,
        influence_levels=influence_levels,
        follower_counts=topology.follower_counts,
        follower_map=topology.follower_map,
        max_concurrent=5,
    )
    print(f"  生成了 {len(users)} 个用户")

    # 5. 初始化记忆
    print("初始化记忆...")
    memory_gen = MemoryGenerator(llm)
    await memory_gen.initialize_memories(users, TEST_BRAND, max_concurrent=5)

    # 6. 构建用户字典
    users_dict = {u.user_id: u for u in users}

    # 7. 运行模拟
    print("运行模拟...")
    engine = SimulationEngine(
        llm=llm,
        users=users_dict,
        brand_account_id=topology.brand_account_id,
        max_concurrent=5,
    )

    def on_step(step, wave_size, new_posts):
        print(f"  Step {step}: 处理 {wave_size} 条反应, 产生 {new_posts} 条新帖子")

    result = await engine.run(
        test_content=TEST_CONTENT,
        max_steps=3,  # 小规模测试只跑3步
        on_step_complete=on_step,
    )

    # 8. 输出结果
    print(f"\n模拟结果:")
    print(f"  - 总触达: {result.total_reach}")
    print(f"  - 运行步数: {result.steps_run}")
    print(f"  - 总帖子数: {len(result.posts)}")
    print(f"  - 传播帖子数: {len(result.get_spreading_posts())}")

    print(f"\n行动分布:")
    for action, count in result.get_action_counts().items():
        print(f"  - {action}: {count}")

    print(f"\n情绪分布:")
    for emotion, count in sorted(
        result.get_emotion_counts().items(), key=lambda x: -x[1]
    )[:5]:
        print(f"  - {emotion}: {count}")

    negative = result.get_negative_responses()
    print(f"\n负面反应: {len(negative)} 条")
    if negative:
        for r in negative[:3]:
            print(f"  - [{r.user_persona}] {r.response.first_reaction}")

    print("✓ 测试通过")


async def run_all_tests():
    """运行所有阶段3测试"""
    print("=" * 50)
    print("SentiSim 阶段3测试：模拟引擎")
    print("=" * 50)

    try:
        # 1. 单个用户反应模拟
        await test_response_simulator()

        # 2. 记忆管理器（不需要 LLM）
        test_memory_manager()

        # 3. 完整模拟引擎（小规模）
        await test_simulation_engine()

        print("\n" + "=" * 50)
        print("所有测试通过 ✓")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())

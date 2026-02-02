"""
阶段2测试：数据生成

运行方式:
    export OPENROUTER_API_KEY="your-api-key"
    cd sentisim
    python tests/test_generators.py
"""

import asyncio
import sys

sys.path.insert(0, ".")

from sentisim.generators import PersonaGenerator, UserGenerator
from sentisim.llm import SentiSimLLM
from sentisim.network import InfluenceAssigner, NetworkBuilder, NetworkTopology
from tests.sample_inputs import DEFAULT_BRAND


async def test_network_topology():
    """测试网络拓扑构建"""
    print("\n=== 测试网络拓扑构建 ===")

    builder = NetworkBuilder()
    topology = builder.build(user_count=50)

    print(f"用户数量: {len(topology.user_ids)}")
    print(f"品牌账号: {topology.brand_account_id}")
    print(f"总边数: {topology.graph.number_of_edges()}")

    # 品牌账号的粉丝数
    brand_followers = topology.get_follower_count(topology.brand_account_id)
    print(f"品牌账号粉丝数: {brand_followers}")

    # 查看粉丝数最高的用户
    top_users = sorted(
        topology.user_ids, key=lambda x: topology.get_follower_count(x), reverse=True
    )[:5]
    print("粉丝数最高的5个用户:")
    for uid in top_users:
        print(f"  {uid}: {topology.get_follower_count(uid)} 粉丝")

    assert len(topology.user_ids) == 50
    assert topology.brand_account_id == "brand_official"
    print("✓ 测试通过")

    return topology


def test_influence_assignment(topology: NetworkTopology):
    """测试影响力分配"""
    print("\n=== 测试影响力分配 ===")

    assigner = InfluenceAssigner()
    influence_levels = assigner.assign(topology)

    stats = assigner.get_stats(influence_levels)
    print(f"影响力分布:")
    for level, count in stats["counts"].items():
        pct = stats["percentages"][level]
        print(f"  {level}: {count} ({pct}%)")

    assert len(influence_levels) == len(topology.user_ids)
    print("✓ 测试通过")

    return influence_levels


async def test_persona_generation():
    """测试人群类型生成"""
    print("\n=== 测试人群类型生成 ===")

    llm = SentiSimLLM()
    generator = PersonaGenerator(llm)

    personas = await generator.generate(DEFAULT_BRAND, count=6)

    print(f"生成了 {len(personas)} 种人群类型:")
    total_proportion = 0
    for p in personas:
        print(f"  [{p.proportion}%] {p.type_name}")
        print(f"       {p.description[:60]}...")
        total_proportion += p.proportion

    print(f"比例总和: {total_proportion}%")

    assert len(personas) >= 4
    assert 95 <= total_proportion <= 105  # 允许一定误差
    print("✓ 测试通过")

    return personas


async def test_user_generation(topology: NetworkTopology, influence_levels, personas):
    """测试用户画像生成（小规模，包含初始记忆）"""
    print("\n=== 测试用户画像生成（含初始记忆）===")

    llm = SentiSimLLM()
    generator = UserGenerator(llm)

    # 只测试前5个用户
    test_user_ids = topology.user_ids[:5]

    users = await generator.generate_users(
        user_ids=test_user_ids,
        persona_types=personas,
        influence_levels={uid: influence_levels[uid] for uid in test_user_ids},
        follower_counts={
            uid: topology.get_follower_count(uid) for uid in test_user_ids
        },
        follower_map={uid: topology.get_followers(uid) for uid in test_user_ids},
        brand_context=DEFAULT_BRAND,
        max_concurrent=3,
    )

    print(f"生成了 {len(users)} 个用户画像:")
    for user in users:
        print(
            f"\n  [{user.influence_level.value}] {user.user_id} ({user.persona_type})"
        )
        print(f"  粉丝: {user.follower_count}")
        print(f"  画像: {user.profile[:80]}...")
        if user.memory:
            print(f"  记忆: {user.memory[0][:50]}...")

    assert len(users) == 5
    print("✓ 测试通过")

    return users


async def run_all_tests():
    """运行所有阶段2测试"""
    print("=" * 50)
    print("SentiSim 阶段2测试：数据生成")
    print("=" * 50)

    try:
        # 1. 网络拓扑（不需要 LLM）
        topology = await test_network_topology()

        # 2. 影响力分配（不需要 LLM）
        influence_levels = test_influence_assignment(topology)

        # 3. 人群类型生成（需要 LLM）
        personas = await test_persona_generation()

        # 4. 用户画像生成（需要 LLM，批量，包含初始记忆）
        users = await test_user_generation(topology, influence_levels, personas)

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

"""
创建世界脚本

用法：
    python create_world.py --desc "主打天然成分的护肤品牌，目标用户是25~35岁注重成分安全的城市女性" \
                           --n-agents 100

流程：
    1. LLM 生成 5~6 个正交维度
    2. 按 tier 比例采样 agents + 生成社交图（固定 seed）
    3. LLM 批量生成每个 agent 的 persona（分批并发）
    4. 持久化到 MongoDB（worlds / agents / social_graphs）
"""

import argparse
import asyncio
import uuid

import networkx as nx
from config import settings
from llm.persona import generate_personas
from llm.world_gen import generate_dimensions
from sim.db import init_db, save_agents, save_graph, save_world
from sim.world import generate_agents, generate_social_graph


async def create_world(description: str, n_agents: int, seed: int, batch_size: int):
    await init_db()
    print(f"\n=== 创建世界 ===")
    print(f"描述：{description}")
    print(f"Agent 数量：{n_agents}，seed={seed}\n")

    # Step 1: 生成维度
    print("Step 1/4  生成维度...")
    community_name, dimensions, demographic_dimensions, brand_agent = (
        await generate_dimensions(description)
    )
    print(f"  社区名称：{community_name}")
    print(f"  品牌：{brand_agent.brand_name}（{brand_agent.tone_of_voice}）")
    print("  数值维度：")
    for d in dimensions:
        print(f"    - {d.name}：{d.low_label} → {d.high_label}")
    print("  人口统计维度：")
    for d in demographic_dimensions:
        print(f"    - {d.name}：{d.labels}")

    # Step 2: 生成 agents + 社交图
    print("\nStep 2/4  生成 agents + 社交图...")
    world_id = f"world_{uuid.uuid4().hex[:8]}"
    agents = generate_agents(world_id, dimensions, n_total=n_agents, seed=seed)
    graph: nx.DiGraph = generate_social_graph(agents, seed=seed)
    edges = list(graph.edges())
    print(
        f"  agents: {len(agents)}（KOL={sum(1 for a in agents if a.tier.value=='kol')}，"
        f"KOC={sum(1 for a in agents if a.tier.value=='koc')}，"
        f"Normal={sum(1 for a in agents if a.tier.value=='normal')}）"
    )
    print(f"  edges:  {len(edges)}")

    # Step 3: 批量生成 persona
    print(f"\nStep 3/4  批量生成 persona（batch_size={batch_size}）...")
    agents = await generate_personas(
        agents, dimensions, demographic_dimensions, description, batch_size=batch_size
    )

    # Step 4: 持久化
    print("\nStep 4/4  持久化到 MongoDB...")
    await save_world(
        world_id,
        community_name,
        description,
        dimensions,
        demographic_dimensions,
        brand_agent,
    )
    await save_agents(agents)
    await save_graph(world_id, edges)
    print(f"  world_id: {world_id}")

    print(f"\n=== 完成 ===")
    print(f"world_id:       {world_id}")
    print(f"community_name: {community_name}")
    print(f"agents:         {len(agents)}")
    print(f"MongoDB DB:     {settings.mongodb_db}")

    # 打印几个 persona 示例
    print("\n--- Persona 示例 ---")
    for a in agents[:3]:
        print(f"[{a.tier.value}] {a.id}: {a.persona}\n")


def main():
    parser = argparse.ArgumentParser(description="创建虚拟社区世界")
    parser.add_argument("--desc", required=True, help="产品/场景/目标人群描述")
    parser.add_argument(
        "--n-agents", type=int, default=100, help="agent 数量（默认100）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="persona 生成并发批次大小（默认10）"
    )
    args = parser.parse_args()

    asyncio.run(
        create_world(
            description=args.desc,
            n_agents=args.n_agents,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()

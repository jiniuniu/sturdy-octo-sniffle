import asyncio
import uuid
from sim.world import create_world
from sim.models import Content, ContentType, Dimension
from sim.engine import SimulationEngine, SimulationConfig

# 用固定维度做本地测试，无需 LLM
dimensions = [
    Dimension("成分敏感度", "对天然/无添加成分的关注程度", "不在乎成分，图效果就行", "成分党，每个原料都要研究"),
    Dimension("价格敏感度", "对性价比的在意程度",         "价格不是问题",           "精打细算，只买高性价比"),
    Dimension("颜值导向",   "外观在购买决策中的权重",     "功效优先，外观无所谓",   "颜值即正义，包装要好看"),
    Dimension("KOL依赖度",  "是否跟随博主决策",           "自己研究，不跟风",       "博主推什么就买什么"),
    Dimension("环保意识",   "对可持续/绿色理念的认同",    "环保不影响我买东西",     "优先选择环保品牌"),
]

world_id, agents, graph = create_world(dimensions, n_agents=50, seed=42)
agents_dict = {a.id: a for a in agents}

seed = Content(
    id=f"post_{uuid.uuid4().hex[:12]}",
    sim_id="sim_test_001",
    author_id="brand_001",
    type=ContentType.ORIGINAL,
    vector=[0.9, 0.4, 0.7, 0.3, 0.8],
    text="全新天然成分精华上市！",
    sim_time=0.0,
)

engine = SimulationEngine(
    sim_id="sim_test_001",
    world_id=world_id,
    agents=agents_dict,
    graph=graph,
    seed_content=seed,
    config=SimulationConfig(time_window_days=3.0, max_events=200, feed_update_interval=50),
)


async def run():
    events = []
    async for sse in engine.run():
        events.append(sse)

    done = next(e for e in events if e["type"] == "simulation_done")
    print("=== 模拟完成 ===")
    print(f"总事件数: {done['total_events']}")
    print(f"指标: {done['metrics']}")
    print(f"转发率: {done['repost_rate']}")
    print(f"评论率: {done['comment_rate']}")

    reacted = [e for e in events if e["type"] == "agent_reacted"]
    actions = {}
    for e in reacted:
        actions[e["action"]] = actions.get(e["action"], 0) + 1
    print(f"行为分布: {actions}")


asyncio.run(run())

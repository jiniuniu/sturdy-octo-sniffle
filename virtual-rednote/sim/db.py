"""
MongoDB 读写封装（motor 异步驱动）
持久化：worlds（维度定义）、agents（含 persona）、social_graph（边列表）
"""
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings
from .models import Dimension, DemographicDimension, BrandAgent, Agent, AgentTier


_client: AsyncIOMotorClient = None


def get_db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_uri)
    return _client[settings.mongodb_db]


async def init_db():
    """建索引（幂等，可重复调用）"""
    db = get_db()
    await db.agents.create_index([("world_id", 1), ("agent_id", 1)], unique=True)
    await db.simulations.create_index([("world_id", 1), ("created_at", -1)])


# ── simulations ───────────────────────────────────────

async def save_simulation(doc: dict):
    await get_db().simulations.replace_one({"_id": doc["_id"]}, doc, upsert=True)


async def list_simulations(world_id: str) -> list[dict]:
    return await get_db().simulations.find(
        {"world_id": world_id},
        {"event_log": 0},   # 列表页不返回事件流，太大
    ).sort("created_at", -1).to_list(length=None)


# ── worlds ────────────────────────────────────────────

async def save_world(
    world_id: str,
    community_name: str,
    description: str,
    dimensions: list[Dimension],
    demographic_dimensions: list[DemographicDimension],
    brand_agent: BrandAgent,
):
    await get_db().worlds.replace_one(
        {"_id": world_id},
        {
            "_id": world_id,
            "community_name": community_name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dimensions": [
                {
                    "name": d.name,
                    "description": d.description,
                    "low_label": d.low_label,
                    "high_label": d.high_label,
                }
                for d in dimensions
            ],
            "demographic_dimensions": [
                {"name": d.name, "labels": d.labels}
                for d in demographic_dimensions
            ],
            "brand_agent": {
                "brand_name":    brand_agent.brand_name,
                "tone_of_voice": brand_agent.tone_of_voice,
                "reply_style":   brand_agent.reply_style,
            },
        },
        upsert=True,
    )


async def load_brand_agent(world_id: str) -> BrandAgent | None:
    doc = await get_db().worlds.find_one({"_id": world_id}, {"brand_agent": 1})
    if doc is None or "brand_agent" not in doc:
        return None
    b = doc["brand_agent"]
    return BrandAgent(
        brand_name=b["brand_name"],
        tone_of_voice=b["tone_of_voice"],
        reply_style=b["reply_style"],
    )


async def get_world(world_id: str) -> dict | None:
    return await get_db().worlds.find_one({"_id": world_id})


async def list_worlds() -> list[dict]:
    return await get_db().worlds.find().sort("created_at", -1).to_list(length=None)


async def delete_world(world_id: str):
    await get_db().worlds.delete_one({"_id": world_id})
    await get_db().agents.delete_many({"world_id": world_id})
    await get_db().social_graphs.delete_one({"_id": world_id})


# ── agents ────────────────────────────────────────────

async def save_agents(agents: list[Agent]):
    if not agents:
        return
    world_id = agents[0].world_id
    docs = [
        {
            "agent_id": a.id,
            "world_id": a.world_id,
            "tier": a.tier.value,
            "vector": a.vector,
            "activity": a.activity,
            "expressiveness": a.expressiveness,
            "sharing": a.sharing,
            "persona": a.persona,
        }
        for a in agents
    ]
    await get_db().agents.delete_many({"world_id": world_id})
    await get_db().agents.insert_many(docs)


async def load_agents(world_id: str) -> list[Agent]:
    docs = await get_db().agents.find({"world_id": world_id}).to_list(length=None)
    return [
        Agent(
            id=d["agent_id"],
            world_id=d["world_id"],
            tier=AgentTier(d["tier"]),
            vector=d["vector"],
            activity=d["activity"],
            expressiveness=d["expressiveness"],
            sharing=d["sharing"],
            persona=d.get("persona", ""),
        )
        for d in docs
    ]


# ── social graph ──────────────────────────────────────

async def save_graph(world_id: str, edges: list[tuple[str, str]]):
    await get_db().social_graphs.replace_one(
        {"_id": world_id},
        {"_id": world_id, "edges": [{"s": s, "t": t} for s, t in edges]},
        upsert=True,
    )


async def load_graph(world_id: str) -> list[tuple[str, str]]:
    doc = await get_db().social_graphs.find_one({"_id": world_id})
    if doc is None:
        return []
    return [(e["s"], e["t"]) for e in doc["edges"]]

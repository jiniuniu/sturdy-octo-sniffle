import asyncio
import json
import secrets
import uuid

import networkx as nx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sim.db import (
    delete_world,
    get_world,
    init_db,
    list_simulations,
    list_worlds,
    load_agents,
    load_brand_agent,
    load_graph,
    save_simulation,
)
from sim.engine import SimulationConfig, SimulationEngine
from sim.models import Content, ContentType, Dimension
from sse_starlette.sse import EventSourceResponse

# world_id -> {agents_dict, graph, dimensions, community_name}
_worlds: dict[str, dict] = {}
# sim_id -> SimulationEngine
_simulations: dict[str, SimulationEngine] = {}

app = FastAPI(title="Virtual RedNote")


# 全局认证
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    from config import settings

    if not settings.auth_username:
        return await call_next(request)
    import base64

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode()
            username, _, password = decoded.partition(":")
            if secrets.compare_digest(
                username, settings.auth_username
            ) and secrets.compare_digest(password, settings.auth_password):
                return await call_next(request)
        except Exception:
            pass
    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Virtual RedNote"'},
    )


@app.on_event("startup")
async def startup():
    await init_db()


# ── 请求/响应模型 ────────────────────────────────────────


class AnalyzeContentRequest(BaseModel):
    world_id: str
    content_text: str


class AnalyzeContentResponse(BaseModel):
    vector: list[float]
    dimension_scores: list[dict]


class SimulateRequest(BaseModel):
    world_id: str
    content_text: str
    content_vector: list[float]


class SimulateResponse(BaseModel):
    sim_id: str
    world_id: str
    seed_content_id: str


# ── API ──────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/worlds")
async def get_worlds():
    worlds = await list_worlds()
    return {
        "worlds": [
            {
                "world_id": w["_id"],
                "community_name": w["community_name"],
                "created_at": w.get("created_at", ""),
                "dimensions": w["dimensions"],
            }
            for w in worlds
        ]
    }


@app.post("/worlds/{world_id}/load")
async def load_world(world_id: str):
    """加载已有世界，从 DB 读取持久化的 agents 和 graph"""
    world_doc = await get_world(world_id)
    if world_doc is None:
        raise HTTPException(404, f"world {world_id} not found")

    dimensions = [
        Dimension(d["name"], d["description"], d["low_label"], d["high_label"])
        for d in world_doc["dimensions"]
    ]
    agents = await load_agents(world_id)
    if not agents:
        raise HTTPException(
            404, f"world {world_id} has no agents, please run create_world.py first"
        )

    edges = await load_graph(world_id)
    graph = nx.DiGraph()
    for a in agents:
        graph.add_node(a.id, tier=a.tier.value)
    graph.add_edges_from(edges)

    brand_agent = await load_brand_agent(world_id)
    _worlds[world_id] = {
        "agents_dict": {a.id: a for a in agents},
        "graph": graph,
        "dimensions": dimensions,
        "community_name": world_doc["community_name"],
        "brand_agent": brand_agent,
    }
    return {
        "world_id": world_id,
        "community_name": world_doc["community_name"],
        "n_agents": len(agents),
        "dimensions": world_doc["dimensions"],
    }


@app.delete("/worlds/{world_id}")
async def delete_world_endpoint(world_id: str):
    await delete_world(world_id)
    _worlds.pop(world_id, None)
    return {"ok": True}


@app.post("/worlds/analyze-content", response_model=AnalyzeContentResponse)
async def analyze_content(req: AnalyzeContentRequest):
    from llm.content import analyze_content_vector

    world = _worlds.get(req.world_id)
    if world is None:
        world_doc = await get_world(req.world_id)
        if world_doc is None:
            raise HTTPException(404, f"world {req.world_id} not found")
        dimensions = [
            Dimension(d["name"], d["description"], d["low_label"], d["high_label"])
            for d in world_doc["dimensions"]
        ]
    else:
        dimensions = world["dimensions"]

    vector = await analyze_content_vector(req.content_text, dimensions)
    return AnalyzeContentResponse(
        vector=vector,
        dimension_scores=[
            {"name": d.name, "score": round(vector[i], 3)}
            for i, d in enumerate(dimensions)
        ],
    )


@app.post("/simulate/start", response_model=SimulateResponse)
async def simulate_start(req: SimulateRequest):
    world = _worlds.get(req.world_id)
    if world is None:
        raise HTTPException(
            404,
            f"world {req.world_id} not loaded, call /worlds/{req.world_id}/load first",
        )

    sim_id = f"sim_{uuid.uuid4().hex[:10]}"
    seed_content = Content(
        id=f"post_{uuid.uuid4().hex[:12]}",
        sim_id=sim_id,
        author_id="brand_001",
        type=ContentType.ORIGINAL,
        vector=req.content_vector,
        text=req.content_text,
        sim_time=0.0,
    )

    brand_agent = world.get("brand_agent") or await load_brand_agent(req.world_id)
    engine = SimulationEngine(
        sim_id=sim_id,
        world_id=req.world_id,
        agents=world["agents_dict"],
        graph=world["graph"],
        seed_content=seed_content,
        config=SimulationConfig(time_window_days=7.0, max_events=5000),
        brand_agent=brand_agent,
    )
    _simulations[sim_id] = engine

    return SimulateResponse(
        sim_id=sim_id, world_id=req.world_id, seed_content_id=seed_content.id
    )


@app.get("/worlds/{world_id}/simulations")
async def get_simulations(world_id: str):
    sims = await list_simulations(world_id)
    return {
        "simulations": [
            {
                "sim_id": s["_id"],
                "content_text": s.get("content_text", ""),
                "created_at": s.get("created_at", ""),
                "metrics": s.get("metrics", {}),
                "total_events": s.get("total_events", 0),
            }
            for s in sims
        ]
    }


class SaveSimRequest(BaseModel):
    content_text: str
    metrics: dict
    total_events: int
    event_log: list[dict] = []
    comments: list[dict] = []
    brand_replies: list[dict] = []


@app.post("/simulate/{sim_id}/save")
async def save_sim(sim_id: str, req: SaveSimRequest):
    engine = _simulations.get(sim_id)
    world_id = engine.world_id if engine else None
    if world_id is None:
        raise HTTPException(404, f"simulation {sim_id} not found")
    await save_simulation(
        {
            "_id": sim_id,
            "world_id": world_id,
            "content_text": req.content_text,
            "created_at": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "metrics": req.metrics,
            "total_events": req.total_events,
            "event_log": req.event_log,
            "comments": req.comments,
            "brand_replies": req.brand_replies,
        }
    )
    return {"ok": True}


@app.get("/simulate/{sim_id}")
async def get_simulation(sim_id: str):
    from sim.db import get_db

    doc = await get_db().simulations.find_one({"_id": sim_id})
    if doc is None:
        raise HTTPException(404, f"simulation {sim_id} not found")
    doc["sim_id"] = doc.pop("_id")

    # 从持久化数据拼 graph_init
    world_id = doc.get("world_id")
    if world_id and not doc.get("graph_init"):
        agents = await load_agents(world_id)
        edges = await load_graph(world_id)
        nodes = [
            {"id": a.id, "tier": a.tier.value, "persona": a.persona} for a in agents
        ]
        nodes.append({"id": "brand_001", "tier": "brand", "persona": ""})
        doc["graph_init"] = {
            "nodes": nodes,
            "edges": [{"source": s, "target": t} for s, t in edges],
        }

    return doc


@app.delete("/simulate/{sim_id}")
async def delete_simulation(sim_id: str):
    from sim.db import get_db

    await get_db().simulations.delete_one({"_id": sim_id})
    _simulations.pop(sim_id, None)
    return {"ok": True}


@app.get("/simulate/{sim_id}/stream")
async def simulate_stream(sim_id: str):
    engine = _simulations.get(sim_id)
    if engine is None:
        raise HTTPException(404, f"simulation {sim_id} not found")

    async def generate():
        async for event in engine.run():
            yield {"data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(generate())


# 静态文件放最后，避免路由冲突
app.mount("/", StaticFiles(directory="web", html=True), name="web")

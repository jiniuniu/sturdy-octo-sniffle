from db.client import get_db
from models.dimension import DimensionDoc, Dimension


def _make_docs(parent_key: str, parent_id: str, dimensions: list[Dimension]) -> list[dict]:
    return [
        DimensionDoc(
            campaign_id=parent_id,
            dim_id=dim.id,
            name=dim.name,
            description=dim.description,
            relevance_reason=dim.relevance_reason,
            source=dim.source,
            segments=dim.segments,
            order=i,
        ).model_dump() | {parent_key: parent_id}
        for i, dim in enumerate(dimensions)
    ]


# ── 通用 study_id 版本 ────────────────────────────────────────

async def insert_many_for_study(study_id: str, dimensions: list[Dimension]):
    db = get_db()
    docs = [
        {
            "study_id": study_id,
            "dim_id": dim.id,
            "name": dim.name,
            "description": dim.description,
            "relevance_reason": dim.relevance_reason,
            "source": dim.source,
            "segments": [s.model_dump() for s in dim.segments],
            "order": i,
        }
        for i, dim in enumerate(dimensions)
    ]
    await db.dimensions.insert_many(docs)


async def get_by_study(study_id: str) -> list[dict]:
    db = get_db()
    cursor = db.dimensions.find({"study_id": study_id}).sort("order", 1)
    return [doc async for doc in cursor]


# ── 向后兼容 campaign_id 版本 ─────────────────────────────────

async def insert_many(campaign_id: str, dimensions: list[Dimension]):
    db = get_db()
    docs = [
        DimensionDoc(
            campaign_id=campaign_id,
            dim_id=dim.id,
            name=dim.name,
            description=dim.description,
            relevance_reason=dim.relevance_reason,
            source=dim.source,
            segments=dim.segments,
            order=i,
        ).model_dump()
        for i, dim in enumerate(dimensions)
    ]
    await db.dimensions.insert_many(docs)


async def delete_by_campaign(campaign_id: str):
    db = get_db()
    await db.dimensions.delete_many({"campaign_id": campaign_id})


async def get_by_campaign(campaign_id: str) -> list[dict]:
    db = get_db()
    cursor = db.dimensions.find({"campaign_id": campaign_id}).sort("order", 1)
    return [doc async for doc in cursor]

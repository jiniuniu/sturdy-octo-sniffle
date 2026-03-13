from db.client import get_db
from models.dimension import DimensionDoc, Dimension


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

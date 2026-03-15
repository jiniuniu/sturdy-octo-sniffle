from db.client import get_db
from models.dimension import Dimension


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

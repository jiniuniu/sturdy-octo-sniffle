from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[settings.mongodb_db]


async def close_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def init_indexes():
    db = get_db()
    await db.studies.create_index("status")
    await db.dimensions.create_index("study_id")
    await db.personas.create_index("study_id")
    await db.personas.create_index([("study_id", 1), ("persona_id", 1)])
    await db.responses.create_index("study_id")
    await db.responses.create_index([("study_id", 1), ("primary_stance", 1)])

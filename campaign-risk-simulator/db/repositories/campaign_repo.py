from datetime import datetime, timezone
from bson import ObjectId
from db.client import get_db
from models.campaign import CampaignDoc, CampaignStatus


def _to_doc(raw: dict) -> dict:
    raw["id"] = str(raw.pop("_id"))
    return raw


async def create(doc: CampaignDoc) -> str:
    db = get_db()
    result = await db.campaigns.insert_one(doc.model_dump())
    return str(result.inserted_id)


async def get(campaign_id: str) -> dict | None:
    db = get_db()
    raw = await db.campaigns.find_one({"_id": ObjectId(campaign_id)})
    if raw is None:
        return None
    return _to_doc(raw)


async def update_status(campaign_id: str, status: CampaignStatus):
    db = get_db()
    update: dict = {"$set": {"status": status.value}}
    if status == CampaignStatus.processing_visual or status == CampaignStatus.extracting_dimensions:
        # 第一次进入运行态时记录 started_at
        update["$setOnInsert"] = {}
        update["$set"]["started_at"] = datetime.now(timezone.utc)
    if status == CampaignStatus.completed:
        update["$set"]["completed_at"] = datetime.now(timezone.utc)
    await db.campaigns.update_one({"_id": ObjectId(campaign_id)}, update)


async def update_visual(campaign_id: str, visual_description: str):
    db = get_db()
    await db.campaigns.update_one(
        {"_id": ObjectId(campaign_id)},
        {"$set": {"visual_description": visual_description}},
    )


async def update_summary(campaign_id: str, summary: dict):
    db = get_db()
    await db.campaigns.update_one(
        {"_id": ObjectId(campaign_id)},
        {"$set": {"summary": summary}},
    )


async def set_error(campaign_id: str, message: str):
    db = get_db()
    await db.campaigns.update_one(
        {"_id": ObjectId(campaign_id)},
        {"$set": {"status": CampaignStatus.error.value, "error_message": message}},
    )

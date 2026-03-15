from datetime import datetime, timezone
from bson import ObjectId
from db.client import get_db
from models.study import StudyDoc, StudyStatus


def _to_doc(raw: dict) -> dict:
    raw["id"] = str(raw.pop("_id"))
    return raw


async def create(doc: StudyDoc) -> str:
    db = get_db()
    result = await db.studies.insert_one(doc.model_dump())
    return str(result.inserted_id)


async def get(study_id: str) -> dict | None:
    db = get_db()
    raw = await db.studies.find_one({"_id": ObjectId(study_id)})
    if raw is None:
        return None
    return _to_doc(raw)


async def update_status(study_id: str, status: StudyStatus):
    db = get_db()
    update: dict = {"$set": {"status": status.value}}
    if status in (StudyStatus.processing_visual, StudyStatus.extracting_dimensions):
        update["$set"]["started_at"] = datetime.now(timezone.utc)
    if status == StudyStatus.completed:
        update["$set"]["completed_at"] = datetime.now(timezone.utc)
    await db.studies.update_one({"_id": ObjectId(study_id)}, update)


async def update_visual(study_id: str, visual_description: str):
    db = get_db()
    await db.studies.update_one(
        {"_id": ObjectId(study_id)},
        {"$set": {"visual_description": visual_description}},
    )


async def update_summary(study_id: str, summary: dict):
    db = get_db()
    await db.studies.update_one(
        {"_id": ObjectId(study_id)},
        {"$set": {"summary": summary}},
    )


async def set_error(study_id: str, message: str):
    db = get_db()
    await db.studies.update_one(
        {"_id": ObjectId(study_id)},
        {"$set": {"status": StudyStatus.error.value, "error_message": message}},
    )

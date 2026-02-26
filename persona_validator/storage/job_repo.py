# storage/job_repo.py
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from storage.client import get_db

# 列表接口只返回摘要字段，不含中间产物大对象
_SUMMARY_PROJECTION = {
    "_id": 0,
    "job_id": 1,
    "status": 1,
    "product": 1,
    "charts": 1,
    "error": 1,
    "created_at": 1,
    "updated_at": 1,
}

_DETAIL_PROJECTION = {"_id": 0}


class JobRepo:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.col = db["validation_jobs"]

    async def create_indexes(self) -> None:
        await self.col.create_index([("job_id", ASCENDING)], unique=True)
        await self.col.create_index([("created_at", DESCENDING)])

    async def create(self, job_id: str, product: dict) -> None:
        now = datetime.now(timezone.utc)
        await self.col.insert_one(
            {
                "job_id": job_id,
                "status": "pending",
                "product": product,
                # 中间产物，逐步填充
                "axes": None,
                "vectors": None,
                "personas": None,
                "questionnaire": None,
                "evaluations": None,
                "insights": None,
                # 最终产物
                "charts": None,        # {name: qiniu_url}
                "report_markdown": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        )

    async def patch(self, job_id: str, fields: dict) -> None:
        """原子更新指定字段，自动刷新 updated_at。"""
        fields["updated_at"] = datetime.now(timezone.utc)
        await self.col.update_one({"job_id": job_id}, {"$set": fields})

    async def get(self, job_id: str) -> dict | None:
        return await self.col.find_one({"job_id": job_id}, _DETAIL_PROJECTION)

    async def list(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[dict], int]:
        skip = (page - 1) * page_size
        cursor = (
            self.col.find({}, _SUMMARY_PROJECTION)
            .sort("created_at", DESCENDING)
            .skip(skip)
            .limit(page_size)
        )
        items = await cursor.to_list(length=page_size)
        total = await self.col.count_documents({})
        return items, total


def get_job_repo() -> JobRepo:
    return JobRepo(get_db())

from db.client import get_db
from models.response import ResponseDoc, ResponseOutput
from models.study import ResponseMode

_STANCE_ORDER = {"负面": 0, "复杂": 1, "中立": 2, "正面": 3}


async def insert(study_id: str, output: ResponseOutput, response_mode: ResponseMode):
    db = get_db()
    doc = ResponseDoc(
        study_id=study_id,
        persona_id=output.persona_id,
        response_mode=response_mode.value,
        primary_stance=output.primary_stance,
        key_quote=output.key_quote,
        reasoning=output.reasoning,
        content=output.content,
        signals=output.signals,
    ).model_dump()
    await db.responses.insert_one(doc)


async def get_by_study(study_id: str) -> list[dict]:
    """按 primary_stance 排序（负面/复杂优先）"""
    db = get_db()
    cursor = db.responses.find({"study_id": study_id})
    docs = [doc async for doc in cursor]
    docs.sort(key=lambda d: _STANCE_ORDER.get(d.get("primary_stance", "中立"), 2))
    return docs


async def count(study_id: str) -> int:
    db = get_db()
    return await db.responses.count_documents({"study_id": study_id})

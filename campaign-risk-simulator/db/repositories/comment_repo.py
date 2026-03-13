from db.client import get_db
from models.comment import CommentDoc, CommentOutput

_RISK_ORDER = {"高": 0, "中": 1, "低": 2}


async def insert(campaign_id: str, output: CommentOutput):
    db = get_db()
    doc = CommentDoc(
        campaign_id=campaign_id,
        persona_id=output.persona_id,
        situation_reading=output.reasoning.situation_reading,
        emotional_state=output.reasoning.emotional_state,
        action_choice=output.reasoning.action_choice,
        platform=output.comment.platform,
        text=output.comment.text,
        tone=output.comment.tone,
        length_type=output.comment.length_type,
        spread_likelihood=output.risk_signals.spread_likelihood,
        spread_reason=output.risk_signals.spread_reason,
        trigger_keywords=output.risk_signals.trigger_keywords,
        escalation_risk=output.risk_signals.escalation_risk,
    ).model_dump()
    await db.comments.insert_one(doc)


async def delete_by_campaign(campaign_id: str):
    db = get_db()
    await db.comments.delete_many({"campaign_id": campaign_id})


async def get_by_campaign(campaign_id: str) -> list[dict]:
    """按 escalation_risk 降序（高→中→低）"""
    db = get_db()
    cursor = db.comments.find({"campaign_id": campaign_id})
    docs = [doc async for doc in cursor]
    docs.sort(key=lambda d: _RISK_ORDER.get(d.get("escalation_risk", "低"), 2))
    return docs


async def count(campaign_id: str) -> int:
    db = get_db()
    return await db.comments.count_documents({"campaign_id": campaign_id})

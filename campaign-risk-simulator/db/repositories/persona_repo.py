from db.client import get_db
from models.persona import PersonaDoc, PersonaOutput


async def insert_placeholders(campaign_id: str, positions: list[dict]):
    """Sobol 采样完成后，预写入占位记录（completed=False）"""
    db = get_db()
    docs = [
        PersonaDoc(
            campaign_id=campaign_id,
            persona_id=pos["id"],
            dimension_scores=pos["scores"],
            dimension_segments=pos["segments"],
        ).model_dump()
        for pos in positions
    ]
    await db.personas.insert_many(docs)


async def update(campaign_id: str, persona_id: str, output: PersonaOutput):
    db = get_db()
    await db.personas.update_one(
        {"campaign_id": campaign_id, "persona_id": persona_id},
        {
            "$set": {
                "identity_summary": output.identity_summary,
                "demographics": output.demographics.model_dump(),
                "life_context": output.life_context,
                "values_and_worldview": output.values_and_worldview,
                "brand_relationship": output.brand_relationship,
                "psychological_trigger": output.psychological_trigger,
                "initial_reaction_hint": output.initial_reaction_hint,
                "completed": True,
            }
        },
    )


async def delete_by_campaign(campaign_id: str):
    db = get_db()
    await db.personas.delete_many({"campaign_id": campaign_id})


async def get_by_campaign(campaign_id: str) -> list[dict]:
    db = get_db()
    cursor = db.personas.find({"campaign_id": campaign_id})
    return [doc async for doc in cursor]


async def count(campaign_id: str) -> int:
    db = get_db()
    return await db.personas.count_documents({"campaign_id": campaign_id})


async def count_completed(campaign_id: str) -> int:
    db = get_db()
    return await db.personas.count_documents(
        {"campaign_id": campaign_id, "completed": True}
    )

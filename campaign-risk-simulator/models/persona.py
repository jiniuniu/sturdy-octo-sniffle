from pydantic import BaseModel
from typing import Optional


class Demographics(BaseModel):
    age: int
    gender: str
    city: str
    occupation: str
    income_range: str


class DimensionSegmentRef(BaseModel):
    segment_id: str
    label: str
    description: str


class PersonaOutput(BaseModel):
    id: str
    identity_summary: str
    demographics: Demographics
    life_context: str
    values_and_worldview: list[str]
    brand_relationship: str
    psychological_trigger: str
    initial_reaction_hint: Optional[str] = None


class PersonaDoc(BaseModel):
    study_id: str
    persona_id: str
    dimension_scores: dict[str, float]
    dimension_segments: dict[str, DimensionSegmentRef]
    identity_summary: Optional[str] = None
    demographics: Optional[Demographics] = None
    life_context: Optional[str] = None
    values_and_worldview: Optional[list[str]] = None
    brand_relationship: Optional[str] = None
    psychological_trigger: Optional[str] = None
    initial_reaction_hint: Optional[str] = None
    completed: bool = False

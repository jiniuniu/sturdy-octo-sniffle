from pydantic import BaseModel, Field
from typing import Literal

DimensionSource = Literal["library", "campaign-specific", "preset", "hidden"]


class Segment(BaseModel):
    id: str
    label: str
    description: str


class Dimension(BaseModel):
    id: str
    name: str
    description: str
    relevance_reason: str
    source: DimensionSource
    segments: list[Segment] = Field(min_length=3, max_length=5)


class DimensionsOutput(BaseModel):
    dimensions: list[Dimension] = Field(min_length=4, max_length=8)


class DimensionDoc(BaseModel):
    study_id: str
    dim_id: str
    name: str
    description: str
    relevance_reason: str
    source: DimensionSource
    segments: list[Segment]
    order: int

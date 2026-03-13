from pydantic import BaseModel, Field
from typing import Literal

DimensionSource = Literal["library", "campaign-specific", "hidden"]


class Segment(BaseModel):
    id: str                        # "seg_1"
    label: str
    description: str


class Dimension(BaseModel):
    id: str                        # "dim_1"
    name: str
    description: str
    relevance_reason: str
    source: DimensionSource
    segments: list[Segment] = Field(min_length=3, max_length=5)


# LLM 结构化输出：2-4 个可见维度 + 2 个 hidden 维度
class DimensionsOutput(BaseModel):
    dimensions: list[Dimension] = Field(min_length=4, max_length=8)


# DB Document
class DimensionDoc(BaseModel):
    campaign_id: str
    dim_id: str
    name: str
    description: str
    relevance_reason: str
    source: DimensionSource
    segments: list[Segment]
    order: int

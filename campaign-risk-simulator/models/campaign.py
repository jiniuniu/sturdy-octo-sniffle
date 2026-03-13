from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum


class CampaignStatus(str, Enum):
    idle = "idle"
    processing_visual = "processing_visual"
    extracting_dimensions = "extracting_dimensions"
    sampling = "sampling"
    generating_personas = "generating_personas"
    generating_comments = "generating_comments"
    completed = "completed"
    error = "error"


TERMINAL_STATUSES = {CampaignStatus.completed, CampaignStatus.error}
RUNNING_STATUSES = {
    CampaignStatus.processing_visual,
    CampaignStatus.extracting_dimensions,
    CampaignStatus.sampling,
    CampaignStatus.generating_personas,
    CampaignStatus.generating_comments,
}


# --- Request ---

class CampaignCreate(BaseModel):
    title: str
    description: str
    n_personas: int = Field(default=8, ge=4, le=16)
    target_market: str = "中国大陆"


# --- Response ---

class CampaignCreated(BaseModel):
    campaign_id: str
    status: CampaignStatus
    created_at: datetime


class PipelineProgress(BaseModel):
    personas_completed: int = 0
    personas_total: int = 0
    comments_completed: int = 0
    comments_total: int = 0


class CampaignStatusResponse(BaseModel):
    campaign_id: str
    status: CampaignStatus
    progress: PipelineProgress
    stages_completed: list[str]
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# --- DB Document ---

class CampaignDoc(BaseModel):
    title: str
    description: str
    n_personas: int = 8
    target_market: str = "中国大陆"
    image_key: Optional[str] = None
    image_url: Optional[str] = None
    visual_description: Optional[str] = None
    status: CampaignStatus = CampaignStatus.idle
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

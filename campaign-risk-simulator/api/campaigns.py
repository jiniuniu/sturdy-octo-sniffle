from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from db.repositories import campaign_repo, dimension_repo, persona_repo, comment_repo
from models.campaign import (
    CampaignCreated,
    CampaignDoc,
    CampaignStatus,
    CampaignStatusResponse,
    PipelineProgress,
    RUNNING_STATUSES,
)
from models.comment import RiskSummary
from pipeline.runner import run_pipeline
from storage.qiniu import upload_image

router = APIRouter(prefix="/campaigns", tags=["campaigns"])
templates = Jinja2Templates(directory="templates")


# ── 辅助函数 ────────────────────────────────────────────────

def _compute_risk_summary(comments: list[dict]) -> RiskSummary:
    if not comments:
        return RiskSummary(
            high_escalation_count=0,
            high_spread_count=0,
            top_trigger_keywords=[],
            tone_distribution={},
            riskiest_persona_id="",
        )

    high_escalation = [c for c in comments if c["escalation_risk"] == "高"]
    high_spread = [c for c in comments if c["spread_likelihood"] == "高"]

    all_keywords = [kw for c in comments for kw in c.get("trigger_keywords", [])]
    top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(5)]

    tone_dist = dict(Counter(c["tone"] for c in comments))

    # 优先取升级风险高且传播风险高的；其次只升级高；再次第一条
    riskiest = next(
        (c for c in comments if c["escalation_risk"] == "高" and c["spread_likelihood"] == "高"),
        next((c for c in comments if c["escalation_risk"] == "高"), comments[0]),
    )

    return RiskSummary(
        high_escalation_count=len(high_escalation),
        high_spread_count=len(high_spread),
        top_trigger_keywords=top_keywords,
        tone_distribution=tone_dist,
        riskiest_persona_id=riskiest["persona_id"],
    )


async def _get_progress(campaign_id: str) -> PipelineProgress:
    personas_total = await persona_repo.count(campaign_id)
    personas_completed = await persona_repo.count_completed(campaign_id)
    comments_total = await comment_repo.count(campaign_id)
    return PipelineProgress(
        personas_completed=personas_completed,
        personas_total=personas_total,
        comments_completed=comments_total,
        comments_total=personas_total,
    )


def _stages_completed(status: CampaignStatus) -> list[str]:
    order = [
        "processing_visual",
        "extracting_dimensions",
        "sampling",
        "generating_personas",
        "generating_comments",
        "completed",
    ]
    try:
        idx = order.index(status.value)
    except ValueError:
        return []
    return order[:idx]


def _require_campaign(campaign: dict | None, campaign_id: str) -> dict:
    if campaign is None:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} 不存在")
    return campaign


# ── 路由 ────────────────────────────────────────────────────

@router.post("", status_code=201, response_model=CampaignCreated)
async def create_campaign(
    title: str = Form(...),
    description: str = Form(...),
    n_personas: int = Form(default=8, ge=4, le=16),
    target_market: str = Form(default="中国大陆"),
    image: Optional[UploadFile] = File(default=None),
):
    image_key, image_url = None, None

    if image is not None:
        data = await image.read()
        image_key, image_url = await upload_image(data, image.filename or "image.jpg")

    doc = CampaignDoc(
        title=title,
        description=description,
        n_personas=n_personas,
        target_market=target_market,
        image_key=image_key,
        image_url=image_url,
    )
    campaign_id = await campaign_repo.create(doc)
    return CampaignCreated(
        campaign_id=campaign_id,
        status=CampaignStatus.idle,
        created_at=datetime.now(timezone.utc),
    )


@router.post("/{campaign_id}/run", status_code=202)
async def run_campaign(campaign_id: str, background_tasks: BackgroundTasks):
    campaign = _require_campaign(await campaign_repo.get(campaign_id), campaign_id)

    current = CampaignStatus(campaign["status"])
    if current in RUNNING_STATUSES:
        raise HTTPException(
            status_code=409,
            detail={"error": "pipeline_already_running", "current_status": current.value},
        )
    if current == CampaignStatus.completed:
        raise HTTPException(
            status_code=409,
            detail={"error": "already_completed", "current_status": current.value},
        )

    # error 状态重跑：清理上次的脏数据
    if current == CampaignStatus.error:
        await dimension_repo.delete_by_campaign(campaign_id)
        await persona_repo.delete_by_campaign(campaign_id)
        await comment_repo.delete_by_campaign(campaign_id)

    # 有图片时首个状态是 processing_visual，否则直接 extracting_dimensions
    first_status = (
        CampaignStatus.processing_visual
        if campaign.get("image_url")
        else CampaignStatus.extracting_dimensions
    )
    await campaign_repo.update_status(campaign_id, first_status)
    background_tasks.add_task(run_pipeline, campaign_id)

    return {"campaign_id": campaign_id, "status": first_status.value, "message": "pipeline 已启动"}


@router.get("/{campaign_id}/status", response_model=CampaignStatusResponse)
async def get_status(campaign_id: str):
    campaign = _require_campaign(await campaign_repo.get(campaign_id), campaign_id)
    status = CampaignStatus(campaign["status"])
    progress = await _get_progress(campaign_id)

    return CampaignStatusResponse(
        campaign_id=campaign_id,
        status=status,
        progress=progress,
        stages_completed=_stages_completed(status),
        error=campaign.get("error_message"),
        started_at=campaign.get("started_at"),
        completed_at=campaign.get("completed_at"),
    )


@router.get("/{campaign_id}/result")
async def get_result(campaign_id: str):
    from config import settings

    campaign = _require_campaign(await campaign_repo.get(campaign_id), campaign_id)
    status = CampaignStatus(campaign["status"])

    if status != CampaignStatus.completed:
        raise HTTPException(
            status_code=425,
            detail={
                "error": "result_not_ready",
                "current_status": status.value,
                "hint": f"请轮询 /campaigns/{campaign_id}/status 直到 status=completed",
            },
        )

    dimensions = await dimension_repo.get_by_campaign(campaign_id)
    personas = await persona_repo.get_by_campaign(campaign_id)
    comments = await comment_repo.get_by_campaign(campaign_id)
    risk_summary = _compute_risk_summary(comments)

    # 清理 MongoDB _id 字段
    for doc in [*dimensions, *personas, *comments]:
        doc.pop("_id", None)

    return {
        "campaign_id": campaign_id,
        "campaign": {
            "title": campaign["title"],
            "description": campaign["description"],
            "visual_description": campaign.get("visual_description"),
        },
        "summary": campaign.get("summary"),
        "dimensions": dimensions,
        "personas": personas,
        "comments": comments,
        "risk_summary": risk_summary.model_dump(),
        "report_url": f"{settings.base_url}/campaigns/{campaign_id}/report",
    }


@router.get("/{campaign_id}/report", response_class=HTMLResponse)
async def get_report(campaign_id: str, request: Request):
    campaign = _require_campaign(await campaign_repo.get(campaign_id), campaign_id)
    status = CampaignStatus(campaign["status"])

    if status != CampaignStatus.completed:
        raise HTTPException(status_code=425, detail="报告尚未就绪")

    dimensions = await dimension_repo.get_by_campaign(campaign_id)
    personas = await persona_repo.get_by_campaign(campaign_id)
    comments = await comment_repo.get_by_campaign(campaign_id)
    risk_summary = _compute_risk_summary(comments)

    personas_map = {p["persona_id"]: p for p in personas}

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "campaign": campaign,
            "summary": campaign.get("summary"),
            "dimensions": dimensions,
            "personas_map": personas_map,
            "comments": comments,
            "risk_summary": risk_summary,
        },
    )

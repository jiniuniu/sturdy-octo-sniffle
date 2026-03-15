from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import settings
from db.repositories import study_repo, dimension_repo, persona_repo, response_repo
from models.study import (
    StudyCreate,
    StudyCreated,
    StudyDoc,
    StudyStatus,
    StudyStatusResponse,
    PipelineProgress,
    StudyDesign,
    StimulusType,
    RUNNING_STATUSES,
)
from pipeline.study_runner import run_study_pipeline
from storage.qiniu import upload_image

router = APIRouter(prefix="/studies", tags=["studies"])
templates = Jinja2Templates(directory="templates")


# ── 辅助函数 ─────────────────────────────────────────────────

def _compute_response_summary(responses: list[dict]) -> dict:
    if not responses:
        return {
            "stance_distribution": {},
            "signal_summary": {},
            "top_quotes": [],
        }

    stance_dist = dict(Counter(r["primary_stance"] for r in responses))

    # 汇总所有 signal 的高风险数量
    signal_counts: dict[str, int] = {}
    for r in responses:
        for signal in r.get("signals", []):
            if signal.get("level") == "高":
                key = signal["key"]
                signal_counts[key] = signal_counts.get(key, 0) + 1

    # 取负面/复杂立场的 key_quote
    priority_stances = {"负面", "复杂"}
    top_quotes = [
        {"persona_id": r["persona_id"], "stance": r["primary_stance"], "quote": r["key_quote"]}
        for r in responses
        if r["primary_stance"] in priority_stances
    ][:5]

    return {
        "stance_distribution": stance_dist,
        "signal_summary": signal_counts,
        "top_quotes": top_quotes,
    }


async def _get_progress(study_id: str) -> PipelineProgress:
    personas_total = await persona_repo.count_by_study(study_id)
    personas_completed = await persona_repo.count_completed_by_study(study_id)
    responses_total = await response_repo.count(study_id)
    return PipelineProgress(
        personas_completed=personas_completed,
        personas_total=personas_total,
        responses_completed=responses_total,
        responses_total=personas_total,
    )


def _stages_completed(status: StudyStatus) -> list[str]:
    order = [
        "processing_visual",
        "extracting_dimensions",
        "sampling",
        "generating_personas",
        "generating_responses",
        "completed",
    ]
    try:
        idx = order.index(status.value)
    except ValueError:
        return []
    return order[:idx]


def _require_study(study: dict | None, study_id: str) -> dict:
    if study is None:
        raise HTTPException(status_code=404, detail=f"Study {study_id} 不存在")
    return study


def _report_url(study_id: str) -> str:
    return f"{settings.base_url}/studies/{study_id}/report"


# ── 路由 ─────────────────────────────────────────────────────

@router.post("", status_code=202, response_model=StudyCreated)
async def create_study(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: str = Form(...),
    study_design: str = Form(...),        # JSON 字符串
    stimulus_type: str = Form(default="campaign"),
    context: str = Form(default=""),
    n_personas: int = Form(default=8, ge=4, le=16),
    target_market: str = Form(default="中国大陆"),
    callback_url: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    import json
    try:
        design = StudyDesign.model_validate(json.loads(study_design))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"study_design 格式错误: {e}")

    image_key, image_url = None, None
    if image is not None:
        data = await image.read()
        image_key, image_url = await upload_image(data, image.filename or "image.jpg")

    first_status = (
        StudyStatus.processing_visual if image_url else StudyStatus.extracting_dimensions
    )

    doc = StudyDoc(
        title=title,
        description=description,
        stimulus_type=StimulusType(stimulus_type),
        context=context,
        study_design=design,
        n_personas=n_personas,
        target_market=target_market,
        image_key=image_key,
        image_url=image_url,
        callback_url=callback_url,
        status=first_status,
    )
    study_id = await study_repo.create(doc)
    background_tasks.add_task(run_study_pipeline, study_id)

    return StudyCreated(
        study_id=study_id,
        status=first_status,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/{study_id}/status", response_model=StudyStatusResponse)
async def get_status(study_id: str):
    study = _require_study(await study_repo.get(study_id), study_id)
    status = StudyStatus(study["status"])
    progress = await _get_progress(study_id)

    return StudyStatusResponse(
        study_id=study_id,
        status=status,
        progress=progress,
        stages_completed=_stages_completed(status),
        error=study.get("error_message"),
        report_url=_report_url(study_id) if status == StudyStatus.completed else None,
        started_at=study.get("started_at"),
        completed_at=study.get("completed_at"),
    )


@router.get("/{study_id}/result")
async def get_result(study_id: str):
    study = _require_study(await study_repo.get(study_id), study_id)
    status = StudyStatus(study["status"])

    if status != StudyStatus.completed:
        raise HTTPException(
            status_code=425,
            detail={
                "error": "result_not_ready",
                "current_status": status.value,
                "hint": f"请轮询 /studies/{study_id}/status 直到 status=completed",
            },
        )

    dimensions = await dimension_repo.get_by_study(study_id)
    personas = await persona_repo.get_by_study(study_id)
    responses = await response_repo.get_by_study(study_id)
    response_summary = _compute_response_summary(responses)

    for doc in [*dimensions, *personas, *responses]:
        doc.pop("_id", None)

    return {
        "study_id": study_id,
        "study": {
            "title": study["title"],
            "description": study["description"],
            "stimulus_type": study.get("stimulus_type"),
            "context": study.get("context"),
            "study_design": study.get("study_design"),
            "visual_description": study.get("visual_description"),
        },
        "summary": study.get("summary"),
        "dimensions": [d for d in dimensions if d.get("source") != "hidden"],
        "personas": personas,
        "responses": responses,
        "response_summary": response_summary,
        "report_url": _report_url(study_id),
    }


@router.get("/{study_id}/report", response_class=HTMLResponse)
async def get_report(study_id: str, request: Request):
    study = _require_study(await study_repo.get(study_id), study_id)
    status = StudyStatus(study["status"])

    if status != StudyStatus.completed:
        raise HTTPException(status_code=425, detail="报告尚未就绪")

    dimensions = await dimension_repo.get_by_study(study_id)
    personas = await persona_repo.get_by_study(study_id)
    responses = await response_repo.get_by_study(study_id)
    response_summary = _compute_response_summary(responses)
    personas_map = {p["persona_id"]: p for p in personas}

    return templates.TemplateResponse(
        "study_report.html",
        {
            "request": request,
            "study": study,
            "summary": study.get("summary"),
            "dimensions": [d for d in dimensions if d.get("source") != "hidden"],
            "personas_map": personas_map,
            "responses": responses,
            "response_summary": response_summary,
        },
    )

# api/routes/validations.py
import asyncio
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from main import run_validation_async
from models.schemas import ProductInfo
from storage.job_repo import get_job_repo
from utils.export import build_stats, render_report
from utils.qiniu_uploader import upload_charts

router = APIRouter(prefix="/validations", tags=["validations"])


# ── 请求 / 响应模型 ────────────────────────────────────────────────────────────


class CreateValidationRequest(BaseModel):
    product: ProductInfo
    n_personas: int | None = Field(default=None, ge=4, le=64)


class CreateValidationResponse(BaseModel):
    job_id: str
    status: str
    created_at: str


# ── 后台任务 ──────────────────────────────────────────────────────────────────


async def _run_job(
    job_id: str,
    product: ProductInfo,
    n_personas: int | None,
) -> None:
    repo = get_job_repo()
    try:
        await repo.patch(job_id, {"status": "running"})

        matrix, report = await run_validation_async(
            product,
            n_personas=n_personas,
            job_id=job_id,
        )

        # 生成图表并上传七牛云（同步操作放线程池）
        charts = await asyncio.to_thread(upload_charts, matrix, report, job_id)

        # 渲染 Markdown（图表 URL 已嵌入；七牛未配置时 charts 为空 dict，图片链接为空）
        stats = build_stats(matrix)
        md = render_report(matrix, report, stats, charts)

        await repo.patch(
            job_id,
            {
                "charts": charts,
                "report_markdown": md,
                "status": "completed",
            },
        )
    except Exception as exc:
        await repo.patch(job_id, {"status": "failed", "error": str(exc)})
        raise


# ── 路由 ──────────────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_validation(
    body: CreateValidationRequest,
    background_tasks: BackgroundTasks,
):
    """
    提交一次产品验证任务。立即返回 job_id，后台异步执行完整 pipeline。
    """
    job_id = str(uuid4())
    repo = get_job_repo()
    await repo.create(job_id, body.product.model_dump())
    background_tasks.add_task(_run_job, job_id, body.product, body.n_personas)
    job = await repo.get(job_id)
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"].isoformat(),
    }


@router.get("/{job_id}")
async def get_validation(job_id: str):
    """
    查询 job 完整信息（含所有已完成阶段的中间数据）。
    前端按各字段是否为 null 决定渲染哪个阶段。
    """
    repo = get_job_repo()
    job = await repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # datetime 转字符串，方便 JSON 序列化
    for key in ("created_at", "updated_at"):
        if job.get(key):
            job[key] = job[key].isoformat()
    return job


@router.get("")
async def list_validations(page: int = 1, page_size: int = 20):
    """
    列出所有验证任务（分页，仅返回摘要，不含中间数据大对象）。
    """
    if page < 1:
        page = 1
    if not (1 <= page_size <= 100):
        page_size = 20
    repo = get_job_repo()
    items, total = await repo.list(page=page, page_size=page_size)
    for item in items:
        for key in ("created_at", "updated_at"):
            if item.get(key):
                item[key] = item[key].isoformat()
    return {"items": items, "total": total, "page": page, "page_size": page_size}

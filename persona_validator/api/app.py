# api/app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.validations import router as validations_router
from storage.client import close_client, get_db
from storage.job_repo import JobRepo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时建索引（幂等）
    await JobRepo(get_db()).create_indexes()
    yield
    # 关闭时释放 Motor 连接
    await close_client()


app = FastAPI(
    title="Persona Validator API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(validations_router, prefix="/api/v1")

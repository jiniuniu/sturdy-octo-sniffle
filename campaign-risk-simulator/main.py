import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from db.client import init_indexes, close_client
from api.studies import router as studies_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_indexes()
    yield
    await close_client()


app = FastAPI(
    title="Consumer Research Framework",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(studies_router)


@app.get("/health")
async def health():
    return {"status": "ok"}

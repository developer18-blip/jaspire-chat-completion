from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.routes import chat
from app.services.memory import memory

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting JASPIRE Chat API (OpenAI-compatible)")
    logger.info(f"VLLM URL : {settings.crawl4ai_llm_base_url}")
    logger.info(f"Model    : {settings.vllm_model_name}")
    logger.info(f"Memory DB: {memory.db_path}")
    logger.info(f"Endpoint : POST /v1/chat/completions  (streaming supported)")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="JASPIRE Chat API",
    description="OpenAI-compatible chat completions API with LangChain web-search agent",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )

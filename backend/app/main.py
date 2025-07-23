
from contextlib import asynccontextmanager
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated

from app.config import get_settings
from app.logging_config import setup_logging
from app.schemas import AskRequest
from app.service import RAGService, create_rag_service

settings = get_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service = create_rag_service(settings)
    rag_service.initialize(settings)
    app.state.rag_service = rag_service
    yield
    logger.info("애플리케이션이 정상적으로 종료되었습니다.")


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


app = FastAPI(lifespan=lifespan, title=get_settings().PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().all_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/comments/query")
async def ask_question(
    request: AskRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
):
    if not request.text:
        logger.warning("API 호출 시 본문이 누락되었습니다.")
        raise HTTPException(status_code=400, detail="본문이 없습니다.")

    logger.info(f"수신된 본문: {request.text}")
    try:
        answer = rag_service.ask(request.text)
        logger.info(f"생성된 댓글: {answer}")
        return JSONResponse(content={"answer": answer})
    except RuntimeError as e:
        logger.error(f"RAG 파이프라인이 준비되지 않았습니다: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {e}")


@app.get("/")
def read_root():
    return {"message": "Autogram LLM Backend is running."}
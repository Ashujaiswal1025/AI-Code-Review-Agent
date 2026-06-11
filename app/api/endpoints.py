from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import traceback

from app.services.agent import AIReviewAgent
from app.services.ingestion import ingest_repository

router = APIRouter()


class IngestRequest(BaseModel):
    repo_url: str


class IngestResponse(BaseModel):
    status: str
    message: str


class ChatRequest(BaseModel):
    message: str
    repo_name: str


class ChatResponse(BaseModel):
    response: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_repo_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    try:
        background_tasks.add_task(
            ingest_repository,
            request.repo_url,
        )

        return IngestResponse(
            status="accepted",
            message=(
                f"Started ingestion for "
                f"{request.repo_url} in background."
            ),
        )

    except Exception as e:
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
):
    try:
        agent = AIReviewAgent()

        response_text = await agent.process_message(
            request.message,
            request.repo_name,
        )

        return ChatResponse(
            response=response_text
        )

    except Exception as e:
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat: {str(e)}",
        )
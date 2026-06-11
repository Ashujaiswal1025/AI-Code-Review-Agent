from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="An AI-powered agent to review code and suggest improvements.",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME} API. Access /docs for the API documentation."}

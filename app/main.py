from fastapi import FastAPI
from app.router import router

app = FastAPI(
    title="LLM Assistant",
    description="Ответы на основе HTML базы знаний",
    version="1.0"
)

app.include_router(router)
from fastapi import APIRouter
from pydantic import BaseModel
from app.faq import find_answer_with_faq_and_ai

HTML_KNOWLEDGE_URL = "https://parks.yandex/ru-ru/moskva/knowledge-base/"

router = APIRouter()

class PromptRequest(BaseModel):
    query: str

def chunk_to_text(chunk: dict) -> str:
    heading = chunk.get("heading") or ""
    text = chunk.get("text") or ""
    return f"{heading} {text}".strip()

@router.post("/ask")
async def ask(payload: PromptRequest):
    query = payload.query

    answer = await find_answer_with_faq_and_ai(query)
    if answer:
        return {"answer": answer}
    else:
        return 'Информация не найдена в Базе знаний'
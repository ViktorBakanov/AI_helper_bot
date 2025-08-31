import json
import os
from app.openrouter_client import ask_openrouter
from sentence_transformers import SentenceTransformer, util
from app.model import ask_llm
from app.prompt import build_prompt
from app.similarity import match_score

# === Константы ===
FAQ_FILES = [
    os.path.join(os.path.dirname(__file__), "library_faq.json"),
    os.path.join(os.path.dirname(__file__), "custom_faq.json"),
]
STOP_WORDS = {
    'о', 'на', 'и', 'в', 'как', 'что', 'к', 'с', 'для', 'то', 'но', 'а', 'у', 'же', 'от', 'до', 'или', 'если'
}

# === NLP модель для семантического поиска ===
model = SentenceTransformer("intfloat/e5-large-v2")

# === Загрузка FAQ и эмбеддингов ===
def load_faq():
    faq_combined = []
    for path in FAQ_FILES:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    try:
                        faq_combined.extend(json.loads(content))
                    except json.JSONDecodeError as e:
                        print(f"[WARNING] Ошибка при чтении {path}: {e}")
                else:
                    print(f"[INFO] Файл пустой и пропущен: {path}")
        else:
            print(f"[INFO] Файл не найден: {path}")
    return faq_combined

faq = load_faq()
faq_texts = [f"{entry['question']} {entry['answer']}" for entry in faq]
faq_embeddings = model.encode(faq_texts, convert_to_tensor=True)

# === Утилиты ===

def format_entry_with_score(entry, score):
    confidence = round(score * 100)
    prefix = "Источник (достоверность: {}%):".format(confidence)
    return f"{prefix}\nВопрос: {entry['question']}\nОтвет: {entry['answer']}"

# === Семантический поиск по FAQ ===
def search_faq_semantic(query: str, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, faq_embeddings, top_k=top_k)[0]

    print("=== SEMANTIC MATCH ===")
    for hit in hits:
        print(f"score: {hit['score']:.3f} → {faq[hit['corpus_id']]['question']}")

    return [(hit['score'], faq[hit['corpus_id']]) for hit in hits if hit['score'] > 0.3]

# === Хинтовый (подсказочный) поиск + эвристика на точный заголовок ===
def find_relevant_faq_entries_with_hint(query: str, faq_list: list, top_k=3):
    query_lower = query.strip().lower()

    # Точное совпадение заголовка — возвращаем сразу
    for entry in faq_list:
        q = entry['question'].strip().lower()
        if query_lower in q or q in query_lower:
            print(f"[DEBUG] Почти точное совпадение: {query_lower} ↔ {q}")
            return [("", entry['answer'])]

    # Ищем близкие по классическому score (эвристика)
    scored = []
    for entry in faq_list:
        combined_text = f"{entry.get('question', '')} {entry.get('answer', '')}"
        score = match_score(combined_text, query)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Добавляем подсказки
    relevant = []
    for score, entry in scored[:top_k]:
        hint = f"Возможно вы хотите узнать: «{entry['question']}». Вот ответ на вашу тему:\n"
        relevant.append((hint, entry['answer']))

    print("=== RELEVANT ENTRIES ===")
    for hint, text in relevant:
        print(hint)
        print(text[:300])
    return relevant

# === Финальный метод — основной вызов ассистента ===
async def find_answer_with_faq_and_ai(query: str, use_semantic: bool = True):
    print(f"[DEBUG] Query: {query}")
    RELIABLE_THRESHOLD = 0.70  # Порог для надёжного источника

    query_lower = query.strip().lower()

    for entry in faq:
        q = entry['question'].strip().lower()
        if query_lower == q or query_lower in q or q in query_lower:
            print(f"[DEBUG] Точное совпадение заголовка FAQ: {q}")
            return entry['answer']

    if use_semantic:
        top_entries = search_faq_semantic(query, top_k=5)
        if not top_entries:
            return "К сожалению, я не обладаю данной информацией."
        
        # Если первое совпадение — прямое, возвращаем без LLM
        if query.strip().lower() == top_entries[0][1]['question'].strip().lower():
            return top_entries[0][1]['answer']
        
        reliable = []
        additional = []

        for score, entry in top_entries:
            if score >= RELIABLE_THRESHOLD:
                reliable.append((score, entry))
            else:
                additional.append((score, entry))
        
        # Если есть хотя бы один надёжный — используем его
        context_parts = []

        if reliable:
            best = reliable[0][1]
            context_parts.append(
                f"Надёжный источник:\nЕсли вы интересуетесь, {best['question'].lower()}, вот ответ:\n{best['answer']}"
            )

        # Добавим до 2 дополнительных источников
        if additional:
            context_parts.append("Дополнительная информация:")
            for _, entry in additional[:2]:
                context_parts.append(f"– {entry['question']}\n  {entry['answer']}")

        context = "\n\n".join(context_parts)
    else:
        relevant_entries = find_relevant_faq_entries_with_hint(query, faq)
        if not relevant_entries:
            return "К сожалению, я не обладаю данной информацией."

        # Если точный заголовок
        if relevant_entries[0][0] == "":
            return relevant_entries[0][1]

        parts = [hint + text for hint, text in relevant_entries]
        context = "\n\n".join(parts)

    # Улучшаем ответ через LLM
    prompt = build_prompt(context, query)
    print("=== FINAL PROMPT ===")
    print(prompt)
    #return ask_llm(prompt)
    return await llm_request(prompt)
    
async def llm_request(prompt: str) -> str:
    try:
        return await ask_openrouter(prompt)
    except Exception as e:
        return f"Ошибка при обращении к LLM: {e}"
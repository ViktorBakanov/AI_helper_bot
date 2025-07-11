try:
    from llama_cpp import Llama
except ImportError:
    Llama = None 
import os
import re

MODEL_PATH = "./models/llama3.Q4_K_M.gguf"

# Подключаем Phi-3 Mini через gguf
n_threads = os.cpu_count()
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    use_mlock=True,
    low_ram=True
)

# Удаляем лишние фразы после ответа
def clean_answer(text: str) -> str:
    # Удаляем всё после ненужных фрагментов
    splitters = [
        "response:", "question:", "answer:", "Iterate",
        "Here is the answer", "Ответ:", "===", "</s>"
    ]
    for splitter in splitters:
        idx = text.lower().find(splitter.lower())
        if idx != -1:
            text = text[:idx]

    # Убираем возможный префикс "- teacher:" или подобный мусор в начале
    text = text.strip()
    if text.lower().startswith("- teacher:"):
        text = text[len("- teacher:"):].strip()

    return re.sub(r'[^\u0000-\u007F\u0400-\u04FF\s.,!?()\[\]\"\'\-–—:;]', '', text)

def ask_llm(prompt: str) -> str:
    output = llm(prompt, max_tokens=256, stop=None)
    raw_text = output["choices"][0]["text"].strip()
    return clean_answer(raw_text)
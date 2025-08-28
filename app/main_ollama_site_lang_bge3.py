from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import json
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer
import time
import re
import os
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Настройки ----------------
INDEX_DIR = "embeddings_bge3"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
CHUNKED_FILE = "data/chunked_data.json"

EMBED_MODEL = "BAAI/bge-m3"
USE_GPU = False
LLM_MODEL = "llama3"
TOP_K = 7
MAX_CONTEXT_CHARS = 20000
OLLAMA_TIMEOUT = 600
LOG_SNIPPET_LEN = 200
# --------------------------------------------

print("[Инициализация] Загружаем FAISS и модель эмбеддингов (BGE3)...")
start = time.time()

index = faiss.read_index(INDEX_FILE)
with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
    docs: List[Dict] = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL, device="cuda" if USE_GPU else "cpu")
print(f"[Готово] Запуск занял {time.time() - start:.2f} сек.")

app = FastAPI(title="AI Assistant (BGE3)")

# ✅ Разрешаем CORS для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно ограничить ["http://127.0.0.1:5500", "https://ai-assistant-demo.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

# ---------------- utils ----------------
def detect_language(text: str) -> str:
    t = text.lower()
    if any(c in t for c in "әғқңөұүі"):
        return "kk"
    if re.search(r"[а-яё]", t):
        return "ru"
    return "en"

def embed_query(text: str):
    q = f"query: {text}"
    v = embed_model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    return v

def retrieve_context(question: str, lang: str, top_k: int = TOP_K):
    q_vec = embed_query(question)
    D, I = index.search(np.array(q_vec), top_k * 5)

    def url_ok(url: str):
        if any(x in url for x in ["vacancies", "tenders", "/search/"]):
            return False
        if lang == "ru":
            return "/ru/" in url
        if lang == "kk":
            return "/kz/" in url or "/kk/" in url
        return "/ru/" in url

    results = []
    for i in I[0]:
        if i == -1:
            continue
        doc = docs[i]
        url = doc.get("url", "")
        if url_ok(url):
            results.append(doc)
        if len(results) >= top_k:
            break
    return results

def ask_ollama(prompt: str) -> str:
    print(f"[PROMPT] >>>\n{prompt}\n<<<")
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=OLLAMA_TIMEOUT
        )
        output = result.stdout.decode("utf-8").strip()
        print(f"[OLLAMA OUTPUT] >>>\n{output}\n<<<")
        return output
    except Exception as e:
        return f"Ошибка Ollama: {e}"

# ---------------- API ----------------
@app.post("/ask")
def ask_bot(query: Query):
    lang = detect_language(query.question)

    lang_hint = {
        "ru": (
            "Отвечай ТОЛЬКО на русском языке, строго по приведённому тексту. "
            "Используй только факты из этого текста. Не добавляй ничего от себя."
        ),
        "kk": (
            "Тек мәтін бойынша, тек нақты деректермен қазақ тілінде жауап бер. "
            "Атауларды аударма және өзгертуге болмайды."
        ),
        "en": (
            "Answer strictly in English, based ONLY on the provided text. "
            "Do not add anything extra."
        )
    }.get(lang, "Answer strictly and only from the provided text.")

    context_items = retrieve_context(query.question, lang, TOP_K)

    if not context_items or sum(len(c["content"]) for c in context_items) < 300:
        fallback_msg = {
            "ru": "Извините, точная информация не найдена.",
            "kk": "Кешіріңіз, нақты ақпарат табылмады.",
            "en": "Sorry, we couldn't find a precise answer."
        }.get(lang, "Information not found.")
        return {"answer": fallback_msg}

    print("[DEBUG] ВЫБРАННЫЕ ДОКУМЕНТЫ:")
    for i, c in enumerate(context_items, 1):
        snippet = c["content"][:LOG_SNIPPET_LEN].replace("\n", " ")
        print(f" {i}. {c.get('url', '')} [{len(c['content'])} chars] :: {snippet}...")

    context = "\n\n---\n\n".join([c["content"][:MAX_CONTEXT_CHARS] for c in context_items])
    used_urls = [c.get("url", "") for c in context_items]

    prompt = f"""{lang_hint}

Контекст:
{context}

Вопрос:
{query.question}

Ответ:"""

    answer = ask_ollama(prompt)
    answer = answer.replace("\n", " ").replace("*", "").replace("\\", "").strip()

    return {
        "answer": answer,
        "used_urls": used_urls
    }

@app.get("/health")
def health():
    return {"status": "ok"}

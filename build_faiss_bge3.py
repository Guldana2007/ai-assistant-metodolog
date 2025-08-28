import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from docx import Document

# Папка с исходными Word-документами
DOCS_DIR = "ВНД"
# Папка для сохранения эмбеддингов и индекса
OUT_DIR = "embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

# Модель эмбеддингов
model = SentenceTransformer("BAAI/bge-m3")

# Функция: извлекаем текст из .docx
def read_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text

# Чанкирование текста
def chunk_text(text, max_len=1200, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start += max_len - overlap
    return chunks

# Загружаем документы
all_chunks = []
for fname in os.listdir(DOCS_DIR):
    if fname.endswith(".docx"):
        path = os.path.join(DOCS_DIR, fname)
        text = read_docx(path)
        chunks = chunk_text(text)
        for c in chunks:
            all_chunks.append({"source": fname, "text": c})

print(f"Всего чанков: {len(all_chunks)}")

# Эмбеддинги
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# FAISS индекс
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Сохраняем
faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(OUT_DIR, "data.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("✅ Индекс успешно построен и сохранён в папку embeddings/")

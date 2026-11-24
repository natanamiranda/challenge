
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.rag import (
    VECTOR_DB_DIR,
    COLLECTION_NAME,
)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from medical_assistant.config import settings


embeddings = OllamaEmbeddings(
    model=settings.ollama_embedding_model,
    base_url=settings.ollama_base_url,
)

db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DB_DIR),
)

print("Chunks indexados:", db._collection.count())

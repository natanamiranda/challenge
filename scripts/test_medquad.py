
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.rag import (
    load_medquad_documents,
    split_medquad_documents,
)

if __name__ == "__main__":
    documents = load_medquad_documents()
    chunks = split_medquad_documents(documents)

    print(f"Documentos carregados: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")

    print("\nExemplo de documento:")
    print(documents[0].page_content[:1000])

    print("\nMetadados:")
    print(documents[0].metadata)

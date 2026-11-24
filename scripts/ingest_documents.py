
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.rag import (
    load_medquad_documents,
    split_medquad_documents,
    create_embeddings,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
)

from langchain_chroma import Chroma


def main():
    print("Carregando 300 documentos MedQuAD...")

    documents = load_medquad_documents(limit=300)
    chunks = split_medquad_documents(documents)

    print(f"Documentos carregados: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")

    print("\nRemovendo a base vetorial anterior...")

    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    batch_size = 16

    print("\nIniciando indexação...")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        ids = [
            f"medquad_{start + index}"
            for index in range(len(batch))
        ]

        vector_store.add_documents(
            documents=batch,
            ids=ids,
        )

        processed = min(start + batch_size, len(chunks))
        print(f"Indexados: {processed}/{len(chunks)}")

    print("\nIndexação concluída!")
    print(f"Total indexado: {vector_store._collection.count()}")


if __name__ == "__main__":
    main()

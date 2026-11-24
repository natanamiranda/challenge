
import json
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from medical_assistant.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEDQUAD_FILE = PROJECT_ROOT / "data" / "processed" / "medquad.jsonl"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "medical_knowledge"


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", text).strip()


def load_medquad_documents(limit=None):
    if not MEDQUAD_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo MedQuAD não encontrado: {MEDQUAD_FILE}"
        )

    documents = []

    with MEDQUAD_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(documents) >= limit:
                break

            if not line.strip():
                continue

            item = json.loads(line)

            answer = (
                item.get("answer")
                or item.get("Answer")
                or item.get("content")
                or ""
            )

            if not answer.strip():
                continue

            metadata = {
                "source": str(item.get("source", "MedQuAD")),
                "source_url": str(item.get("source_url", "")),
                "focus": str(item.get("focus", "")),
                "category": str(item.get("category", "")),
                "question_id": str(item.get("question_id", "")),
                "question_type": str(item.get("question_type", "")),
                "document_id": str(item.get("document_id", "")),
                "source_file": str(item.get("source_file", "")),
                "question": str(item.get("question", "")),
            }

            documents.append(
                Document(
                    page_content=answer,
                    metadata=metadata,
                )
            )

    return documents


def split_medquad_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
    )

    return splitter.split_documents(documents)


def create_embeddings():
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vector_store():
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            "Base vetorial não encontrada. "
            "Execute primeiro o script de indexação."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(VECTOR_DB_DIR),
    )


def detect_question_type(question: str) -> str:
    normalized = normalize_text(question)

    symptom_terms = [
        "sintoma",
        "sintomas",
        "manifestacoes",
        "sinais",
    ]

    if any(term in normalized for term in symptom_terms):
        return "symptoms"

    return "information"


def translate_search_query(question: str) -> str:
    normalized = normalize_text(question)

    if (
        "leucemia linfoblastica aguda" in normalized
        or re.search(r"\blla\b", normalized)
    ):
        if any(
            term in normalized
            for term in ["sintoma", "sintomas", "sinais"]
        ):
            return (
                "Adult Acute Lymphoblastic Leukemia "
                "symptoms signs clinical presentation"
            )

        return (
            "Adult Acute Lymphoblastic Leukemia "
            "definition information overview"
        )

    return question


def get_documents_by_focus(documents, focus):
    normalized_focus = normalize_text(focus)

    return [
        document
        for document in documents
        if normalized_focus
        in normalize_text(document.metadata.get("focus", ""))
    ]


def extract_part_number(content: str):
    match = re.search(
        r"\bpart\s*(\d+)\b",
        content,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return 1


def rank_documents(documents, question: str):
    requested_type = detect_question_type(question)

    matching_type = [
        document
        for document in documents
        if normalize_text(
            document.metadata.get("question_type", "")
        ) == requested_type
    ]

    if matching_type:
        documents = matching_type

    def score(document):
        metadata = document.metadata

        question_type = normalize_text(
            metadata.get("question_type", "")
        )

        document_question = normalize_text(
            metadata.get("question", "")
        )

        document_focus = normalize_text(
            metadata.get("focus", "")
        )

        normalized_query = normalize_text(question)

        points = 0

        if question_type == requested_type:
            points += 100

        if "adult acute lymphoblastic leukemia" in document_focus:
            points += 30

        if (
            "sintoma" in normalized_query
            and "symptom" in document_question
        ):
            points += 50

        if (
            "o que e" in normalized_query
            and question_type == "information"
        ):
            points += 30

        if extract_part_number(document.page_content) == 1:
            points += 10

        return points

    return sorted(
        documents,
        key=score,
        reverse=True,
    )


def remove_duplicate_documents(documents):
    unique_documents = []
    seen = set()

    for document in documents:
        metadata = document.metadata

        identity = (
            metadata.get("question_id")
            or metadata.get("document_id")
            or metadata.get("question")
        )

        part_number = extract_part_number(
            document.page_content
        )

        unique_key = (identity, part_number)

        if unique_key in seen:
            continue

        seen.add(unique_key)
        unique_documents.append(document)

    return unique_documents


def search_protocols(question: str, k: int = 3):
    if not question or not question.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    vector_store = get_vector_store()

    search_query = translate_search_query(question)

    results = vector_store.similarity_search(
        search_query,
        k=15,
    )

    if not results:
        return []

    focus_documents = get_documents_by_focus(
        results,
        "Adult Acute Lymphoblastic Leukemia",
    )

    if focus_documents:
        results = focus_documents

    ranked_documents = rank_documents(
        results,
        question,
    )

    unique_documents = remove_duplicate_documents(
        ranked_documents
    )

    return unique_documents[:k]
    
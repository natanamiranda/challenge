import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "medquad.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "finetuning"

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
REPORT_FILE = OUTPUT_DIR / "dataset_report.json"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.10
MAX_QUESTION_LENGTH = 1000
MAX_ANSWER_LENGTH = 8000

INSTRUCTION = (
    "Responda à pergunta sobre saúde utilizando exclusivamente "
    "as informações médicas fornecidas. Seja claro, objetivo e "
    "não invente informações. Não substitua a avaliação de um "
    "profissional de saúde."
)


def load_records():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset original não encontrado: {INPUT_FILE}"
        )

    records = []
    seen = set()
    rejected = 0

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue

            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()

            if not question or not answer:
                rejected += 1
                continue

            if (
                len(question) > MAX_QUESTION_LENGTH
                or len(answer) > MAX_ANSWER_LENGTH
            ):
                rejected += 1
                continue

            unique_key = (
                question.casefold(),
                answer.casefold(),
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)

            focus = str(item.get("focus") or "").strip()

            training_item = {
                "instruction": INSTRUCTION,
                "input": (
                    f"Tema: {focus}\n"
                    f"Pergunta: {question}"
                ),
                "output": answer,
                "metadata": {
                    "source": item.get("source"),
                    "source_url": item.get("source_url"),
                    "question_type": item.get("question_type"),
                    "document_id": item.get("document_id"),
                },
            }

            records.append(training_item)

    return records, rejected


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparando dataset para fine-tuning...")

    records, rejected = load_records()

    if len(records) < 10:
        raise ValueError(
            "Quantidade insuficiente de exemplos válidos."
        )

    random.Random(RANDOM_SEED).shuffle(records)

    validation_size = max(
        1,
        int(len(records) * VALIDATION_RATIO),
    )

    validation_records = records[:validation_size]
    train_records = records[validation_size:]

    save_jsonl(TRAIN_FILE, train_records)
    save_jsonl(VALIDATION_FILE, validation_records)

    report = {
        "dataset": "MedQuAD",
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "total_examples": len(records),
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "rejected_records": rejected,
        "validation_ratio": VALIDATION_RATIO,
        "random_seed": RANDOM_SEED,
        "format": "instruction_input_output",
        "source_preserved_in_metadata": True,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n=== DATASET PREPARADO ===")
    print(f"Total de exemplos: {len(records)}")
    print(f"Treinamento: {len(train_records)}")
    print(f"Validação: {len(validation_records)}")
    print(f"Registros rejeitados: {rejected}")
    print(f"\nArquivos gerados:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()

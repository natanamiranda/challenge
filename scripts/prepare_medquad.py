
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medquad"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "medquad.jsonl"


def clean_text(element):
    if element is None:
        return ""

    return " ".join(
        " ".join(element.itertext()).split()
    )


def find_child(element, tag_name):
    for child in element.iter():
        if child.tag.split("}")[-1] == tag_name:
            return child
    return None


def process_xml(file_path):
    try:
        root = ET.parse(file_path).getroot()
    except ET.ParseError:
        return []

    source = root.attrib.get("source", "unknown")
    source_url = root.attrib.get("url", "")
    document_id = root.attrib.get("id", "")

    focus = clean_text(find_child(root, "Focus"))
    category = clean_text(find_child(root, "Category"))

    qa_pairs = []

    for pair in root.iter():
        if pair.tag.split("}")[-1] != "QAPair":
            continue

        question_element = find_child(pair, "Question")
        answer_element = find_child(pair, "Answer")

        question = clean_text(question_element)
        answer = clean_text(answer_element)

        if not question or len(answer) < 20:
            continue

        question_type = ""
        if question_element is not None:
            question_type = question_element.attrib.get("qtype", "")

        question_id = ""
        if question_element is not None:
            question_id = question_element.attrib.get("qid", "")

        qa_pairs.append({
            "content": (
                f"Tema: {focus}\n"
                f"Categoria: {category}\n"
                f"Pergunta: {question}\n"
                f"Resposta: {answer}"
            ),
            "question": question,
            "answer": answer,
            "focus": focus,
            "category": category,
            "source": source,
            "source_url": source_url,
            "document_id": document_id,
            "question_id": question_id,
            "question_type": question_type,
            "source_file": file_path.name,
        })

    return qa_pairs


def main():
    if not RAW_DIR.exists():
        print(f"Diretório não encontrado: {RAW_DIR}")
        sys.exit(1)

    xml_files = list(RAW_DIR.rglob("*.xml"))

    print(f"Arquivos XML encontrados: {len(xml_files)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    sources = Counter()

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for index, file_path in enumerate(xml_files, start=1):
            records = process_xml(file_path)

            for record in records:
                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                sources[record["source"]] += 1
                total += 1

            if index % 1000 == 0:
                print(f"Processados: {index}/{len(xml_files)}")

    print("\nPreparação concluída!")
    print(f"Respostas válidas: {total}")
    print(f"Arquivo gerado: {OUTPUT_FILE}")

    print("\nRegistros por fonte:")
    for source, count in sources.most_common():
        print(f"- {source}: {count}")


if __name__ == "__main__":
    main()

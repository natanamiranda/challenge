
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.rag import search_protocols


def main():
    question = (
        "Quais são os sintomas da leucemia "
        "linfoblástica aguda em adultos?"
    )

    print("=" * 70)
    print("TESTE DE BUSCA RAG - MEDQUAD")
    print("=" * 70)

    print(f"\nPergunta: {question}")
    print("\nBuscando documentos relevantes...\n")

    results = search_protocols(question, k=5)

    if not results:
        print("Nenhum documento relevante foi encontrado.")
        return

    for index, document in enumerate(results, start=1):
        metadata = document.metadata

        print("\n" + "=" * 70)
        print(f"RESULTADO {index}")
        print("=" * 70)

        print("Tema:", metadata.get("focus", "Não informado"))
        print("Tipo:", metadata.get("question_type", "Não informado"))
        print("Fonte:", metadata.get("source", "Não informada"))
        print("Pergunta original:", metadata.get("question", "Não informada"))

        print("\nConteúdo:")
        print(document.page_content[:2500])

        print("\nURL:", metadata.get("source_url", "Não informada"))

    print("\n" + "=" * 70)
    print(f"TOTAL DE DOCUMENTOS RECUPERADOS: {len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    main()


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.rag import search_protocols


TEST_CASES = [
    {
        "question": "O que é leucemia linfoblástica aguda em adultos?",
        "expected_terms": [
            "adult acute lymphoblastic leukemia",
            "lymphocytes",
            "bone marrow",
        ],
    },
    {
        "question": (
            "Quais são os sintomas da leucemia "
            "linfoblástica aguda em adultos?"
        ),
        "expected_terms": [
            "fever",
            "tired",
            "bleeding",
        ],
    },
]


def main():
    print("=" * 60)
    print("TESTE DE QUALIDADE DA BUSCA RAG")
    print("=" * 60)

    for index, test in enumerate(TEST_CASES, start=1):
        print(f"\nTESTE {index}")
        print("-" * 60)

        question = test["question"]

        print("Pergunta:", question)
        print("\nBuscando documentos relevantes...")

        documents = search_protocols(question, k=3)

        if not documents:
            print("\nNenhum documento foi recuperado.")
            continue

        for result_index, document in enumerate(documents, start=1):
            print("\n" + "=" * 60)
            print(f"DOCUMENTO RECUPERADO {result_index}")
            print("=" * 60)

            print("Tema:", document.metadata.get("focus"))
            print("Categoria:", document.metadata.get("category"))
            print("Tipo de pergunta:", document.metadata.get("question_type"))
            print("Pergunta original:", document.metadata.get("question"))

            print("\nConteúdo recuperado:")
            print(document.page_content[:1200])

            content = document.page_content.lower()

            found_terms = [
                term
                for term in test["expected_terms"]
                if term in content
            ]

            print("\nTermos esperados encontrados:", found_terms)

        print("\nTotal de documentos recuperados:", len(documents))

    print("\n" + "=" * 60)
    print("TESTE DE BUSCA CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()

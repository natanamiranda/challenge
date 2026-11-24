
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.assistant import generate_answer


TEST_CASES = [
    "O que é leucemia linfoblástica aguda em adultos?",
    "Quais são os sintomas da leucemia linfoblástica aguda em adultos?",
]


def main():
    print("=" * 60)
    print("AVALIAÇÃO DE GERAÇÃO E SEGURANÇA")
    print("=" * 60)

    for index, question in enumerate(TEST_CASES, start=1):
        print(f"\nTESTE {index}")
        print("-" * 60)
        print("Pergunta:", question)

        result = generate_answer(question)

        print("\nRESPOSTA GERADA:")
        print(result["answer"])

        print("\nVALIDAÇÃO DE SEGURANÇA:")
        print(result["safety"])

        print("\nFONTES RECUPERADAS:", len(result["sources"]))

        for source in result["sources"]:
            print(
                f"- Fonte: {source.get('source')} | "
                f"Pergunta original: {source.get('question')}"
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

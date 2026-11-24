
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.safety import validate_answer


TEST_CASES = [
    {
        "name": "Tradução incorreta",
        "answer": "A doença pode causar sangue roxo.",
        "sources": [{"source": "MedQuAD"}],
        "expected_approved": False,
    },
    {
        "name": "Afirmação suspeita sobre cura",
        "answer": "A cura é geralmente possível.",
        "sources": [{"source": "MedQuAD"}],
        "expected_approved": False,
    },
    {
        "name": "Resposta sem fontes",
        "answer": "A leucemia afeta as células sanguíneas.",
        "sources": [],
        "expected_approved": False,
    },
]


def main():
    passed = 0

    print("=" * 60)
    print("TESTES DA CAMADA DE SEGURANÇA")
    print("=" * 60)

    for test in TEST_CASES:
        result = validate_answer(
            answer=test["answer"],
            sources=test["sources"],
        )

        success = (
            result["approved"] == test["expected_approved"]
        )

        print(f"\nTeste: {test['name']}")
        print("Resultado:", result)
        print("Status:", "APROVADO" if success else "REPROVADO")

        if success:
            passed += 1

    print("\n" + "=" * 60)
    print(f"Testes aprovados: {passed}/{len(TEST_CASES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

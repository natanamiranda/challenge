
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_assistant.assistant import generate_answer


def main():
    question = "What is adult acute lymphoblastic leukemia?"

    print("=" * 60)
    print("TESTE DO ASSISTENTE MEDICO")
    print("=" * 60)

    print("\nPergunta:", question)
    print("\nGerando resposta com Llama 3.2...\n")

    result = generate_answer(question)

    print("=" * 60)
    print("RESPOSTA")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("FONTES")
    print("=" * 60)

    for source in result["sources"]:
        print("\nFonte:", source["source"])
        print("URL:", source["url"])


if __name__ == "__main__":
    main()


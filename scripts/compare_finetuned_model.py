
import gc
import json
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_DIR = PROJECT_ROOT / "outputs" / "finetuning" / "smollm2-lora"
REPORT_DIR = PROJECT_ROOT / "outputs" / "finetuning"
REPORT_FILE = REPORT_DIR / "comparison_report.json"

QUESTION = (
    "What are the common signs and symptoms of acute "
    "lymphoblastic leukemia in adults?"
)

MAX_NEW_TOKENS = 160


def generate_answer(model, tokenizer):
    messages = [
        {
            "role": "user",
            "content": QUESTION,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][input_ids.shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def main():
    torch.set_num_threads(4)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError("Adaptador LoRA não encontrado.")

    print("=" * 65)
    print("COMPARAÇÃO: MODELO ORIGINAL X MODELO FINE-TUNED")
    print("=" * 65)

    print("\nCarregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print("\n1. Carregando modelo original...")
    original_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    original_model.to("cpu")
    original_model.eval()

    print("Gerando resposta do modelo original...")
    original_answer = generate_answer(original_model, tokenizer)

    del original_model
    gc.collect()

    print("\n2. Carregando modelo com adaptador LoRA...")
    finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        dtype=torch.float32,
    )

    finetuned_model.to("cpu")
    finetuned_model.eval()

    print("Gerando resposta do modelo fine-tuned...")
    finetuned_answer = generate_answer(finetuned_model, tokenizer)

    del finetuned_model
    gc.collect()

    report = {
        "question": QUESTION,
        "base_model": MODEL_NAME,
        "fine_tuning_method": "LoRA",
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "device": "cpu",
        },
        "original_model": {
            "answer": original_answer,
        },
        "finetuned_model": {
            "answer": finetuned_answer,
        },
        "evaluation_note": (
            "Comparação textual inicial. "
            "As respostas precisam de avaliação factual "
            "com fontes médicas; geração bem-sucedida "
            "não comprova melhoria clínica."
        ),
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 65)
    print("RESPOSTA DO MODELO ORIGINAL")
    print("=" * 65)
    print(original_answer or "Nenhuma resposta gerada.")

    print("\n" + "=" * 65)
    print("RESPOSTA DO MODELO FINE-TUNED")
    print("=" * 65)
    print(finetuned_answer or "Nenhuma resposta gerada.")

    print("\n" + "=" * 65)
    print("COMPARAÇÃO FINALIZADA")
    print("=" * 65)
    print("Relatório salvo em:", REPORT_FILE)


if __name__ == "__main__":
    main()

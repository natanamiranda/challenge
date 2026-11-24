
from pathlib import Path

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = PROJECT_ROOT / "outputs" / "finetuning" / "smollm2-lora"

QUESTION = (
    "What are the common signs and symptoms of acute "
    "lymphoblastic leukemia in adults?"
)


def main():
    print("=" * 60)
    print("TESTE DE INFERÃŠNCIA - MODELO FINE-TUNED LoRA")
    print("=" * 60)

    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError("ConfiguraÃ§Ã£o do adaptador nÃ£o encontrada.")

    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        raise FileNotFoundError("Pesos do adaptador nÃ£o encontrados.")

    torch.set_num_threads(4)

    print("\nCarregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print("Carregando modelo-base com adaptador LoRA...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        torch_dtype=torch.float32,
    )

    model.eval()
    model.to("cpu")

    messages = [{"role": "user", "content": QUESTION}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("\nPergunta:")
    print(QUESTION)
    print("\nGerando resposta...\n")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][input_ids.shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    print("=" * 60)
    print("RESPOSTA DO MODELO")
    print("=" * 60)
    print(answer if answer else "O modelo nÃ£o gerou texto.")

    print("\nAdaptador carregado: OK")
    print("InferÃªncia executada: OK")
    print("Resposta gerada:", "Sim" if answer else "NÃ£o")


if __name__ == "__main__":
    main()

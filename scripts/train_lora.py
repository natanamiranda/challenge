
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "finetuning" / "train.jsonl"
VALIDATION_FILE = PROJECT_ROOT / "data" / "finetuning" / "validation.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetuning"
ADAPTER_DIR = OUTPUT_DIR / "smollm2-lora"
REPORT_FILE = OUTPUT_DIR / "training_report.json"

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

MAX_TRAIN_SAMPLES = 128
MAX_VALIDATION_SAMPLES = 32
MAX_LENGTH = 256
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
RANDOM_SEED = 42


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_prompt(item):
    return (
        "### Instrução:\n"
        + item["instruction"]
        + "\n\n### Contexto:\n"
        + item["input"]
        + "\n\n### Resposta:\n"
    )


class MedicalDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.examples = []

        for item in records:
            prompt_ids = tokenizer(
                build_prompt(item),
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LENGTH - 16,
            )["input_ids"]

            answer_ids = tokenizer(
                str(item["output"]).strip(),
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LENGTH - len(prompt_ids) - 1,
            )["input_ids"]

            answer_ids.append(tokenizer.eos_token_id)

            self.examples.append({
                "input_ids": prompt_ids + answer_ids,
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id):
    max_size = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:
        padding = max_size - len(item["input_ids"])

        input_ids.append(
            item["input_ids"] + [pad_token_id] * padding
        )
        attention_masks.append(
            [1] * len(item["input_ids"]) + [0] * padding
        )
        labels.append(
            item["labels"] + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            total_loss += model(**batch).loss.item()

    model.train()
    return total_loss / len(loader)


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(4)

    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists():
        raise FileNotFoundError("Dataset de fine-tuning não encontrado.")

    train_records = load_jsonl(TRAIN_FILE)
    validation_records = load_jsonl(VALIDATION_FILE)

    random.Random(RANDOM_SEED).shuffle(train_records)
    random.Random(RANDOM_SEED).shuffle(validation_records)

    train_records = train_records[:MAX_TRAIN_SAMPLES]
    validation_records = validation_records[:MAX_VALIDATION_SAMPLES]

    print("Carregando tokenizer:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer sem token EOS.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Carregando modelo-base...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to("cpu")

    train_dataset = MedicalDataset(train_records, tokenizer)
    validation_dataset = MedicalDataset(validation_records, tokenizer)

    collate = lambda batch: collate_batch(
        batch,
        tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    start_time = time.time()
    train_losses = []
    validation_losses = []

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        print(f"Iniciando época {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to("cpu")
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            total_loss += loss.item()

            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"Etapa {step}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        train_loss = total_loss / len(train_loader)
        validation_loss = evaluate(
            model,
            validation_loader,
            "cpu",
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(f"Loss treinamento: {train_loss:.4f}")
        print(f"Loss validação: {validation_loss:.4f}")

    elapsed = time.time() - start_time

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    report = {
        "status": "completed",
        "base_model": MODEL_NAME,
        "method": "LoRA",
        "device": "cpu",
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "epochs": EPOCHS,
        "max_sequence_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "training_loss": train_losses,
        "validation_loss": validation_losses,
        "elapsed_seconds": round(elapsed, 2),
        "adapter_path": str(ADAPTER_DIR.relative_to(PROJECT_ROOT)),
        "seed": RANDOM_SEED,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print("Treinamento concluído.")
    print("Adaptador:", ADAPTER_DIR)
    print("Relatório:", REPORT_FILE)


if __name__ == "__main__":
    main()

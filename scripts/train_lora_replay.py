import json
import math
import random
import yaml
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm


class AbstractDataset(Dataset):
    def __init__(self, papers: list[dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self._tokenize(papers)

    def _tokenize(self, papers: list[dict]) -> dict:
        texts = [p["abstract"] for p in papers]
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    return math.exp(avg_loss)


def train_one_batch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def build_replay_dataset(current_dataset, previous_datasets, replay_ratio):
    if not previous_datasets:
        return current_dataset

    n_replay = int(len(current_dataset) * replay_ratio)
    all_previous = ConcatDataset(previous_datasets)
    replay_indices = random.sample(
        range(len(all_previous)), min(n_replay, len(all_previous))
    )
    replay_subset = torch.utils.data.Subset(all_previous, replay_indices)

    return ConcatDataset([current_dataset, replay_subset])


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    raw_dir = Path(config["paths"]["data_raw"])
    batch_files = sorted(raw_dir.glob("*.json"))

    all_datasets = []
    results = {}

    for _, batch_file in enumerate(batch_files):
        batch_name = batch_file.stem
        print(f"\n=== Training on {batch_name} ===")

        with open(batch_file, "r", encoding="utf-8") as f:
            papers = json.load(f)

        current_dataset = AbstractDataset(
            papers, tokenizer, config["model"]["max_length"]
        )
        train_dataset = build_replay_dataset(
            current_dataset, all_datasets, config["data"]["replay_ratio"]
        )
        all_datasets.append(current_dataset)

        dataloader = DataLoader(
            train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        for epoch in range(config["training"]["epochs_per_batch"]):
            loss = train_one_batch(model, dataloader, optimizer, device)
            print(f"Epoch {epoch+1} — loss: {loss:.4f}")

        print("Eval on all seen batches...")
        results[batch_name] = {}
        for seen_name, seen_dataset in zip(
            [f.stem for f in batch_files[: len(all_datasets)]], all_datasets
        ):
            seen_loader = DataLoader(
                seen_dataset, batch_size=config["training"]["batch_size"]
            )
            ppl = compute_perplexity(model, seen_loader, device)
            results[batch_name][seen_name] = ppl
            print(f"  Perplexity on {seen_name} : {ppl:.2f}")

    results_path = Path(config["paths"]["results"]) / "lora_replay_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

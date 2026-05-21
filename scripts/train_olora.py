import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from utils import load_config, load_model, AbstractDataset, compute_perplexity


def get_lora_A_matrices(model):
    lora_As = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_As[name] = param.detach().clone()
    return lora_As


def orthogonal_loss(model, previous_lora_As, lambda_orth):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if "lora_A" in name and name in previous_lora_As:
            overlap = torch.mm(param, previous_lora_As[name].T)
            loss += (overlap**2).sum()

    return lambda_orth * loss


def train_one_batch_olora(
    model, dataloader, optimizer, device, previous_lora_As=None, lambda_orth=0.0
):
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
        if previous_lora_As is not None:
            loss = loss + orthogonal_loss(model, previous_lora_As, lambda_orth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model, tokenizer = load_model(config["model"]["name"])

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
    lambda_orth = config["olora"]["lambda"]

    all_datasets = []
    results = {}
    previous_lora_As = None

    for _, batch_file in enumerate(batch_files):
        batch_name = batch_file.stem
        print(f"\n=== Training on {batch_name} ===")

        with open(batch_file, "r", encoding="utf-8") as f:
            papers = json.load(f)

        dataset = AbstractDataset(papers, tokenizer, config["model"]["max_length"])
        dataloader = DataLoader(
            dataset, batch_size=config["training"]["batch_size"], shuffle=True
        )
        all_datasets.append((batch_name, dataset))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        for epoch in range(config["training"]["epochs_per_batch"]):
            loss = train_one_batch_olora(
                model, dataloader, optimizer, device, previous_lora_As, lambda_orth
            )
            print(f"Epoch {epoch+1} — loss: {loss:.4f}")

        previous_lora_As = get_lora_A_matrices(model)

        print("Eval on all seen batches...")
        results[batch_name] = {}
        for seen_name, seen_dataset in all_datasets:
            seen_loader = DataLoader(
                seen_dataset, batch_size=config["training"]["batch_size"]
            )
            ppl = compute_perplexity(model, seen_loader, device)
            results[batch_name][seen_name] = ppl
            print(f"  Perplexity on {seen_name} : {ppl:.2f}")

    results_path = Path(config["paths"]["results"]) / "olora_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

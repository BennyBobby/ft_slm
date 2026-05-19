import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_config, load_model, AbstractDataset, compute_perplexity


def compute_fisher(model, dataloader, device):
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    model.eval()
    for batch in tqdm(dataloader, desc="Computing Fisher"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        outputs.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

    for name in fisher:
        fisher[name] /= len(dataloader)

    return fisher


def ewc_loss(model, fisher, params_before, ewc_lambda):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in fisher:
            diff = param - params_before[name]
            loss += (fisher[name] * diff**2).sum()

    return ewc_lambda * loss


def train_one_batch_ewc(
    model,
    dataloader,
    optimizer,
    device,
    fisher=None,
    params_before=None,
    ewc_lambda=0.0,
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
        if fisher is not None and params_before is not None:
            loss = loss + ewc_loss(model, fisher, params_before, ewc_lambda)

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
    model.to(device)

    raw_dir = Path(config["paths"]["data_raw"])
    batch_files = sorted(raw_dir.glob("*.json"))
    ewc_lambda = config["ewc"]["lambda"]

    all_datasets = []
    results = {}
    fisher = None
    params_before = None

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
            loss = train_one_batch_ewc(
                model, dataloader, optimizer, device, fisher, params_before, ewc_lambda
            )
            print(f"Epoch {epoch+1} — loss: {loss:.4f}")

        print("Computing Fisher...")
        fisher = compute_fisher(model, dataloader, device)
        params_before = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }

        print("Eval on all seen batches...")
        results[batch_name] = {}
        for seen_name, seen_dataset in all_datasets:
            seen_loader = DataLoader(
                seen_dataset, batch_size=config["training"]["batch_size"]
            )
            ppl = compute_perplexity(model, seen_loader, device)
            results[batch_name][seen_name] = ppl
            print(f"  Perplexity on {seen_name} : {ppl:.2f}")

    results_path = Path(config["paths"]["results"]) / "ewc_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

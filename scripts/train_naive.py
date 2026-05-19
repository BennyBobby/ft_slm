import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils import load_config, load_model, AbstractDataset, compute_perplexity, train_one_batch


def main():
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    model, tokenizer = load_model(config["model"]["name"])
    model.to(device)

    raw_dir = Path(config["paths"]["data_raw"])
    batch_files = sorted(raw_dir.glob("*.json"))

    all_datasets = []
    results = {}

    for _, batch_file in enumerate(batch_files):
        batch_name = batch_file.stem
        print(f"\n=== Train on {batch_name} ===")

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
            loss = train_one_batch(model, dataloader, optimizer, device)
            print(f"Epoch {epoch+1} — loss: {loss:.4f}")

        print("Eval on all seen batches...")
        results[batch_name] = {}
        for seen_name, seen_dataset in all_datasets:
            seen_loader = DataLoader(
                seen_dataset, batch_size=config["training"]["batch_size"]
            )
            ppl = compute_perplexity(model, seen_loader, device)
            results[batch_name][seen_name] = ppl
            print(f"  Perplexity on {seen_name} : {ppl:.2f}")

    results_path = Path(config["paths"]["results"]) / "naive_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

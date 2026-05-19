import json
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from peft import get_peft_model, LoraConfig, TaskType
from utils import load_config, load_model, AbstractDataset, compute_perplexity, train_one_batch


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

        merged_model = model.merge_and_unload()
        checkpoint_path = Path(config["paths"]["models"]) / "lora_replay" / batch_name
        merged_model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        model = get_peft_model(merged_model, lora_config)
        model.to(device)

    results_path = Path(config["paths"]["results"]) / "lora_replay_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

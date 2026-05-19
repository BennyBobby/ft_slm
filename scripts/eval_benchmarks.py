import json
from pathlib import Path
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from utils import load_config


def evaluate(pretrained, tasks, batch_size=8):
    lm = HFLM(pretrained=pretrained, dtype="bfloat16")
    eval_results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
    )
    return {task: eval_results["results"][task]["acc,none"] for task in tasks}


def main():
    config = load_config()

    checkpoints_dir = Path(config["paths"]["models"]) / "lora_replay"
    checkpoints = sorted(checkpoints_dir.iterdir())
    tasks = ["arc_easy", "hellaswag", "piqa"]
    results = {}

    print("\n=== Evaluating base model (no fine-tuning) ===")
    results["base"] = evaluate(config["model"]["name"], tasks)
    print(results["base"])

    for checkpoint_path in checkpoints:
        batch_name = checkpoint_path.name
        print(f"\n=== Evaluating {batch_name} ===")

        scores = evaluate(str(checkpoint_path), tasks)
        results[batch_name] = scores
        print(scores)

    results_path = Path(config["paths"]["results"]) / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

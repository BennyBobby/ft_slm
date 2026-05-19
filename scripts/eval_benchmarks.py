import json
import yaml
from pathlib import Path
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    checkpoints_dir = Path(config["paths"]["models"]) / "lora_replay"
    checkpoints = sorted(checkpoints_dir.iterdir())
    tasks = ["arc_easy", "hellaswag", "piqa"]
    results = {}

    for checkpoint_path in checkpoints:
        batch_name = checkpoint_path.name
        print(f"\n=== Evaluating {batch_name} ===")

        lm = HFLM(pretrained=str(checkpoint_path), dtype="bfloat16")
        eval_results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=0,
            batch_size=8,
        )

        scores = {task: eval_results["results"][task]["acc,none"] for task in tasks}
        results[batch_name] = scores
        print(scores)

    results_path = Path(config["paths"]["results"]) / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()

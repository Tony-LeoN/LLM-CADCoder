from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from vlm_cadcoder.models.registry import build_model
from vlm_cadcoder.utils.json_utils import append_jsonl, write_json
from vlm_cadcoder.utils.yaml_utils import read_config

from .dataset import default_page_image, read_split
from .evaluators import count_accuracy, exact_match, parse_json_object, token_f1
from .tasks import build_image_inputs, default_tasks


def run_model_screening(
    model_name: str,
    task_name: str,
    image_paths: list[str | Path],
    model_config_path: str | Path = "configs/models.json",
    output_root: str | Path = "experiments/model_screening",
) -> Path:
    model_config = read_config(model_config_path)
    models = model_config.get("models", {})
    if model_name not in models:
        raise KeyError(f"Model not found in config: {model_name}")

    model = build_model(model_name, models[model_name])
    tasks = default_tasks(Path(__file__).parent / "prompts")
    if task_name not in tasks:
        raise KeyError(f"Unsupported task: {task_name}")

    task = tasks[task_name]
    run_dir = Path(output_root) / f"{datetime.now():%Y%m%d_%H%M%S}_{model_name}_{task_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    record = _run_one_case(model, model_name, task_name, image_paths, ground_truth=None)
    append_jsonl(run_dir / "predictions.jsonl", record)
    write_json(run_dir / "metrics.json", _aggregate_metrics([record]))
    write_json(run_dir / "config.json", {"model": model_name, "task": task_name, "image_paths": [str(p) for p in image_paths]})
    return run_dir


def run_split_screening(
    model_name: str,
    split_path: str | Path,
    model_config_path: str | Path = "configs/models.json",
    output_root: str | Path = "experiments/model_screening",
    dataflow_root: str | Path = "DataFlow",
    dpi: int = 600,
    task_filter: str | None = None,
) -> Path:
    model_config = read_config(model_config_path)
    models = model_config.get("models", {})
    if model_name not in models:
        raise KeyError(f"Model not found in config: {model_name}")

    model = build_model(model_name, models[model_name])
    run_dir = Path(output_root) / f"{datetime.now():%Y%m%d_%H%M%S}_{model_name}_split"
    run_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case in read_split(split_path):
        tasks = [task_filter] if task_filter else case.tasks
        image_paths = case.input_images or [default_page_image(case.sample_id, dpi=dpi, dataflow_root=dataflow_root)]
        for task_name in tasks:
            record = _run_one_case(
                model=model,
                model_name=model_name,
                task_name=task_name,
                image_paths=image_paths,
                ground_truth=case.ground_truth.get(task_name),
                sample_id=case.sample_id,
            )
            records.append(record)
            append_jsonl(run_dir / "predictions.jsonl", record)

    write_json(run_dir / "metrics.json", _aggregate_metrics(records))
    write_json(
        run_dir / "config.json",
        {
            "model": model_name,
            "split_path": str(split_path),
            "model_config_path": str(model_config_path),
            "dataflow_root": str(dataflow_root),
            "dpi": dpi,
            "task_filter": task_filter,
        },
    )
    return run_dir


def _run_one_case(
    model: Any,
    model_name: str,
    task_name: str,
    image_paths: list[str | Path],
    ground_truth: Any | None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    tasks = default_tasks(Path(__file__).parent / "prompts")
    if task_name not in tasks:
        raise KeyError(f"Unsupported task: {task_name}")
    task = tasks[task_name]
    response = model.generate(
        images=build_image_inputs(image_paths),
        prompt=task.load_prompt(),
        generation_config={"task_name": task_name},
    )
    parsed = response.parsed_json if response.parsed_json is not None else parse_json_object(response.text)
    score = _score_task(task_name, parsed, ground_truth)
    return {
        "sample_id": sample_id,
        "task": task_name,
        "model": model_name,
        "input_images": [str(path) for path in image_paths],
        "prediction_text": response.text,
        "prediction": parsed,
        "ground_truth": ground_truth,
        "score": score,
        "is_json_valid": parsed is not None,
        "latency_sec": response.latency_sec,
        "error": response.error,
    }


def _score_task(task_name: str, prediction: Any, ground_truth: Any | None) -> float | None:
    if ground_truth is None or prediction is None:
        return None
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return None
    if task_name == "view_count":
        return exact_match(prediction.get("view_count"), ground_truth.get("view_count"))
    if task_name == "dimension_ocr":
        pred_items = [item.get("normalized") or item.get("text", "") for item in prediction.get("dimensions", [])]
        gt_items = ground_truth.get("dimensions", [])
        return token_f1(pred_items, gt_items)
    if task_name == "feature_count":
        return count_accuracy(prediction.get("feature_counts", {}), ground_truth.get("feature_counts", {}))
    return None


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    json_valid = sum(1 for record in records if record["is_json_valid"])
    scored = [record["score"] for record in records if record.get("score") is not None]
    return {
        "num_records": total,
        "json_parse_rate": json_valid / total if total else 0.0,
        "mean_score": sum(scored) / len(scored) if scored else None,
        "num_scored_records": len(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single model-screening task.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--split")
    parser.add_argument("--dataflow-root", default="DataFlow")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--model-config", default="configs/models.json")
    parser.add_argument("--output-root", default="experiments/model_screening")
    args = parser.parse_args()

    if args.split:
        run_dir = run_split_screening(
            model_name=args.model,
            split_path=args.split,
            model_config_path=args.model_config,
            output_root=args.output_root,
            dataflow_root=args.dataflow_root,
            dpi=args.dpi,
            task_filter=args.task,
        )
    else:
        if not args.task:
            raise SystemExit("Provide --task for single-image mode.")
        if not args.image:
            raise SystemExit("Provide at least one --image path or use --split.")
        run_dir = run_model_screening(
            model_name=args.model,
            task_name=args.task,
            image_paths=args.image,
            model_config_path=args.model_config,
            output_root=args.output_root,
        )
    print(run_dir)


if __name__ == "__main__":
    main()

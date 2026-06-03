from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def parse_json_object(text: str) -> Any | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def exact_match(prediction: Any, ground_truth: Any) -> float:
    return 1.0 if prediction == ground_truth else 0.0


def normalized_text_tokens(items: list[str]) -> list[str]:
    return [normalize_dimension_text(item) for item in items if item]


def normalize_dimension_text(text: str) -> str:
    return (
        text.strip()
        .replace("⌀", "Φ")
        .replace("φ", "Φ")
        .replace("Ø", "Φ")
        .replace(" ", "")
        .upper()
    )


def token_f1(predicted: list[str], ground_truth: list[str]) -> float:
    pred_counter = Counter(normalized_text_tokens(predicted))
    gt_counter = Counter(normalized_text_tokens(ground_truth))
    if not pred_counter and not gt_counter:
        return 1.0
    if not pred_counter or not gt_counter:
        return 0.0

    overlap = sum((pred_counter & gt_counter).values())
    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(gt_counter.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_accuracy(predicted: dict[str, int], ground_truth: dict[str, int]) -> float:
    keys = set(predicted) | set(ground_truth)
    if not keys:
        return 1.0
    correct = sum(1 for key in keys if int(predicted.get(key, 0)) == int(ground_truth.get(key, 0)))
    return correct / len(keys)


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re


@dataclass(frozen=True)
class ImageInput:
    path: Path
    role: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    parsed_json: Any | None = None
    raw: Any | None = None
    latency_sec: float | None = None
    error: str | None = None


class VisionModel:
    name: str

    def generate(
        self,
        images: list[ImageInput],
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError


def parse_json_from_text(text: str) -> Any | None:
    """Parse strict JSON, then fall back to the first JSON object in the text."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def torch_dtype_from_name(dtype_name: str | None) -> Any:
    if dtype_name is None:
        return None
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]

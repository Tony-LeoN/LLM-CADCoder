from __future__ import annotations

import json
from typing import Any

from .base import ImageInput, ModelResponse, VisionModel


class MockVisionModel(VisionModel):
    """Deterministic adapter for pipeline tests without loading a real VLM."""

    def __init__(self, name: str = "mock", response_by_task: dict[str, Any] | None = None):
        self.name = name
        self.response_by_task = response_by_task or {}

    def generate(
        self,
        images: list[ImageInput],
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> ModelResponse:
        task_name = (generation_config or {}).get("task_name", "unknown")
        payload = self.response_by_task.get(task_name, {"task": task_name, "items": []})
        text = json.dumps(payload, ensure_ascii=False)
        return ModelResponse(text=text, parsed_json=payload, latency_sec=0.0)


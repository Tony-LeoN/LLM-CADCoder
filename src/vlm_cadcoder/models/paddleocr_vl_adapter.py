from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from .base import ImageInput, ModelResponse, VisionModel, parse_json_from_text


class PaddleOCRVLAdapter(VisionModel):
    """Adapter for OCR/layout-heavy tasks.

    PaddleOCR-VL is intentionally isolated from the main VLM environment. Use
    `server_url` in model config to call a separately managed OCR service. This
    keeps CUDA/Paddle/PyTorch dependency conflicts out of the benchmark runner.
    """

    def __init__(self, name: str, model_path: str, **kwargs: Any):
        self.name = name
        self.model_path = model_path
        self.kwargs = kwargs
        self.server_url = kwargs.get("server_url")

    def generate(
        self,
        images: list[ImageInput],
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> ModelResponse:
        if not self.server_url:
            return ModelResponse(
                text="",
                error=(
                    "PaddleOCRVLAdapter requires server_url in model config. "
                    "Run PaddleOCR-VL in a separate environment/service and expose a JSON endpoint."
                ),
            )

        start = time.perf_counter()
        payload = {
            "prompt": prompt,
            "images": [str(image.path) for image in images],
            "generation_config": generation_config or {},
            "output_schema": output_schema,
        }
        try:
            request = urllib.request.Request(
                self.server_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=float(self.kwargs.get("timeout", 120))) as response:
                text = response.read().decode("utf-8")
            return ModelResponse(
                text=text,
                parsed_json=parse_json_from_text(text),
                latency_sec=time.perf_counter() - start,
            )
        except Exception as exc:
            return ModelResponse(
                text="",
                parsed_json=None,
                latency_sec=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

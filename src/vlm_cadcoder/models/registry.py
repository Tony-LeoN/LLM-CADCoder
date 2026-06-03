from __future__ import annotations

from typing import Any

from .base import VisionModel
from .internvl_adapter import InternVLAdapter
from .mock_adapter import MockVisionModel
from .paddleocr_vl_adapter import PaddleOCRVLAdapter
from .qwen_vl_adapter import QwenVLAdapter


def build_model(name: str, config: dict[str, Any]) -> VisionModel:
    adapter = config.get("adapter")
    if adapter == "mock":
        return MockVisionModel(name=name)
    if adapter == "internvl":
        return InternVLAdapter(name=name, **_adapter_kwargs(config))
    if adapter == "qwen_vl":
        return QwenVLAdapter(name=name, **_adapter_kwargs(config))
    if adapter == "paddleocr_vl":
        return PaddleOCRVLAdapter(name=name, **_adapter_kwargs(config))
    raise ValueError(f"Unsupported model adapter for {name}: {adapter}")


def _adapter_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "adapter"}


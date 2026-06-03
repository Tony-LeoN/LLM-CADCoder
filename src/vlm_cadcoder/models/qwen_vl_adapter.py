from __future__ import annotations

import time
from typing import Any

from .base import ImageInput, ModelResponse, VisionModel, parse_json_from_text, torch_dtype_from_name


class QwenVLAdapter(VisionModel):
    """Qwen-VL adapter placeholder for Qwen2.5/3-VL model families."""

    def __init__(self, name: str, model_path: str, **kwargs: Any):
        self.name = name
        self.model_path = model_path
        self.kwargs = kwargs
        self.device = kwargs.get("device", "cuda")
        self.dtype_name = kwargs.get("dtype", "bfloat16")
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 1024))
        self.device_map = kwargs.get("device_map", "auto")
        self.model: Any | None = None
        self.processor: Any | None = None

    def generate(
        self,
        images: list[ImageInput],
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self._load()
        assert self.model is not None
        assert self.processor is not None

        start = time.perf_counter()
        try:
            messages = [_build_user_message(images, prompt)]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs, video_kwargs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to(self.model.device)

            max_new_tokens = int((generation_config or {}).get("max_new_tokens", self.max_new_tokens))
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return ModelResponse(
                text=decoded,
                parsed_json=parse_json_from_text(decoded),
                latency_sec=time.perf_counter() - start,
            )
        except Exception as exc:  # Keep benchmark runs alive and record failures.
            return ModelResponse(
                text="",
                parsed_json=None,
                latency_sec=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        from transformers import AutoProcessor

        model_cls = _resolve_qwen_model_class()
        dtype = torch_dtype_from_name(self.dtype_name)
        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if dtype is not None:
            load_kwargs["dtype"] = dtype

        try:
            self.model = model_cls.from_pretrained(self.model_path, **load_kwargs).eval()
        except TypeError:
            if "dtype" in load_kwargs:
                load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self.model = model_cls.from_pretrained(self.model_path, **load_kwargs).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

    def _process_vision_info(self, messages: list[dict[str, Any]]) -> tuple[Any, Any, dict[str, Any]]:
        from qwen_vl_utils import process_vision_info

        qwen3_kwargs: dict[str, Any] = {}
        if "qwen3" in self.model_path.lower():
            qwen3_kwargs["image_patch_size"] = 16

        try:
            result = process_vision_info(messages, return_video_kwargs=True, **qwen3_kwargs)
            if len(result) == 3:
                return result[0], result[1], result[2] or {}
        except TypeError:
            pass

        image_inputs, video_inputs = process_vision_info(messages, **qwen3_kwargs)
        return image_inputs, video_inputs, {}


def _build_user_message(images: list[ImageInput], prompt: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": str(image.path)})
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def _resolve_qwen_model_class() -> Any:
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

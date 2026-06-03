from __future__ import annotations

import time
from typing import Any

from .base import ImageInput, ModelResponse, VisionModel, parse_json_from_text, torch_dtype_from_name


class InternVLAdapter(VisionModel):
    """InternVL adapter using dynamic tiling for high-resolution drawings."""

    def __init__(self, name: str, model_path: str, **kwargs: Any):
        self.name = name
        self.model_path = model_path
        self.kwargs = kwargs
        self.device = kwargs.get("device", "cuda")
        self.dtype_name = kwargs.get("dtype", "bfloat16")
        self.device_map = kwargs.get("device_map", "auto")
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 1024))
        self.image_size = int(kwargs.get("image_size", 448))
        self.max_num_tiles = int(kwargs.get("max_num_tiles", 12))
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def generate(
        self,
        images: list[ImageInput],
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self._load()
        assert self.model is not None
        assert self.tokenizer is not None

        start = time.perf_counter()
        try:
            import torch

            pixel_values_list = [_load_image_tiles(image.path, self.image_size, self.max_num_tiles) for image in images]
            num_patches_list = [tensor.shape[0] for tensor in pixel_values_list]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            dtype = torch_dtype_from_name(self.dtype_name)
            if dtype is not None:
                pixel_values = pixel_values.to(dtype)
            pixel_values = pixel_values.to(self.device)

            question = prompt
            if "<image>" not in question:
                question = "<image>\n" + question

            config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
            config.update(generation_config or {})

            try:
                text = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config=config,
                    num_patches_list=num_patches_list if len(images) > 1 else None,
                )
            except TypeError:
                text = self.model.chat(self.tokenizer, pixel_values, question, generation_config=config)

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

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        dtype = torch_dtype_from_name(self.dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        load_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs).eval()


def _build_transform(image_size: int) -> Any:
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _load_image_tiles(image_path: Any, image_size: int, max_num_tiles: int) -> Any:
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    tiles = _dynamic_preprocess(image, image_size=image_size, max_num=max_num_tiles, use_thumbnail=True)
    transform = _build_transform(image_size)
    return torch.stack([transform(tile) for tile in tiles])


def _dynamic_preprocess(
    image: Any,
    image_size: int = 448,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> list[Any]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

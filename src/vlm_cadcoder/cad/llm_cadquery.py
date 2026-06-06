from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.models.base import ImageInput
from vlm_cadcoder.models.registry import build_model
from vlm_cadcoder.utils.yaml_utils import read_config


@dataclass(frozen=True)
class CadQueryLLMGenerationResult:
    sample_id: str
    raw_response_path: Path
    script_path: Path
    error: str | None


def sanitize_cadquery_code(text: str) -> str:
    code = _strip_markdown_fence(text).strip()
    code = _normalize_imports(code)
    code = _normalize_export_calls(code)
    return code.rstrip() + "\n"


def generate_cadquery_with_llm(
    sample_id: str,
    model_name: str,
    dataflow_root: str | Path = "DataFlow",
    model_config_path: str | Path = "configs/models.json",
    input_set: str = "testView2CAD",
    output_set: str = "testView2CAD",
    max_new_tokens: int = 4096,
) -> CadQueryLLMGenerationResult:
    root = Path(dataflow_root)
    out_dir = root / "11.CADProgram" / output_set / sample_id
    prompt_path = out_dir / "cadquery_generation_prompt.md"
    raw_response_path = out_dir / "cadquery_llm_generated.raw.md"
    script_path = out_dir / "cadquery_llm_generated.py"

    prompt = _build_strict_generation_prompt(prompt_path.read_text(encoding="utf-8"))
    image_paths = _collect_generation_images(root, input_set, sample_id)

    config = read_config(model_config_path)
    model = build_model(model_name, config["models"][model_name])
    response = model.generate(
        images=[ImageInput(path=path) for path in image_paths],
        prompt=prompt,
        generation_config={"max_new_tokens": max_new_tokens},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_response_path.write_text(response.text, encoding="utf-8", newline="\n")
    script_path.write_text(sanitize_cadquery_code(response.text), encoding="utf-8", newline="\n")

    return CadQueryLLMGenerationResult(
        sample_id=sample_id,
        raw_response_path=raw_response_path,
        script_path=script_path,
        error=response.error,
    )


def sanitize_cadquery_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(input_path)
    target = Path(output_path) if output_path else source
    target.write_text(sanitize_cadquery_code(source.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    return target


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:python|py)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else stripped


def _normalize_imports(code: str) -> str:
    lines = code.splitlines()
    cleaned: list[str] = []
    has_cq_import = False
    for line in lines:
        stripped = line.strip()
        if stripped in {"from cq import *", "import cq"}:
            continue
        if stripped == "from cadquery import *":
            continue
        if stripped == "import cadquery as cq":
            has_cq_import = True
        cleaned.append(line)

    if not has_cq_import:
        insert_at = 0
        if cleaned and cleaned[0].startswith("from __future__ import"):
            insert_at = 1
            while insert_at < len(cleaned) and cleaned[insert_at].strip() == "":
                insert_at += 1
        cleaned.insert(insert_at, "import cadquery as cq")
        cleaned.insert(insert_at + 1, "")
    return "\n".join(cleaned)


def _normalize_export_calls(code: str) -> str:
    return re.sub(r"(?<![\w.])export\(", "cq.exporters.export(", code)


def _build_strict_generation_prompt(base_prompt: str) -> str:
    return (
        base_prompt.rstrip()
        + "\n\n"
        + "## Strict CadQuery API Rules\n\n"
        + "Return ONLY a complete Python source file. Do not use markdown fences.\n"
        + "Use exactly `import cadquery as cq`; never use `from cq import *`, `import cq`, or `from cadquery import *`.\n"
        + "Use `cq.exporters.export(result, str(output_path))` to export STEP.\n"
        + "Use CadQuery selector methods such as `.faces(\">Z\")`, `.faces(\"<Z\")`, `.edges(\"|Z\")`; "
        + "never write `faces[\"top\"]`, `faces[\"bottom\"]`, `edges[0]`, or any subscript on CadQuery methods.\n"
        + "For hole patterns, prefer `.faces(...).workplane().pushPoints(points).hole(diameter)` or "
        + "`.cboreHole(hole_diameter, counterbore_diameter, counterbore_depth)`.\n"
        + "Do not use nonexistent APIs such as `.thread(...)` unless you define metadata only.\n"
        + "If a dimension or feature position is uncertain, keep it as a named parameter with a REVIEW comment.\n"
    )


def _collect_generation_images(root: Path, input_set: str, sample_id: str) -> list[Path]:
    image_paths: list[Path] = []
    clean_image = root / "04.CleanPNG" / input_set / f"{sample_id}.png"
    if clean_image.exists():
        image_paths.append(clean_image)
    crop_dir = root / "06.SingleViews" / input_set / sample_id / "cut-img"
    if crop_dir.exists():
        image_paths.extend(sorted(crop_dir.glob("*.png")))
    return image_paths

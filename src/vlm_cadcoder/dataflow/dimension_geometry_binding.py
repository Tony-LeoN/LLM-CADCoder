from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.models.base import ImageInput, parse_json_from_text
from vlm_cadcoder.models.registry import build_model
from vlm_cadcoder.utils.json_utils import read_json, write_json


@dataclass(frozen=True)
class DimensionGeometryBindingResult:
    sample_id: str
    output_path: Path
    view_count: int
    dimension_count: int
    binding_candidate_count: int


@dataclass(frozen=True)
class DimensionGeometryBindingRecord:
    sample_id: str
    output_path: Path | None
    view_count: int = 0
    dimension_count: int = 0
    binding_candidate_count: int = 0
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DimensionGeometryBindingSummary:
    records: list[DimensionGeometryBindingRecord]
    csv_path: Path | None = None
    json_path: Path | None = None

    @property
    def bound_count(self) -> int:
        return sum(1 for record in self.records if record.output_path is not None and not record.error and not record.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for record in self.records if record.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def bind_dimensions_to_geometry_sample(
    *,
    sample_id: str,
    dataflow_root: str | Path = "DataFlow",
    model_name: str | None = None,
    model_config_path: str | Path = "configs/models.json",
    max_rule_candidates: int = 5,
    max_visual_candidates: int = 20,
    output_path: str | Path | None = None,
) -> DimensionGeometryBindingResult:
    root = Path(dataflow_root)
    drawing_ir_path = root / "10.StructuredCADRepresentation" / sample_id / "drawing_ir.json"
    view_features_path = root / "08.Multi-viewFeatureExtraction" / sample_id / "view_features.json"
    dimensions_path = root / "08.Multi-viewFeatureExtraction" / sample_id / "dimension_candidates.json"

    drawing_ir = _read_required_json(drawing_ir_path, "drawing_ir")
    view_features = _read_required_json(view_features_path, "view_feature_extraction")
    dimensions = _read_required_json(dimensions_path, "dimension_extraction")
    _assert_sample_id(sample_id, drawing_ir, view_features, dimensions)

    views = _views_by_id(drawing_ir)
    feature_candidates, feature_candidate_source = _binding_feature_candidates(view_features, drawing_ir)
    dimension_candidates = [item for item in dimensions.get("dimension_candidates") or [] if isinstance(item, dict)]
    features_by_view = _group_by_view(feature_candidates)
    dimensions_by_view = _group_by_view(dimension_candidates)

    rule_context_by_view: dict[str, dict[str, Any]] = {}
    binding_candidates: list[dict[str, Any]] = []
    unbound_dimensions: list[dict[str, Any]] = []
    ambiguous_bindings: list[dict[str, Any]] = []

    for view_id, view in views.items():
        view_dimensions = dimensions_by_view.get(view_id, [])
        view_features_for_binding = features_by_view.get(view_id, [])
        view_rule_context = _rule_context(
            view_id=view_id,
            dimensions=view_dimensions,
            features=view_features_for_binding,
            max_candidates=max_rule_candidates,
            max_visual_candidates=max_visual_candidates,
        )
        rule_context_by_view[view_id] = view_rule_context
        for dimension_context in view_rule_context["dimensions"]:
            dimension = dimension_context["dimension"]
            candidates = dimension_context["rule_candidates"]
            if not candidates:
                unbound_dimensions.append(
                    {
                        "dimension_id": _id(dimension),
                        "view_id": view_id,
                        "reason": "no_rule_candidate_targets",
                    }
                )
                continue
            best = candidates[0]
            binding = _binding_candidate(
                index=len(binding_candidates) + 1,
                dimension=dimension,
                target=best["feature"],
                view_id=view_id,
                binding_type=_binding_type(dimension, best["feature"]),
                confidence=best["score"],
                rule_support=best,
                source="rule_candidate_scaffold",
            )
            binding_candidates.append(binding)
            if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < 0.08:
                ambiguous_bindings.append(
                    {
                        "dimension_id": _id(dimension),
                        "view_id": view_id,
                        "reason": "top_rule_candidates_close_score",
                        "candidate_feature_ids": [_id(item["feature"]) for item in candidates[:2]],
                    }
                )

    target = (
        Path(output_path)
        if output_path
        else root / "09.Cross-viewGeometricReasoning" / sample_id / "dimension_geometry_bindings.json"
    )
    vlm_requests = _build_vlm_requests(
        root=root,
        views=views,
        rule_context_by_view=rule_context_by_view,
        overlay_dir=target.parent / "overlays",
    )
    vlm_responses: list[dict[str, Any]] = []
    if model_name:
        vlm_responses = _run_vlm_binding(
            model_name=model_name,
            model_config_path=Path(model_config_path),
            requests=vlm_requests,
            root=root,
        )
    vlm_binding_candidates = _vlm_binding_candidates(vlm_responses)

    view_blocks = [
        {
            "view_id": view_id,
            "view_type": str(view.get("type") or "unknown"),
            "image_clean": view.get("image_clean"),
            "dimension_count": len(dimensions_by_view.get(view_id, [])),
            "feature_candidate_count": len(features_by_view.get(view_id, [])),
            "binding_candidate_count": sum(1 for item in binding_candidates if item["view_id"] == view_id),
            "vlm_request_id": request["id"] if (request := _request_for_view(vlm_requests, view_id)) else None,
        }
        for view_id, view in views.items()
    ]

    payload = {
        "schema": "dimension_geometry_binding",
        "version": "0.1.0",
        "sample_id": sample_id,
        "source_drawing_ir": _stage_path(root, drawing_ir_path),
        "inputs": {
            "view_features": _stage_path(root, view_features_path),
            "dimension_candidates": _stage_path(root, dimensions_path),
            "feature_candidate_source": feature_candidate_source,
        },
        "method": {
            "name": "vlm_assisted_dimension_geometry_binding_mvp",
            "version": "0.1.0",
            "role": "candidate_binding_layer_before_constraint_graph",
            "vlm_model": model_name,
            "rule_role": "top_k_target_generation_and_sanity_check",
            "vlm_role": "visual_semantic_selection_from_numbered_candidates",
        },
        "views": view_blocks,
        "binding_candidates": binding_candidates,
        "unbound_dimensions": unbound_dimensions,
        "ambiguous_bindings": ambiguous_bindings,
        "vlm_requests": vlm_requests,
        "vlm_responses": vlm_responses,
        "vlm_binding_candidates": vlm_binding_candidates,
        "quality": _quality_block(
            views=view_blocks,
            dimensions=dimension_candidates,
            binding_candidates=binding_candidates,
            vlm_binding_candidates=vlm_binding_candidates,
            unbound_dimensions=unbound_dimensions,
            ambiguous_bindings=ambiguous_bindings,
            vlm_responses=vlm_responses,
            model_name=model_name,
        ),
    }
    write_json(target, payload)
    return DimensionGeometryBindingResult(
        sample_id=sample_id,
        output_path=target,
        view_count=len(view_blocks),
        dimension_count=len(dimension_candidates),
        binding_candidate_count=len(binding_candidates),
    )


def bind_dimensions_to_geometry_samples(
    *,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    model_name: str | None = None,
    model_config_path: str | Path = "configs/models.json",
    include_copy: bool = False,
    fail_fast: bool = False,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> DimensionGeometryBindingSummary:
    root = Path(dataflow_root)
    sample_ids = [sample_id] if sample_id else _iter_dimension_sample_ids(root / "08.Multi-viewFeatureExtraction")
    records: list[DimensionGeometryBindingRecord] = []
    for current_sample_id in sample_ids:
        if not current_sample_id:
            continue
        if not include_copy and _looks_like_copy_sample(current_sample_id):
            records.append(DimensionGeometryBindingRecord(sample_id=current_sample_id, output_path=None, skipped=True))
            continue
        try:
            result = bind_dimensions_to_geometry_sample(
                sample_id=current_sample_id,
                dataflow_root=root,
                model_name=model_name,
                model_config_path=model_config_path,
            )
            records.append(
                DimensionGeometryBindingRecord(
                    sample_id=current_sample_id,
                    output_path=result.output_path,
                    view_count=result.view_count,
                    dimension_count=result.dimension_count,
                    binding_candidate_count=result.binding_candidate_count,
                )
            )
        except Exception as exc:  # pragma: no cover - sample-specific data failures are reported in summary
            if fail_fast:
                raise
            records.append(DimensionGeometryBindingRecord(sample_id=current_sample_id, output_path=None, error=str(exc)))

    out_root = root / "09.Cross-viewGeometricReasoning"
    csv_path = Path(output_csv) if output_csv else out_root / "dimension_geometry_binding_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "dimension_geometry_binding_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return DimensionGeometryBindingSummary(records=records, csv_path=csv_path, json_path=json_path)


def _rule_context(
    *,
    view_id: str,
    dimensions: list[dict[str, Any]],
    features: list[dict[str, Any]],
    max_candidates: int,
    max_visual_candidates: int,
) -> dict[str, Any]:
    dimension_blocks = []
    for dimension in dimensions:
        scored = [
            candidate
            for feature in features
            if (candidate := _score_rule_candidate(dimension=dimension, feature=feature)) is not None
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        dimension_blocks.append(
            {
                "dimension": _compact_dimension(dimension),
                "rule_candidates": scored[:max_candidates],
            }
        )
    return {
        "view_id": view_id,
        "feature_candidates": [
            _compact_feature(feature) for feature in _visual_feature_candidates(features, max_visual_candidates)
        ],
        "dimensions": dimension_blocks,
    }


def _binding_feature_candidates(view_features: dict[str, Any], drawing_ir: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    semantic = [item for item in view_features.get("feature_candidates") or [] if isinstance(item, dict)]
    if semantic:
        return semantic, "view_features"
    fallback = [
        {
            "id": str(item.get("id") or ""),
            "type": "unknown_geometry_candidate",
            "semantic_status": "candidate",
            "view_id": item.get("view_id"),
            "source_candidate_id": item.get("id"),
            "source_type": "geometry_component",
            "bbox": item.get("bbox"),
            "bbox_on_page": item.get("bbox_on_page"),
            "area_px": item.get("area_px"),
            "confidence": item.get("confidence", 0.2),
            "needs_manual_review": True,
            "review_reasons": ["fallback_from_drawing_ir_geometry_component"],
            "evidence": list(item.get("evidence") or []) + ["feature_candidate_source=drawing_ir"],
        }
        for item in drawing_ir.get("feature_candidates") or []
        if isinstance(item, dict) and item.get("type") == "geometry_component"
    ]
    return fallback, "drawing_ir_geometry_components"


def _visual_feature_candidates(features: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    usable = [feature for feature in features if _id(feature) and _bbox_or_none(feature.get("bbox")) is not None]
    usable.sort(key=_feature_visual_priority, reverse=True)
    return usable[: max(0, limit)]


def _feature_visual_priority(feature: dict[str, Any]) -> tuple[float, float]:
    area = _float_or_default(feature.get("area_px"), 0.0)
    bbox = _bbox_or_none(feature.get("bbox"))
    if area <= 0 and bbox is not None:
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    confidence = _float_or_default(feature.get("confidence"), 0.0)
    return area, confidence


def _score_rule_candidate(*, dimension: dict[str, Any], feature: dict[str, Any]) -> dict[str, Any] | None:
    dim_bbox = _bbox_or_none(dimension.get("bbox_on_page") or dimension.get("bbox"))
    feature_bbox = _bbox_or_none(feature.get("bbox_on_page") or feature.get("bbox"))
    if dim_bbox is None or feature_bbox is None:
        return None
    dim_type = str(dimension.get("dimension_type") or "unknown")
    feature_type = str(feature.get("type") or "unknown")
    compatibility = _compatibility_score(dim_type, feature_type)
    if compatibility <= 0:
        return None

    distance = _bbox_distance(dim_bbox, feature_bbox)
    page_scale = max(1.0, _bbox_diag(_union_bbox(dim_bbox, feature_bbox)))
    distance_score = max(0.0, 1.0 - distance / page_scale)
    score = round(min(0.95, 0.25 + 0.45 * compatibility + 0.30 * distance_score), 4)
    return {
        "feature": _compact_feature(feature),
        "score": score,
        "rank": 0,
        "compatibility_score": compatibility,
        "bbox_distance_px": round(distance, 3),
        "evidence": [
            f"dimension_type={dim_type}",
            f"feature_type={feature_type}",
            f"bbox_distance_px={round(distance, 1)}",
        ],
    }


def _binding_candidate(
    *,
    index: int,
    dimension: dict[str, Any],
    target: dict[str, Any],
    view_id: str,
    binding_type: str,
    confidence: float,
    rule_support: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    support = {key: value for key, value in rule_support.items() if key != "feature"}
    support["rank"] = 1
    return {
        "id": f"{view_id}_binding_{index:03d}",
        "type": "dimension_geometry_binding_candidate",
        "semantic_status": "candidate",
        "source": source,
        "view_id": view_id,
        "dimension_id": _id(dimension),
        "dimension_text": str(dimension.get("text") or ""),
        "dimension_type": str(dimension.get("dimension_type") or "unknown"),
        "dimension_value": dimension.get("value"),
        "target_feature_id": _id(target),
        "target_feature_type": str(target.get("type") or "unknown"),
        "binding_type": binding_type,
        "confidence": confidence,
        "needs_manual_review": True,
        "needs_vlm_review": source == "rule_candidate_scaffold",
        "review_reasons": [
            "dimension_geometry_binding_candidate_needs_validation",
            "constraint_graph_not_built",
        ],
        "rule_support": support,
        "vlm_support": None,
        "evidence": list(support.get("evidence") or []),
    }


def _build_vlm_requests(
    *,
    root: Path,
    views: dict[str, dict[str, Any]],
    rule_context_by_view: dict[str, dict[str, Any]],
    overlay_dir: Path,
) -> list[dict[str, Any]]:
    requests = []
    for view_id, context in rule_context_by_view.items():
        view = views[view_id]
        labeled_context, visual_labels = _labeled_rule_context(context)
        dimensions = labeled_context["dimensions"]
        if not dimensions:
            continue
        overlay = _write_binding_overlay(
            root=root,
            view=view,
            context=labeled_context,
            visual_labels=visual_labels,
            overlay_dir=overlay_dir,
        )
        prompt = _binding_prompt(view_id=view_id, context=labeled_context, visual_labels=visual_labels)
        requests.append(
            {
                "id": f"{view_id}_vlm_binding_request",
                "view_id": view_id,
                "image_clean": view.get("image_clean"),
                "resolved_image_clean": _resolve_stage_path(root, view.get("image_clean")).as_posix()
                if isinstance(view.get("image_clean"), str)
                else None,
                "overlay_image": _stage_path(root, overlay["path"]) if overlay.get("path") else None,
                "resolved_overlay_image": overlay["path"].as_posix() if overlay.get("path") else None,
                "overlay_error": overlay.get("error"),
                "visual_labels": visual_labels,
                "prompt": prompt,
                "dimension_count": len(dimensions),
                "rule_candidate_count": sum(len(item["rule_candidates"]) for item in dimensions),
            }
        )
    return requests


def _labeled_rule_context(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_labels: dict[str, str] = {}
    labeled_dimensions = []
    visual_dimensions = []
    visual_features: list[dict[str, Any]] = []

    def add_visual_feature(feature: dict[str, Any]) -> None:
        feature_id = _id(feature)
        if not feature_id or feature_id in feature_labels:
            return
        feature_labels[feature_id] = f"G{len(feature_labels) + 1}"
        visual_features.append(
            {
                "label": feature_labels[feature_id],
                "id": feature_id,
                "type": feature.get("type"),
                "bbox": feature.get("bbox"),
                "bbox_on_page": feature.get("bbox_on_page"),
            }
        )

    for dimension_index, block in enumerate(context.get("dimensions") or [], start=1):
        dimension = dict(block.get("dimension") or {})
        dimension_label = f"D{dimension_index}"
        dimension["label"] = dimension_label
        visual_dimensions.append(
            {
                "label": dimension_label,
                "id": _id(dimension),
                "text": dimension.get("text"),
                "bbox": dimension.get("bbox"),
                "bbox_on_page": dimension.get("bbox_on_page"),
            }
        )

        labeled_candidates = []
        for candidate in block.get("rule_candidates") or []:
            candidate_copy = {key: value for key, value in candidate.items() if key != "feature"}
            feature = dict(candidate.get("feature") or {})
            feature_id = _id(feature)
            add_visual_feature(feature)
            feature["label"] = feature_labels.get(feature_id)
            candidate_copy["feature"] = feature
            labeled_candidates.append(candidate_copy)

        labeled_dimensions.append({"dimension": dimension, "rule_candidates": labeled_candidates})

    for feature in context.get("feature_candidates") or []:
        if isinstance(feature, dict):
            add_visual_feature(feature)

    return (
        {"view_id": context.get("view_id"), "feature_candidates": visual_features, "dimensions": labeled_dimensions},
        {"dimensions": visual_dimensions, "features": visual_features},
    )


def _write_binding_overlay(
    *,
    root: Path,
    view: dict[str, Any],
    context: dict[str, Any],
    visual_labels: dict[str, Any],
    overlay_dir: Path,
) -> dict[str, Any]:
    image_text = view.get("image_clean")
    if not isinstance(image_text, str):
        return {"path": None, "error": "view_image_clean_missing"}
    image_path = _resolve_stage_path(root, image_text)
    if not image_path.exists():
        return {"path": None, "error": f"view_image_clean_not_found: {image_path}"}

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return {"path": None, "error": "pillow_not_available"}

    try:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        target = overlay_dir / f"{context.get('view_id')}_binding_overlay.png"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        font_size = max(18, min(72, round(max(image.size) / 48)))
        line_width = max(3, round(max(image.size) / 900))
        font = _overlay_font(ImageFont, font_size)
        _draw_labeled_boxes(
            draw=draw,
            items=visual_labels.get("features") or [],
            color=(219, 68, 55),
            font=font,
            line_width=line_width,
            font_size=font_size,
        )
        _draw_labeled_boxes(
            draw=draw,
            items=visual_labels.get("dimensions") or [],
            color=(30, 136, 229),
            font=font,
            line_width=line_width,
            font_size=font_size,
        )
        image.save(target)
        return {"path": target, "error": None}
    except Exception as exc:  # pragma: no cover - image codecs and corrupt user files vary by environment
        return {"path": None, "error": f"overlay_write_failed: {exc}"}


def _overlay_font(image_font: Any, font_size: int) -> Any:
    for font_name in ("arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return image_font.truetype(font_name, font_size)
        except OSError:
            continue
    return image_font.load_default()


def _draw_labeled_boxes(
    *,
    draw: Any,
    items: list[dict[str, Any]],
    color: tuple[int, int, int],
    font: Any,
    line_width: int,
    font_size: int,
) -> None:
    for item in items:
        bbox = _bbox_or_none(item.get("bbox"))
        if bbox is None:
            continue
        label = str(item.get("label") or "")
        draw.rectangle(bbox, outline=color, width=line_width)
        text_x = bbox[0]
        text_y = max(0, bbox[1] - font_size - 8)
        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        padding = max(3, round(font_size / 8))
        background = [
            text_bbox[0] - padding,
            text_bbox[1] - padding,
            text_bbox[2] + padding,
            text_bbox[3] + padding,
        ]
        draw.rectangle(background, fill=color)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)


def _run_vlm_binding(
    *,
    model_name: str,
    model_config_path: Path,
    requests: list[dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    config = read_json(model_config_path)
    models = config.get("models", {}) if isinstance(config, dict) else {}
    if model_name not in models:
        raise KeyError(f"Model not found in config: {model_name}")
    model = build_model(model_name, models[model_name])
    responses: list[dict[str, Any]] = []
    for request in requests:
        image_text = request.get("resolved_overlay_image") or request.get("resolved_image_clean")
        image_path = Path(image_text) if image_text else None
        image_role = "binding_overlay" if request.get("resolved_overlay_image") else "annotated_view"
        images = [ImageInput(path=image_path, role=image_role)] if image_path and image_path.exists() else []
        response = model.generate(
            images=images,
            prompt=request["prompt"],
            generation_config={"task_name": "dimension_geometry_binding"},
        )
        parsed = response.parsed_json if response.parsed_json is not None else parse_json_from_text(response.text)
        responses.append(
            {
                "request_id": request["id"],
                "view_id": request["view_id"],
                "model": model_name,
                "visual_label_map": _visual_label_map(request.get("visual_labels")),
                "prediction_text": response.text,
                "prediction": parsed,
                "is_json_valid": parsed is not None,
                "latency_sec": response.latency_sec,
                "error": response.error,
            }
        )
    return responses


def _vlm_binding_candidates(vlm_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for response in vlm_responses:
        prediction = response.get("prediction")
        if not isinstance(prediction, dict):
            continue
        visual_label_map = {
            str(label): str(feature_id) for label, feature_id in (response.get("visual_label_map") or {}).items()
        }
        bindings = prediction.get("bindings")
        if not isinstance(bindings, list):
            continue
        for binding in bindings:
            if not isinstance(binding, dict):
                continue
            target_labels, target_ids, invalid_labels = _resolve_vlm_target_features(
                binding=binding,
                visual_label_map=visual_label_map,
            )
            review_reasons = [
                "vlm_dimension_geometry_binding_candidate_needs_validation",
                "constraint_graph_not_built",
            ]
            semantic_status = "candidate"
            if invalid_labels:
                review_reasons.append("vlm_target_label_not_in_request")
                if not target_ids:
                    semantic_status = "invalid_candidate"
            candidates.append(
                {
                    "id": f"{response.get('view_id')}_vlm_binding_{len(candidates) + 1:03d}",
                    "type": "dimension_geometry_binding_candidate",
                    "semantic_status": semantic_status,
                    "source": "vlm_binding_suggestion",
                    "view_id": response.get("view_id"),
                    "dimension_id": str(binding.get("dimension_id") or ""),
                    "target_feature_labels": target_labels,
                    "target_feature_ids": target_ids,
                    "target_feature_id": target_ids[0] if target_ids else None,
                    "invalid_target_feature_labels": invalid_labels,
                    "binding_type": str(binding.get("binding_type") or "unknown"),
                    "confidence": _float_or_default(binding.get("confidence"), 0.35),
                    "needs_manual_review": True,
                    "review_reasons": review_reasons,
                    "vlm_support": {
                        "model": response.get("model"),
                        "request_id": response.get("request_id"),
                        "raw_target_feature_ids": _raw_vlm_target_features(binding),
                        "evidence": list(binding.get("evidence") or []),
                    },
                }
            )
    return candidates


def _visual_label_map(visual_labels: Any) -> dict[str, str]:
    if not isinstance(visual_labels, dict):
        return {}
    mapping: dict[str, str] = {}
    for feature in visual_labels.get("features") or []:
        if not isinstance(feature, dict):
            continue
        label = str(feature.get("label") or "")
        feature_id = str(feature.get("id") or "")
        if label and feature_id:
            mapping[label] = feature_id
    return mapping


def _raw_vlm_target_features(binding: dict[str, Any]) -> list[str]:
    raw_items: list[str] = []
    for key in ("target_feature_ids", "target_feature_labels"):
        for item in binding.get(key) or []:
            if item:
                text = str(item)
                if text not in raw_items:
                    raw_items.append(text)
    return raw_items


def _resolve_vlm_target_features(
    *,
    binding: dict[str, Any],
    visual_label_map: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    target_labels: list[str] = []
    target_ids: list[str] = []
    invalid_labels: list[str] = []
    valid_feature_ids = set(visual_label_map.values())
    for item in _raw_vlm_target_features(binding):
        if item in visual_label_map:
            target_labels.append(item)
            feature_id = visual_label_map[item]
            if feature_id not in target_ids:
                target_ids.append(feature_id)
            continue
        if item in valid_feature_ids:
            if item not in target_ids:
                target_ids.append(item)
            continue
        if item.upper().startswith("G"):
            invalid_labels.append(item)
            target_labels.append(item)
            continue
        if item not in target_ids:
            target_ids.append(item)
    return target_labels, target_ids, invalid_labels


def _binding_prompt(*, view_id: str, context: dict[str, Any], visual_labels: dict[str, Any]) -> str:
    prompt_context = {
        "view_id": view_id,
        "instructions": [
            "Use the overlay image labels: dimensions are D1, D2, ... and geometry candidates are G1, G2, ...",
            "Select which numbered geometry candidate each dimension candidate refers to.",
            "Only use geometry labels listed in visual_labels.features; do not invent G labels.",
            "Use visible leaders, arrows, extension lines, centerlines, and engineering drawing conventions.",
            "Rules provide top-k candidates only as support; reject them when visual evidence disagrees.",
            "Return JSON object only with bindings, unbound_dimensions, and ambiguous_bindings.",
        ],
        "expected_schema": {
            "bindings": [
                {
                    "dimension_id": "dimension candidate id",
                    "target_feature_ids": ["feature candidate id"],
                    "binding_type": "diameter_of_hole|thread_of_hole|linear_extent|chamfer_of_edge|angle_between_edges|unknown",
                    "confidence": 0.0,
                    "evidence": ["short visual reason"],
                }
            ],
            "unbound_dimensions": ["dimension candidate id"],
            "ambiguous_bindings": [{"dimension_id": "id", "target_feature_ids": ["id1", "id2"], "reason": "why ambiguous"}],
        },
        "visual_labels": visual_labels,
        "candidates": context,
    }
    return json.dumps(prompt_context, ensure_ascii=False, indent=2)


def _quality_block(
    *,
    views: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    binding_candidates: list[dict[str, Any]],
    vlm_binding_candidates: list[dict[str, Any]],
    unbound_dimensions: list[dict[str, Any]],
    ambiguous_bindings: list[dict[str, Any]],
    vlm_responses: list[dict[str, Any]],
    model_name: str | None,
) -> dict[str, Any]:
    reasons = ["dimension_geometry_bindings_need_validation", "constraint_graph_not_built"]
    if model_name is None:
        reasons.append("vlm_binding_not_run")
    if unbound_dimensions:
        reasons.append("unbound_dimensions")
    if ambiguous_bindings:
        reasons.append("ambiguous_rule_bindings")
    if any(not response.get("is_json_valid") for response in vlm_responses):
        reasons.append("invalid_vlm_binding_response")
    if any(candidate.get("invalid_target_feature_labels") for candidate in vlm_binding_candidates):
        reasons.append("invalid_vlm_target_labels")
    return {
        "view_count": len(views),
        "dimension_candidate_count": len(dimensions),
        "binding_candidate_count": len(binding_candidates),
        "vlm_binding_candidate_count": len(vlm_binding_candidates),
        "unbound_dimension_count": len(unbound_dimensions),
        "ambiguous_binding_count": len(ambiguous_bindings),
        "vlm_response_count": len(vlm_responses),
        "needs_manual_review": True,
        "review_reasons": reasons,
        "ready_for_constraint_graph": False,
        "blocking_items": [
            "dimension_geometry_bindings_not_validated",
            "cross_view_relations_not_built",
            "constraint_graph_not_built",
        ],
    }


def _read_required_json(path: Path, schema: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {schema} file: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{schema} JSON must be an object: {path}")
    if data.get("schema") != schema:
        raise ValueError(f"Expected {schema} schema in {path}")
    return data


def _assert_sample_id(sample_id: str, *payloads: dict[str, Any]) -> None:
    for payload in payloads:
        if payload.get("sample_id") != sample_id:
            raise ValueError(f"sample_id mismatch: expected {sample_id}, got {payload.get('sample_id')}")


def _views_by_id(drawing_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    for view in drawing_ir.get("views") or []:
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("id") or view.get("view_id") or "")
        if view_id:
            views[view_id] = view
    return views


def _group_by_view(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        view_id = str(item.get("view_id") or "")
        if view_id:
            grouped.setdefault(view_id, []).append(item)
    return grouped


def _compact_dimension(dimension: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _id(dimension),
        "text": dimension.get("text"),
        "normalized": dimension.get("normalized"),
        "dimension_type": dimension.get("dimension_type"),
        "value": dimension.get("value"),
        "quantity": dimension.get("quantity"),
        "bbox": dimension.get("bbox"),
        "bbox_on_page": dimension.get("bbox_on_page"),
    }


def _compact_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _id(feature),
        "type": feature.get("type"),
        "view_id": feature.get("view_id"),
        "bbox": feature.get("bbox"),
        "bbox_on_page": feature.get("bbox_on_page"),
        "confidence": feature.get("confidence"),
    }


def _compatibility_score(dimension_type: str, feature_type: str) -> float:
    if dimension_type in {"diameter", "radius", "thread"}:
        return 1.0 if feature_type == "hole_candidate" else 0.25 if feature_type == "unknown_geometry_candidate" else 0.0
    if dimension_type == "chamfer":
        if feature_type in {"outer_profile_candidate", "hole_candidate"}:
            return 0.75
        return 0.25 if feature_type == "unknown_geometry_candidate" else 0.0
    if dimension_type == "linear":
        if feature_type in {"outer_profile_candidate", "slot_candidate", "hole_candidate"}:
            return 0.65
        return 0.25 if feature_type == "unknown_geometry_candidate" else 0.0
    if dimension_type == "angle":
        return 0.45 if feature_type in {"outer_profile_candidate", "slot_candidate"} else 0.0
    return 0.0


def _binding_type(dimension: dict[str, Any], feature: dict[str, Any]) -> str:
    dim_type = str(dimension.get("dimension_type") or "unknown")
    feature_type = str(feature.get("type") or "unknown")
    if dim_type == "diameter" and feature_type == "hole_candidate":
        return "diameter_of_hole"
    if dim_type == "thread" and feature_type == "hole_candidate":
        return "thread_of_hole"
    if dim_type == "chamfer":
        return "chamfer_of_edge"
    if dim_type == "linear":
        return "linear_extent"
    if dim_type == "angle":
        return "angle_between_edges"
    return "unknown"


def _bbox_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [int(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_distance(a: list[int], b: list[int]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def _bbox_diag(bbox: list[int]) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _union_bbox(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _request_for_view(requests: list[dict[str, Any]], view_id: str) -> dict[str, Any] | None:
    for request in requests:
        if request.get("view_id") == view_id:
            return request
    return None


def _resolve_stage_path(root: Path, path_text: Any) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == root.name:
        return root.parent / path
    return root / path


def _iter_dimension_sample_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "dimension_candidates.json").exists())


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _write_summary_csv(path: Path, records: list[DimensionGeometryBindingRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "output_path",
                "view_count",
                "dimension_count",
                "binding_candidate_count",
                "skipped",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else "",
                    "view_count": record.view_count,
                    "dimension_count": record.dimension_count,
                    "binding_candidate_count": record.binding_candidate_count,
                    "skipped": record.skipped,
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[DimensionGeometryBindingRecord]) -> None:
    write_json(
        path,
        {
            "schema": "dimension_geometry_binding_summary",
            "version": "0.1.0",
            "records": [
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else None,
                    "view_count": record.view_count,
                    "dimension_count": record.dimension_count,
                    "binding_candidate_count": record.binding_candidate_count,
                    "skipped": record.skipped,
                    "error": record.error,
                }
                for record in records
            ],
        },
    )


def _stage_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return (Path(root.name) / relative).as_posix()


def _id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .layout_schema import BBox


@dataclass(frozen=True)
class ViewFilterConfig:
    min_score: float = 0.5
    hard_min_score: float = 0.08
    top_strip_y2_ratio: float = 0.10
    top_strip_height_ratio: float = 0.08
    top_strip_score: float = 0.60
    dense_ink_ratio: float = 0.16
    dense_thick_ink_ratio: float = 0.14
    dense_component_ratio: float = 0.10
    dense_component_count: int = 8
    min_view_ink_ratio: float = 0.003
    max_line_view_ink_ratio: float = 0.12
    max_line_view_component_ratio: float = 0.12
    min_components_for_low_score_rescue: int = 3
    layout_overlap_coverage: float = 0.35
    layout_reject_types: tuple[str, ...] = (
        "revision_table",
        "title_or_tolerance_table",
        "hole_table",
        "tolerance_table",
    )
    feature_max_dim: int = 640


@dataclass(frozen=True)
class ViewFilterResult:
    sample_id: str
    page: int
    filtered_path: Path
    raw_path: Path | None
    rejected_path: Path
    overlay_path: Path | None
    accepted_views: list[dict[str, Any]]
    rejected_views: list[dict[str, Any]]


def filter_view_detections_file(
    *,
    detection_path: str | Path,
    clean_image_path: str | Path | None = None,
    layout_path: str | Path | None = None,
    dataflow_root: str | Path = "DataFlow",
    output_path: str | Path | None = None,
    raw_output_path: str | Path | None = None,
    rejected_output_path: str | Path | None = None,
    overlay_path: str | Path | None = None,
    config: ViewFilterConfig | None = None,
    save_overlay: bool = True,
) -> ViewFilterResult:
    config = config or ViewFilterConfig()
    detection_path = Path(detection_path)
    dataflow_root = Path(dataflow_root)
    output_path = Path(output_path) if output_path else detection_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detection_data = _read_json(detection_path)
    sample_id = str(detection_data.get("sample_id") or detection_path.parent.name)
    page = int(detection_data.get("page") or _parse_page_from_name(detection_path.name) or 1)
    stem = f"page_{page:03d}"

    if raw_output_path is None and output_path == detection_path:
        raw_output_path = detection_path.parent / f"{stem}_views_raw.json"
    raw_path = Path(raw_output_path) if raw_output_path else None
    if raw_path and not raw_path.exists():
        raw_path.write_text(
            json.dumps(detection_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    rejected_path = (
        Path(rejected_output_path)
        if rejected_output_path
        else output_path.parent / f"{stem}_rejected_views.json"
    )
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    clean_image_path = _default_clean_image_path(
        clean_image_path=clean_image_path,
        dataflow_root=dataflow_root,
        sample_id=sample_id,
        page=page,
    )
    layout_path = _default_layout_path(
        layout_path=layout_path,
        dataflow_root=dataflow_root,
        sample_id=sample_id,
        page=page,
    )

    clean_image = _load_image(clean_image_path)
    layout_regions = _load_layout_regions(layout_path)
    accepted, rejected = filter_view_detections(
        detection_data=detection_data,
        clean_image=clean_image,
        layout_regions=layout_regions,
        config=config,
    )

    filtered_data = dict(detection_data)
    filtered_data["views"] = accepted
    filtered_data["filter"] = {
        "name": "view_candidate_filter",
        "version": 1,
        "config": _config_to_json(config),
        "input_path": detection_path.as_posix(),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
    }
    output_path.write_text(
        json.dumps(filtered_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rejected_data = {
        "sample_id": sample_id,
        "page": page,
        "image_size": detection_data.get("image_size"),
        "source_path": detection_path.as_posix(),
        "filter": filtered_data["filter"],
        "rejected_views": rejected,
    }
    rejected_path.write_text(
        json.dumps(rejected_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_overlay_path: Path | None = None
    if save_overlay and clean_image is not None:
        final_overlay_path = (
            Path(overlay_path)
            if overlay_path
            else output_path.parent / f"{stem}_view_filter_overlay.png"
        )
        _save_overlay(clean_image, accepted, rejected, final_overlay_path)

    return ViewFilterResult(
        sample_id=sample_id,
        page=page,
        filtered_path=output_path,
        raw_path=raw_path,
        rejected_path=rejected_path,
        overlay_path=final_overlay_path,
        accepted_views=accepted,
        rejected_views=rejected,
    )


def filter_view_detections(
    *,
    detection_data: dict[str, Any],
    clean_image: Any | None = None,
    layout_regions: list[dict[str, Any]] | None = None,
    config: ViewFilterConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or ViewFilterConfig()
    image_size = _image_size(detection_data, clean_image)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    layout_regions = layout_regions or []

    for view in detection_data.get("views", []):
        decision = _classify_candidate(
            view=view,
            image_size=image_size,
            clean_image=clean_image,
            layout_regions=layout_regions,
            config=config,
        )
        if decision["accepted"]:
            accepted.append(_accepted_view(view, len(accepted) + 1, decision))
        else:
            rejected.append(_rejected_view(view, decision))

    return accepted, rejected


def _classify_candidate(
    *,
    view: dict[str, Any],
    image_size: tuple[int, int],
    clean_image: Any | None,
    layout_regions: list[dict[str, Any]],
    config: ViewFilterConfig,
) -> dict[str, Any]:
    bbox = _view_bbox(view)
    if bbox is None or bbox.area <= 0:
        return _decision(False, ["invalid_bbox"], None)

    layout_reason = _layout_overlap_reason(bbox, layout_regions, config)
    if layout_reason:
        return _decision(False, [layout_reason], None)

    features = _extract_crop_features(clean_image, bbox, config) if clean_image is not None else None
    score = float(view.get("score") or 0.0)

    if features and _is_dense_text_or_stamp(features, config):
        return _decision(False, ["dense_text_or_stamp"], features)

    if _is_top_strip(bbox, image_size, config) and score < config.top_strip_score:
        return _decision(False, ["top_strip_low_score"], features)

    if score < config.hard_min_score:
        return _decision(False, ["below_hard_min_score"], features)

    if score < config.min_score:
        if features and _is_line_view_like(features, config):
            return _decision(True, [], features)
        return _decision(False, ["below_min_score"], features)

    return _decision(True, [], features)


def _accepted_view(
    view: dict[str, Any],
    new_index: int,
    decision: dict[str, Any],
) -> dict[str, Any]:
    item = dict(view)
    source_view_id = str(view.get("view_id") or f"source_{new_index:03d}")
    item["source_view_id"] = source_view_id
    item["view_id"] = f"view_{new_index:03d}"
    item["filter"] = {
        "accepted": True,
        "reject_reasons": [],
        "features": decision.get("features"),
        "source": "view_candidate_filter_v1",
    }
    return item


def _rejected_view(view: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    item = dict(view)
    item["source_view_id"] = str(view.get("view_id") or "")
    item["filter"] = {
        "accepted": False,
        "reject_reasons": decision["reasons"],
        "features": decision.get("features"),
        "source": "view_candidate_filter_v1",
    }
    return item


def _decision(
    accepted: bool,
    reasons: list[str],
    features: dict[str, Any] | None,
) -> dict[str, Any]:
    return {"accepted": accepted, "reasons": reasons, "features": features}


def _view_bbox(view: dict[str, Any]) -> BBox | None:
    values = view.get("bbox")
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    except (TypeError, ValueError):
        return None
    return BBox(x1, y1, x2, y2)


def _image_size(detection_data: dict[str, Any], clean_image: Any | None) -> tuple[int, int]:
    if clean_image is not None:
        return clean_image.size
    image_size = detection_data.get("image_size") or {}
    return int(image_size.get("width") or 0), int(image_size.get("height") or 0)


def _is_top_strip(
    bbox: BBox,
    image_size: tuple[int, int],
    config: ViewFilterConfig,
) -> bool:
    _, image_height = image_size
    if image_height <= 0:
        return False
    return (
        bbox.y2 <= image_height * config.top_strip_y2_ratio
        and bbox.height <= image_height * config.top_strip_height_ratio
    )


def _is_dense_text_or_stamp(features: dict[str, Any], config: ViewFilterConfig) -> bool:
    return (
        features["ink_ratio"] >= config.dense_ink_ratio
        and features["thick_ink_ratio"] >= config.dense_thick_ink_ratio
        and features["max_component_ratio"] >= config.dense_component_ratio
        and features["component_count"] <= config.dense_component_count
    )


def _is_line_view_like(features: dict[str, Any], config: ViewFilterConfig) -> bool:
    return (
        config.min_view_ink_ratio <= features["ink_ratio"] <= config.max_line_view_ink_ratio
        and features["max_component_ratio"] <= config.max_line_view_component_ratio
        and features["component_count"] >= config.min_components_for_low_score_rescue
    )


def _extract_crop_features(
    clean_image: Any,
    bbox: BBox,
    config: ViewFilterConfig,
) -> dict[str, Any]:
    import numpy as np

    image_width, image_height = clean_image.size
    crop = clean_image.crop(tuple(bbox.clamp(image_width, image_height).as_list())).convert("L")
    original_width, original_height = crop.size
    if not original_width or not original_height:
        return {
            "width": original_width,
            "height": original_height,
            "ink_ratio": 0.0,
            "thick_ink_ratio": 0.0,
            "max_component_ratio": 0.0,
            "component_count": 0,
        }

    crop.thumbnail((config.feature_max_dim, config.feature_max_dim))
    arr = np.asarray(crop)
    dark = arr < 180
    total = int(dark.size)
    dark_count = int(dark.sum())
    if dark_count == 0:
        return {
            "width": original_width,
            "height": original_height,
            "ink_ratio": 0.0,
            "thick_ink_ratio": 0.0,
            "max_component_ratio": 0.0,
            "component_count": 0,
        }

    padded = np.pad(dark.astype("uint8"), 1)
    height, width = dark.shape
    neighbors = sum(padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3))
    thick_count = int(((neighbors >= 8) & dark).sum())
    component_count, max_component = _connected_component_stats(dark)

    return {
        "width": original_width,
        "height": original_height,
        "ink_ratio": round(dark_count / total, 6),
        "thick_ink_ratio": round(thick_count / total, 6),
        "max_component_ratio": round(max_component / total, 6),
        "component_count": component_count,
    }


def _connected_component_stats(dark: Any) -> tuple[int, int]:
    import numpy as np

    seen = np.zeros_like(dark, dtype=bool)
    ys, xs = np.nonzero(dark)
    component_count = 0
    max_component = 0
    height, width = dark.shape

    for y_raw, x_raw in zip(ys, xs):
        y = int(y_raw)
        x = int(x_raw)
        if seen[y, x]:
            continue
        component_count += 1
        size = 0
        stack = [(y, x)]
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and dark[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        max_component = max(max_component, size)

    return component_count, max_component


def _layout_overlap_reason(
    bbox: BBox,
    layout_regions: list[dict[str, Any]],
    config: ViewFilterConfig,
) -> str | None:
    for region in layout_regions:
        region_type = str(region.get("type") or region.get("region_type") or "")
        if region_type not in config.layout_reject_types:
            continue
        region_bbox = _bbox_from_values(region.get("bbox"))
        if region_bbox is None:
            continue
        coverage = _intersection_area(bbox, region_bbox) / max(1, bbox.area)
        if coverage >= config.layout_overlap_coverage:
            return f"layout_region_overlap:{region_type}"
    return None


def _bbox_from_values(values: Any) -> BBox | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    except (TypeError, ValueError):
        return None
    return BBox(x1, y1, x2, y2)


def _intersection_area(left: BBox, right: BBox) -> int:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _load_image(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def _load_layout_regions(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = _read_json(path)
    regions = data.get("regions")
    return regions if isinstance(regions, list) else []


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_clean_image_path(
    *,
    clean_image_path: str | Path | None,
    dataflow_root: Path,
    sample_id: str,
    page: int,
) -> Path | None:
    if clean_image_path:
        return Path(clean_image_path)
    candidate = dataflow_root / "04.CleanPNG" / sample_id / f"page_{page:03d}_clean.png"
    return candidate if candidate.exists() else None


def _default_layout_path(
    *,
    layout_path: str | Path | None,
    dataflow_root: Path,
    sample_id: str,
    page: int,
) -> Path | None:
    if layout_path:
        return Path(layout_path)
    candidate = dataflow_root / "03.LayoutAnalysis" / sample_id / f"page_{page:03d}_layout.json"
    return candidate if candidate.exists() else None


def _parse_page_from_name(name: str) -> int | None:
    parts = name.split("_")
    for index, part in enumerate(parts):
        if part == "page" and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def _config_to_json(config: ViewFilterConfig) -> dict[str, Any]:
    data = asdict(config)
    data["layout_reject_types"] = list(config.layout_reject_types)
    return data


def _save_overlay(
    clean_image: Any,
    accepted_views: list[dict[str, Any]],
    rejected_views: list[dict[str, Any]],
    overlay_path: Path,
) -> None:
    from PIL import ImageDraw

    overlay = clean_image.copy()
    draw = ImageDraw.Draw(overlay)
    for view in accepted_views:
        bbox = _view_bbox(view)
        if bbox:
            draw.rectangle(tuple(bbox.as_list()), outline=(0, 160, 0), width=4)
            draw.text((bbox.x1 + 4, bbox.y1 + 4), view["view_id"], fill=(0, 120, 0))
    for view in rejected_views:
        bbox = _view_bbox(view)
        if bbox:
            reasons = ",".join(view["filter"]["reject_reasons"])
            draw.rectangle(tuple(bbox.as_list()), outline=(220, 0, 0), width=4)
            draw.text((bbox.x1 + 4, bbox.y1 + 4), reasons[:48], fill=(180, 0, 0))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path)

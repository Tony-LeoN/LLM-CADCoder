from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import write_json

_VIEW_DIR_RE = re.compile(r"^view_\d+$")


@dataclass(frozen=True)
class GeometryCoreAuditConfig:
    dataflow_root: Path = Path("DataFlow")
    sample_id: str | None = None
    page: int = 1
    include_copy: bool = False
    use_view_classification: bool = True
    include_isometric: bool = False
    overrides_path: Path | None = None
    black_threshold: int = 250
    component_max_side: int = 512
    save_contact_sheet: bool = True
    contact_sheet_limit: int = 120


@dataclass(frozen=True)
class GeometryCoreAuditRecord:
    sample_id: str
    view_id: str
    view_type: str | None
    view_type_confidence: float | None
    is_primary_view: bool | None
    classification_source: str
    has_clean_view: bool
    has_geometry_core: bool
    has_mask: bool
    has_probability: bool
    size_matches: bool | None
    width: int | None
    height: int | None
    geometry_width: int | None
    geometry_height: int | None
    clean_black_ratio: float | None
    geometry_black_ratio: float | None
    retained_ink_ratio: float | None
    missing_ink_ratio: float | None
    excess_ink_ratio: float | None
    geometry_component_count: int | None
    quality_tier: str
    needs_manual_review: bool
    review_reasons: list[str]
    manual_quality_label: str | None
    manual_notes: str | None
    clean_image_path: str | None
    geometry_core_path: str | None
    mask_path: str | None
    probability_path: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["paths"] = {
            "clean_view_with_annotations": self.clean_image_path,
            "geometry_core": self.geometry_core_path,
            "geometry_core_mask": self.mask_path,
            "geometry_core_prob": self.probability_path,
        }
        return data


@dataclass(frozen=True)
class GeometryCoreAuditSummary:
    records: list[GeometryCoreAuditRecord]
    csv_path: Path | None = None
    json_path: Path | None = None
    contact_sheet_path: Path | None = None

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def tier_counts(self) -> dict[str, int]:
        return {tier: sum(1 for record in self.records if record.quality_tier == tier) for tier in ("A", "B", "C")}

    @property
    def review_count(self) -> int:
        return sum(1 for record in self.records if record.needs_manual_review)


def audit_geometry_core(
    *,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    page: int = 1,
    include_copy: bool = False,
    use_view_classification: bool = True,
    include_isometric: bool = False,
    overrides_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
    contact_sheet: str | Path | None = None,
    save_contact_sheet: bool = True,
    contact_sheet_limit: int = 120,
) -> GeometryCoreAuditSummary:
    config = GeometryCoreAuditConfig(
        dataflow_root=Path(dataflow_root),
        sample_id=sample_id,
        page=page,
        include_copy=include_copy,
        use_view_classification=use_view_classification,
        include_isometric=include_isometric,
        overrides_path=Path(overrides_json) if overrides_json else None,
        save_contact_sheet=save_contact_sheet,
        contact_sheet_limit=contact_sheet_limit,
    )
    out_root = config.dataflow_root / "06.SingleViews"
    overrides_path = config.overrides_path or out_root / "geometry_core_audit_overrides.json"
    overrides = _load_manual_overrides(overrides_path)
    records = [
        _apply_manual_override(_audit_view(target, config), overrides)
        for target in _iter_view_targets(config, overrides)
    ]

    csv_path = Path(output_csv) if output_csv else out_root / "geometry_core_audit.csv"
    json_path = Path(output_json) if output_json else out_root / "geometry_core_audit.json"
    contact_sheet_path = Path(contact_sheet) if contact_sheet else out_root / "geometry_core_audit_contact_sheet.png"
    actual_contact_sheet_path = contact_sheet_path if save_contact_sheet and records else None

    _write_csv(csv_path, records)
    summary = GeometryCoreAuditSummary(
        records=records,
        csv_path=csv_path,
        json_path=json_path,
        contact_sheet_path=actual_contact_sheet_path,
    )
    if actual_contact_sheet_path:
        _write_contact_sheet(actual_contact_sheet_path, records, limit=contact_sheet_limit)
    _write_json(json_path, summary, csv_path, actual_contact_sheet_path)
    return summary


@dataclass(frozen=True)
class _ViewAuditTarget:
    view_dir: Path
    view_type: str | None = None
    view_type_confidence: float | None = None
    is_primary_view: bool | None = None
    classification_source: str = "06.SingleViews"


def _iter_view_targets(
    config: GeometryCoreAuditConfig,
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> list[_ViewAuditTarget]:
    root = config.dataflow_root / "06.SingleViews"
    if config.sample_id:
        sample_dirs = [root / config.sample_id]
    elif not root.exists():
        sample_dirs = []
    else:
        sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    targets: list[_ViewAuditTarget] = []
    for sample_dir in sample_dirs:
        if sample_dir.name == "testView2CAD":
            continue
        if _looks_like_copy_sample(sample_dir.name) and not config.include_copy:
            continue
        if config.use_view_classification:
            classified = _view_targets_from_classification(sample_dir, config)
            if classified is not None:
                targets.extend(_exclude_manual_targets(classified, overrides))
            continue
        raw_targets = [
            _ViewAuditTarget(path)
            for path in sorted(path for path in sample_dir.iterdir() if path.is_dir() and _VIEW_DIR_RE.match(path.name))
        ]
        targets.extend(_exclude_manual_targets(raw_targets, overrides))
    return targets


def _view_targets_from_classification(
    sample_dir: Path,
    config: GeometryCoreAuditConfig,
) -> list[_ViewAuditTarget] | None:
    classification_path = (
        config.dataflow_root
        / "07.ViewClassification"
        / sample_dir.name
        / f"page_{config.page:03d}_view_classification.json"
    )
    if not classification_path.exists():
        return None

    with classification_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    views = data.get("views")
    if not isinstance(views, list):
        return []

    targets: list[_ViewAuditTarget] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        view_id = view.get("view_id")
        if not isinstance(view_id, str) or not _VIEW_DIR_RE.match(view_id):
            continue
        view_type = view.get("type")
        view_type_text = str(view_type) if view_type is not None else None
        if view_type_text == "isometric" and not config.include_isometric:
            continue
        view_dir = sample_dir / view_id
        if not view_dir.exists():
            continue
        confidence = view.get("confidence")
        targets.append(
            _ViewAuditTarget(
                view_dir=view_dir,
                view_type=view_type_text,
                view_type_confidence=float(confidence) if confidence is not None else None,
                is_primary_view=bool(view.get("is_primary")) if view.get("is_primary") is not None else None,
                classification_source=classification_path.as_posix(),
            )
        )
    return targets


def _audit_view(target: _ViewAuditTarget, config: GeometryCoreAuditConfig) -> GeometryCoreAuditRecord:
    view_dir = target.view_dir
    clean_path = view_dir / "clean_view_with_annotations.png"
    geometry_path = view_dir / "geometry_core.png"
    mask_path = view_dir / "geometry_core_mask.png"
    probability_path = view_dir / "geometry_core_prob.png"
    has_clean = clean_path.exists()
    has_geometry = geometry_path.exists()
    has_mask = mask_path.exists()
    has_probability = probability_path.exists()

    sample_id = view_dir.parent.name
    view_id = view_dir.name
    if not has_clean or not has_geometry:
        reasons = []
        if not has_clean:
            reasons.append("missing_clean_view")
        if not has_geometry:
            reasons.append("missing_geometry_core")
        return GeometryCoreAuditRecord(
            sample_id=sample_id,
            view_id=view_id,
            view_type=target.view_type,
            view_type_confidence=target.view_type_confidence,
            is_primary_view=target.is_primary_view,
            classification_source=target.classification_source,
            has_clean_view=has_clean,
            has_geometry_core=has_geometry,
            has_mask=has_mask,
            has_probability=has_probability,
            size_matches=None,
            width=None,
            height=None,
            geometry_width=None,
            geometry_height=None,
            clean_black_ratio=None,
            geometry_black_ratio=None,
            retained_ink_ratio=None,
            missing_ink_ratio=None,
            excess_ink_ratio=None,
            geometry_component_count=None,
            quality_tier="C",
            needs_manual_review=True,
            review_reasons=reasons,
            manual_quality_label=None,
            manual_notes=None,
            clean_image_path=clean_path.as_posix() if has_clean else None,
            geometry_core_path=geometry_path.as_posix() if has_geometry else None,
            mask_path=mask_path.as_posix() if has_mask else None,
            probability_path=probability_path.as_posix() if has_probability else None,
        )

    clean_mask, clean_size = _load_black_mask(clean_path, config.black_threshold)
    geometry_mask, geometry_size = _load_black_mask(geometry_path, config.black_threshold)
    size_matches = clean_size == geometry_size
    clean_black_ratio = _black_ratio(clean_mask)
    geometry_black_ratio = _black_ratio(geometry_mask)

    retained_ink_ratio: float | None = None
    missing_ink_ratio: float | None = None
    excess_ink_ratio: float | None = None
    if size_matches:
        retained_ink_ratio = geometry_black_ratio / clean_black_ratio if clean_black_ratio > 0 else None
        clean_black_count = int(clean_mask.sum())
        if clean_black_count > 0:
            missing_ink_ratio = float((clean_mask & ~geometry_mask).sum() / clean_black_count)
            excess_ink_ratio = float((geometry_mask & ~clean_mask).sum() / clean_black_count)

    component_count = _connected_component_count(geometry_mask, max_side=config.component_max_side)
    quality_tier, reasons = _classify_quality(
        size_matches=size_matches,
        clean_black_ratio=clean_black_ratio,
        geometry_black_ratio=geometry_black_ratio,
        retained_ink_ratio=retained_ink_ratio,
        excess_ink_ratio=excess_ink_ratio,
        component_count=component_count,
        has_mask=has_mask,
        has_probability=has_probability,
    )

    return GeometryCoreAuditRecord(
        sample_id=sample_id,
        view_id=view_id,
        view_type=target.view_type,
        view_type_confidence=target.view_type_confidence,
        is_primary_view=target.is_primary_view,
        classification_source=target.classification_source,
        has_clean_view=has_clean,
        has_geometry_core=has_geometry,
        has_mask=has_mask,
        has_probability=has_probability,
        size_matches=size_matches,
        width=clean_size[0],
        height=clean_size[1],
        geometry_width=geometry_size[0],
        geometry_height=geometry_size[1],
        clean_black_ratio=clean_black_ratio,
        geometry_black_ratio=geometry_black_ratio,
        retained_ink_ratio=retained_ink_ratio,
        missing_ink_ratio=missing_ink_ratio,
        excess_ink_ratio=excess_ink_ratio,
        geometry_component_count=component_count,
        quality_tier=quality_tier,
        needs_manual_review=quality_tier != "A",
        review_reasons=reasons,
        manual_quality_label=None,
        manual_notes=None,
        clean_image_path=clean_path.as_posix(),
        geometry_core_path=geometry_path.as_posix(),
        mask_path=mask_path.as_posix() if has_mask else None,
        probability_path=probability_path.as_posix() if has_probability else None,
    )


def _load_manual_overrides(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_items = data.get("overrides") if isinstance(data, dict) else None
    if not isinstance(raw_items, list):
        return {}

    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id")
        view_id = item.get("view_id")
        if not isinstance(sample_id, str) or not isinstance(view_id, str):
            continue
        overrides[(sample_id, view_id)] = item
    return overrides


def _exclude_manual_targets(
    targets: list[_ViewAuditTarget],
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> list[_ViewAuditTarget]:
    kept: list[_ViewAuditTarget] = []
    for target in targets:
        sample_id = target.view_dir.parent.name
        view_id = target.view_dir.name
        override = overrides.get((sample_id, view_id), {})
        if override.get("exclude") is True:
            continue
        kept.append(target)
    return kept


def _apply_manual_override(
    record: GeometryCoreAuditRecord,
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> GeometryCoreAuditRecord:
    override = overrides.get((record.sample_id, record.view_id))
    if not override:
        return record
    quality_tier = override.get("quality_tier")
    if quality_tier not in {"A", "B", "C"}:
        return record
    reason = override.get("reason")
    notes = override.get("notes")
    review_reasons = [] if quality_tier == "A" else list(record.review_reasons)
    if isinstance(reason, str) and reason:
        if quality_tier != "A":
            review_reasons.append(reason)
    return GeometryCoreAuditRecord(
        **{
            **asdict(record),
            "quality_tier": quality_tier,
            "needs_manual_review": quality_tier != "A",
            "review_reasons": review_reasons,
            "manual_quality_label": quality_tier,
            "manual_notes": notes if isinstance(notes, str) else reason if isinstance(reason, str) else None,
        }
    )


def _load_black_mask(path: Path, threshold: int):
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised on minimal server env only
        raise RuntimeError("geometry-core audit requires Pillow and numpy.") from exc

    image = Image.open(path).convert("L")
    array = np.asarray(image)
    return array < threshold, image.size


def _black_ratio(mask) -> float:
    return float(mask.sum() / max(1, mask.size))


def _connected_component_count(mask, *, max_side: int) -> int:
    import numpy as np
    from PIL import Image

    height, width = mask.shape
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = Image.fromarray(mask.astype("uint8") * 255, mode="L")
        resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.NEAREST)
        mask = np.asarray(resized) > 0

    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        components += 1
        stack = [(y, x)]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return components


def _classify_quality(
    *,
    size_matches: bool,
    clean_black_ratio: float,
    geometry_black_ratio: float,
    retained_ink_ratio: float | None,
    excess_ink_ratio: float | None,
    component_count: int,
    has_mask: bool,
    has_probability: bool,
) -> tuple[str, list[str]]:
    critical: list[str] = []
    warnings: list[str] = []
    if not size_matches:
        critical.append("image_size_mismatch")
    if clean_black_ratio <= 0.0005:
        critical.append("clean_view_has_too_little_ink")
    if geometry_black_ratio <= 0.0005:
        critical.append("geometry_core_too_sparse")
    if geometry_black_ratio >= 0.18:
        critical.append("geometry_core_too_dense")
    if retained_ink_ratio is not None and retained_ink_ratio < 0.12:
        critical.append("ink_retention_too_low")
    if retained_ink_ratio is not None and retained_ink_ratio > 1.40:
        critical.append("ink_retention_too_high")
    if excess_ink_ratio is not None and excess_ink_ratio > 0.35:
        warnings.append("excess_geometry_ink")
    if component_count > 500:
        critical.append("geometry_core_over_fragmented")
    elif component_count > 180:
        warnings.append("geometry_core_fragmented")
    if not has_mask:
        warnings.append("missing_geometry_core_mask")
    if not has_probability:
        warnings.append("missing_geometry_core_prob")

    if critical:
        return "C", critical + warnings
    if warnings:
        return "B", warnings
    return "A", []


def _write_csv(path: Path, records: list[GeometryCoreAuditRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "view_id",
        "view_type",
        "view_type_confidence",
        "is_primary_view",
        "classification_source",
        "quality_tier",
        "needs_manual_review",
        "review_reasons",
        "manual_quality_label",
        "manual_notes",
        "has_clean_view",
        "has_geometry_core",
        "has_mask",
        "has_probability",
        "size_matches",
        "width",
        "height",
        "geometry_width",
        "geometry_height",
        "clean_black_ratio",
        "geometry_black_ratio",
        "retained_ink_ratio",
        "missing_ink_ratio",
        "excess_ink_ratio",
        "geometry_component_count",
        "clean_image_path",
        "geometry_core_path",
        "mask_path",
        "probability_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow(
                {
                    **row,
                    "review_reasons": ";".join(record.review_reasons),
                }
            )


def _write_json(
    path: Path,
    summary: GeometryCoreAuditSummary,
    csv_path: Path,
    contact_sheet_path: Path | None,
) -> None:
    write_json(
        path,
        {
            "schema": "geometry_core_audit",
            "version": "0.1.0",
            "summary": {
                "total_count": summary.total_count,
                "tier_counts": summary.tier_counts,
                "review_count": summary.review_count,
                "csv_path": csv_path.as_posix(),
                "contact_sheet_path": contact_sheet_path.as_posix() if contact_sheet_path else None,
            },
            "records": [record.to_dict() for record in summary.records],
        },
    )


def _write_contact_sheet(path: Path, records: list[GeometryCoreAuditRecord], *, limit: int) -> None:
    from PIL import Image, ImageDraw

    selected = sorted(records, key=lambda record: (record.quality_tier, record.sample_id, record.view_id), reverse=True)[
        :limit
    ]
    thumb_w, thumb_h = 180, 130
    label_h = 46
    cell_w = thumb_w * 2 + 16
    cell_h = thumb_h + label_h + 12
    columns = 2
    rows = max(1, (len(selected) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for index, record in enumerate(selected):
        row, col = divmod(index, columns)
        x = col * cell_w
        y = row * cell_h
        draw.text((x + 6, y + 4), f"{record.sample_id}/{record.view_id} tier={record.quality_tier}", fill=(0, 0, 0))
        if record.review_reasons:
            draw.text((x + 6, y + 20), ",".join(record.review_reasons)[:54], fill=(160, 0, 0))
        _paste_thumb(sheet, record.clean_image_path, (x + 6, y + label_h), (thumb_w, thumb_h))
        _paste_thumb(sheet, record.geometry_core_path, (x + thumb_w + 10, y + label_h), (thumb_w, thumb_h))
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(210, 210, 210))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _paste_thumb(sheet, path_text: str | None, origin: tuple[int, int], size: tuple[int, int]) -> None:
    from PIL import Image, ImageDraw

    if not path_text:
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]), outline=(220, 0, 0))
        draw.text((origin[0] + 8, origin[1] + 8), "missing", fill=(220, 0, 0))
        return
    image = Image.open(path_text).convert("RGB")
    image.thumbnail(size)
    x = origin[0] + (size[0] - image.width) // 2
    y = origin[1] + (size[1] - image.height) // 2
    sheet.paste(image, (x, y))


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered

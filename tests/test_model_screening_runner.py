from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from vlm_cadcoder.benchmarks.model_screening.runner import run_single_view_screening


def test_run_single_view_screening_uses_formal_non_isometric_drawing_ir_views(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-Views")
    model_config = tmp_path / "models.json"
    model_config.write_text(json.dumps({"models": {"mock": {"adapter": "mock"}}}), encoding="utf-8")

    run_dir = run_single_view_screening(
        model_name="mock",
        task_name="dimension_ocr",
        dataflow_root=dataflow,
        sample_id="Part-Views",
        model_config_path=model_config,
        output_root=tmp_path / "experiments",
    )

    records = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["sample_id"] == "Part-Views"
    assert records[0]["view_id"] == "view_001"
    assert records[0]["task"] == "dimension_ocr"
    assert records[0]["input_images"] == [
        (dataflow / "06.SingleViews" / "Part-Views" / "view_001" / "clean_view_with_annotations.png").as_posix()
    ]

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["input_mode"] == "formal_single_views_from_drawing_ir"
    assert config["processed_view_count"] == 1
    assert config["skipped_views"][0]["view_id"] == "view_002"
    assert config["skipped_views"][0]["reason"] == "isometric_view_excluded"


def test_model_screening_cli_runs_single_view_dimension_ocr_batch(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-CLI")
    model_config = tmp_path / "models.json"
    model_config.write_text(json.dumps({"models": {"mock": {"adapter": "mock"}}}), encoding="utf-8")
    output_root = tmp_path / "experiments"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vlm_cadcoder.benchmarks.model_screening.runner",
            "--model",
            "mock",
            "--single-views",
            "--sample-id",
            "Part-CLI",
            "--dataflow-root",
            str(dataflow),
            "--model-config",
            str(model_config),
            "--output-root",
            str(output_root),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = Path(result.stdout.strip())
    records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert [(record["sample_id"], record["view_id"]) for record in records] == [("Part-CLI", "view_001")]


def _write_drawing_ir(dataflow: Path, sample_id: str) -> None:
    drawing_ir_dir = dataflow / "10.StructuredCADRepresentation" / sample_id
    drawing_ir_dir.mkdir(parents=True)
    views = []
    for view_id, view_type in [("view_001", "front"), ("view_002", "isometric")]:
        image_path = dataflow / "06.SingleViews" / sample_id / view_id / "clean_view_with_annotations.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"test-image")
        views.append(
            {
                "id": view_id,
                "type": view_type,
                "image_clean": f"DataFlow/06.SingleViews/{sample_id}/{view_id}/clean_view_with_annotations.png",
            }
        )
    payload = {
        "schema": "drawing_ir",
        "version": "0.1.0",
        "sample_id": sample_id,
        "views": views,
    }
    (drawing_ir_dir / "drawing_ir.json").write_text(json.dumps(payload), encoding="utf-8")

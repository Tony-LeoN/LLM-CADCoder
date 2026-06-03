from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    pdf_path: Path
    step_path: Path | None
    level: str = ""
    part_family: str = ""
    notes: str = ""


def find_pdf_step_pairs(raw_dir: str | Path) -> list[SampleRecord]:
    root = Path(raw_dir)
    pdfs = sorted(root.glob("*.pdf"), key=lambda p: p.name.lower())
    step_by_stem = {
        path.stem.lower(): path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    }

    records: list[SampleRecord] = []
    for pdf in pdfs:
        step = step_by_stem.get(pdf.stem.lower())
        records.append(
            SampleRecord(
                sample_id=pdf.stem,
                pdf_path=pdf,
                step_path=step,
            )
        )
    return records


def write_samples_csv(records: list[SampleRecord], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "pdf_path", "step_path", "level", "part_family", "notes"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "pdf_path": record.pdf_path.as_posix(),
                    "step_path": record.step_path.as_posix() if record.step_path else "",
                    "level": record.level,
                    "part_family": record.part_family,
                    "notes": record.notes,
                }
            )


def read_samples_csv(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_value = (row.get("step_path") or "").strip()
            records.append(
                SampleRecord(
                    sample_id=row["sample_id"],
                    pdf_path=Path(row["pdf_path"]),
                    step_path=Path(step_value) if step_value else None,
                    level=row.get("level", ""),
                    part_family=row.get("part_family", ""),
                    notes=row.get("notes", ""),
                )
            )
    return records


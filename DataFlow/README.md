# DataFlow Stage Contract

`DataFlow` stores generated artifacts, not source code. Keep source code under `src/vlm_cadcoder`.

For stage-by-stage processing commands, see [`COMMANDS.md`](COMMANDS.md).

## Stages

1. `01.RawPDFWithSTEP`
   - Input PDF drawings and paired STEP/STP ground truth.
   - File stems should match when possible.

2. `02.RawPNG`
   - Rendered PDF pages.
   - Each sample should contain `page_001_600dpi.png` and `page_001_600dpi.meta.json`.

3. `03.LayoutAnalysis`
   - Page-level layout results: title block, notes, drawing views, border frames.

4. `04.CleanPNG`
   - Cleaned page or view images after denoising, binarization, or annotation removal.
   - The first implementation keeps the original page size and whites out removable layout regions.
   - Removed regions should still be preserved as crops when they contain useful manufacturing knowledge, such as hole tables or title blocks.

5. `05.ViewDetection`
   - Detected view bounding boxes and view-region metadata.

6. `06.SingleViews`
   - Cropped views.
   - Prefer three layers per view: raw view with annotations, clean geometry view, annotation layer.
   - `testView2CAD/` may store externally cropped view samples for downstream prototype tests. These samples can be used to bypass `05.ViewDetection` temporarily and test DrawingIR/CadQuery generation, but they must be marked as external crops and should not be used as evidence for automatic view detection performance.

7. `07.ViewClassification`
   - View labels: front, top, left, section, detail, unknown.

8. `08.Multi-viewFeatureExtraction`
   - Feature candidates from each view: holes, slots, counterbores, chamfers, fillets.

9. `09.Cross-viewGeometricReasoning`
   - Cross-view correspondences and candidate modeling base plane.

10. `10.StructuredCADRepresentation`
   - DrawingIR, ConstraintGraph, and ModelingIR JSON artifacts.
   - For the external-crop prototype, `testView2CAD/<sample>/external_crop_manifest.json`, `minimal_drawing_ir.json`, and `modeling_plan.json` are generated from external crops, clean images, VLM benchmark outputs, and paired STEP files.

11. `11.CADProgram`
   - Generated CadQuery/FreeCAD scripts, execution logs, and exported STEP/BREP models.
   - For the external-crop prototype, `testView2CAD/<sample>/cadquery_generation_prompt.md` is generated before executable CadQuery code. This prompt is a controlled bridge from minimal DrawingIR to a reviewed CadQuery script.
   - `cadquery_parameters.json` and `cadquery_draft.py` are prototype artifacts. They may be executable, but parameters marked as `needs_review` must not be treated as final drawing-derived constraints.

## Coordinate Convention

All image-space annotations should use:

```text
image_xy_top_left
x increases rightward
y increases downward
bbox = [x1, y1, x2, y2]
```

Keep the render metadata for coordinate transforms between PDF points and image pixels.

## Layout Cleaning MVP

The first layout module is rule based and targets page-level cleanup before view detection:

```text
RawPNG full page
-> black long-line extraction
-> connected line-component grouping
-> page-frame and table-component classification
-> layout JSON + removed-region crops
-> full-size clean PNG
```

Current removable region types:

- `page_border`: outer frame, coordinate-grid strips, and page-edge lines
- `hole_table`: left-side hole/coordinate table
- `title_or_tolerance_table`: bottom title block and general tolerance tables
- `revision_table`: top-right revision table
- `technical_requirements`: page-level multiline technical requirement or technical condition text block

The clean image should mainly preserve drawing views and view-related annotations: dimensions, leader callouts, surface roughness callouts, geometric tolerances, and local view notes. Detected tables and page-level technical requirement blocks are removed from the clean page but preserved as typed crops in `03.LayoutAnalysis/<sample>/regions`, so later stages can still extract richer drawing metadata from title blocks, revision tables, hole tables, tolerance tables, and technical requirements.

The detector classifies each connected long-line component independently. It should not merge every line inside a fixed left, bottom, or top-right window, because nearby dimensions and view outlines can otherwise be swallowed by an oversized removable region. Bottom tables are classified by bottom-edge location, grid strength, width, and area rather than a fixed x-start threshold, because title blocks may start from the left, center, or right side of the sheet.

Example command:

```bash
python -m vlm_cadcoder.cli clean-layout \
  --image DataFlow/LayoutSamples/raw_img/2023-2024-1-335.png \
  --sample-id 2023-2024-1-335 \
  --dataflow-root DataFlow
```

Expected outputs:

```text
DataFlow/03.LayoutAnalysis/<sample_id>/page_001_layout.json
DataFlow/03.LayoutAnalysis/<sample_id>/page_001_overlay.png
DataFlow/03.LayoutAnalysis/<sample_id>/regions/*.png
DataFlow/04.CleanPNG/<sample_id>/page_001_clean.png
DataFlow/04.CleanPNG/<sample_id>/page_001_remove_mask.png
```

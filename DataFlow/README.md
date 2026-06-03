# DataFlow Stage Contract

`DataFlow` stores generated artifacts, not source code. Keep source code under `src/vlm_cadcoder`.

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

5. `05.ViewDetection`
   - Detected view bounding boxes and view-region metadata.

6. `06.SingleViews`
   - Cropped views.
   - Prefer three layers per view: raw view with annotations, clean geometry view, annotation layer.

7. `07.ViewClassification`
   - View labels: front, top, left, section, detail, unknown.

8. `08.Multi-viewFeatureExtraction`
   - Feature candidates from each view: holes, slots, counterbores, chamfers, fillets.

9. `09.Cross-viewGeometricReasoning`
   - Cross-view correspondences and candidate modeling base plane.

10. `10.StructuredCADRepresentation`
   - DrawingIR, ConstraintGraph, and ModelingIR JSON artifacts.

11. `11.CADProgram`
   - Generated CadQuery/FreeCAD scripts, execution logs, and exported STEP/BREP models.

## Coordinate Convention

All image-space annotations should use:

```text
image_xy_top_left
x increases rightward
y increases downward
bbox = [x1, y1, x2, y2]
```

Keep the render metadata for coordinate transforms between PDF points and image pixels.


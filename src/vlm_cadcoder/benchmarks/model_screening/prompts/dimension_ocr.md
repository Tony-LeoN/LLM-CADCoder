You are reading dimensions from a mechanical engineering drawing crop.

Return JSON only:

{
  "dimensions": [
    {
      "text": "raw dimension text exactly as shown",
      "normalized": "normalized dimension text",
      "type": "linear|diameter|radius|angle|thread|tolerance|geometric_tolerance|surface_roughness|pattern|chamfer|unknown",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

Use crop-local pixel coordinates for `bbox`. If you cannot localize a text item, set `"bbox": null`.

Preserve symbols such as Φ, R, M, C, degree marks, plus-minus tolerances, and pattern counts like 4-Φ8.

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- Only include dimension texts visibly present in the provided drawing image.
- Include every visible dimension, including standalone diameters such as Φ65 and plain linear values such as 80.
- Split compound callouts into separate dimension items when they contain more than one CAD parameter. For example, `φ52完全贯穿孔口倒角C0.5` must produce one diameter item for `φ52...` and one chamfer item for `C0.5`.
- Preserve geometric tolerance frames as `geometric_tolerance` items, including the symbol, numeric tolerance, and datum letters when readable.
- Do not duplicate the same physical callout if it appears once inside a single tolerance frame.
- If no readable dimensions are visible, return `"dimensions": []`.

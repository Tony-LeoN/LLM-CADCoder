You are identifying CAD-relevant features in a clean mechanical drawing view.

Return JSON only:

{
  "feature_counts": {
    "through_hole": 0,
    "blind_hole": 0,
    "counterbore": 0,
    "slot": 0,
    "chamfer": 0,
    "fillet": 0,
    "side_hole": 0
  },
  "evidence": ["short visual evidence"]
}

Count feature candidates conservatively. Do not infer features that are not visible or dimensioned.

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- Evidence must refer only to visible geometric features in the provided drawing image.

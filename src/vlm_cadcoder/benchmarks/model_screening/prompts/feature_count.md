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


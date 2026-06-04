You are identifying CAD-relevant features in a mechanical engineering drawing view or page crop.

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

Count physical part features conservatively from visible geometry and feature callouts. Do not count title blocks, revision tables, tolerance tables, logos, borders, notes, or duplicate depictions of the same physical feature across views.

Feature definitions:

- `through_hole`: holes explicitly shown or called out as through holes, including "through", "THRU", "完全贯穿", "通孔".
- `blind_hole`: holes with a finite bottom depth that do not pass through the part. Do not count a through hole with a counterbore as a blind hole.
- `counterbore`: cylindrical step recesses, counterbores, spotfaces, or "沉孔"; include reverse-side counterbores marked as "反面沉孔".
- `slot`: elongated closed slots or slotted cuts.
- `chamfer`: chamfered edges called out with C notation or visible chamfer geometry.
- `fillet`: rounded edge features called out with R notation or visible fillet geometry.
- `side_hole`: holes located on a side face and shown mainly in a side/profile view.

Multiplier and callout rules:

- Treat `4 x`, `4×`, `4X`, and `4-` as a multiplier. For example, `4 x Φ4.5`, `4×Φ4.5`, and `4-Φ4.5` mean 4 holes.
- Treat `4xC2` or `4 x C2` as 4 chamfers.
- A compound callout such as `4 x Φ4.5 完全贯穿 / Φ8 深 4.6` means 4 through holes and 4 counterbores.
- A compound callout such as `4 x Φ3.4 完全贯穿 / Φ6 深 3±0.1 / 反面沉孔` means 4 through holes and 4 reverse-side counterbores.
- Do not count ordinary size dimensions such as 60, 50, 35, 40, 30, 11, or thickness 8 as features.
- If the drawing contains both geometry and text for the same feature, count it once using the multiplier, not once per visible circle.

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- Evidence must refer only to visible geometric features in the provided drawing image.
- Evidence should cite the callout or visible geometry used for each nonzero count.

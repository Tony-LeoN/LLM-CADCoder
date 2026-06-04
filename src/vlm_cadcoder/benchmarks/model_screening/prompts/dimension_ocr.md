You are reading dimensions from a mechanical engineering drawing crop.

Return JSON only:

{
  "dimensions": [
    {
      "text": "raw dimension text exactly as shown",
      "normalized": "normalized dimension text",
      "type": "linear|diameter|radius|angle|thread|tolerance|surface_roughness|pattern|unknown"
    }
  ]
}

Preserve symbols such as Φ, R, M, degree marks, plus-minus tolerances, and pattern counts like 4-Φ8.

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- Only include dimension texts visibly present in the provided drawing image.
- If no readable dimensions are visible, return `"dimensions": []`.

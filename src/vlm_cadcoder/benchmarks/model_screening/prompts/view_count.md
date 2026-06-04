You are analyzing a mechanical engineering drawing.

Return JSON only:

{
  "view_count": <integer>,
  "evidence": ["short visual evidence"]
}

Count only drawing views of the part. Exclude title blocks, notes, tables, logos, and border frames.

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- Evidence must describe only visible regions in the provided drawing image, such as "one main orthographic view in the center" or "three aligned orthographic views".
- If the image is unreadable or blank, return `"view_count": 0` and explain the visible issue in evidence.

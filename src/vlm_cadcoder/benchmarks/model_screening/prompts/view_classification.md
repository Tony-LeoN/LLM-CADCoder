You are analyzing a mechanical engineering drawing.

Return JSON only:

{
  "views": [
    {
      "id": "view_1",
      "type": "front|top|left|right|section|detail|unknown",
      "confidence": 0.0,
      "evidence": "short reason"
    }
  ],
  "main_view_id": "view_1"
}

Use engineering drawing projection relationships. If uncertain, use "unknown" and explain briefly in evidence.

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- Evidence must describe visible geometry or projection layout in the provided drawing image.

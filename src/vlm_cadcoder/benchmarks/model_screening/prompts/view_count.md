You are analyzing a mechanical engineering drawing.

Return JSON only:

{
  "view_count": <integer>,
  "views": [
    {
      "id": "view_1",
      "description": "short visible description",
      "approx_location": "left|center|right|top|bottom"
    }
  ],
  "evidence": ["short visual evidence"]
}

Count all part drawing views on the whole engineering drawing sheet. Do not count only the main/front view.

Count as drawing views:

- main/front orthographic views
- top, side, left, right, or profile views
- narrow rectangular side/profile views
- section views
- detail or auxiliary views

Exclude:

- title blocks
- notes and technical requirements
- tolerance tables
- logos
- border frames
- dimension text, arrows, and leader annotations by themselves

Rules:

- Do not use markdown code fences.
- Do not output URLs, file names, or external references.
- First enumerate every visible part view in `views`, then set `view_count` equal to the number of enumerated views.
- A narrow standalone side/profile outline on the right side of the sheet counts as one view.
- Evidence must describe only visible regions in the provided drawing image, such as "large main view on the left and narrow side view on the right".
- If the image is unreadable or blank, return `"view_count": 0` and explain the visible issue in evidence.

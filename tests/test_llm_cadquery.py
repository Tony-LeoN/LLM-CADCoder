from __future__ import annotations

from vlm_cadcoder.cad.llm_cadquery import sanitize_cadquery_code


def test_sanitize_cadquery_code_strips_fences_and_normalizes_imports() -> None:
    text = """```python
from cq import *

result = cq.Workplane("XY").box(1, 2, 3)
export(result, "part.step")
```"""

    code = sanitize_cadquery_code(text)

    assert code.startswith("import cadquery as cq")
    assert "```" not in code
    assert "from cq import *" not in code
    assert "cq.exporters.export(result, \"part.step\")" in code


def test_sanitize_cadquery_code_preserves_future_import_order() -> None:
    text = """from __future__ import annotations
from cadquery import *

result = cq.Workplane("XY").box(1, 2, 3)
"""

    code = sanitize_cadquery_code(text)

    assert code.splitlines()[:3] == [
        "from __future__ import annotations",
        "import cadquery as cq",
        "",
    ]

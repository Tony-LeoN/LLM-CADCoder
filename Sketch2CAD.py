import os
import subprocess
import torch

from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

# =========================================================
# PATHS
# =========================================================

IMAGE_PATH = "/home/zxwcax/Dataset/SketchSeg/1/0.images/2023-2024-1-329.png"
MODEL_PATH = "/home/zxwcax/models/Mini-InternVL-2B"

OUTPUT_PY = "generated_freecad.py"
OUTPUT_STEP = "generated_model.step"

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# =========================================================
# LOAD IMAGE
# =========================================================

print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

# InternVL commonly uses 448x448
image = image.resize((448, 448))

# =========================================================
# IMAGE -> TENSOR
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

pixel_values = transform(image).unsqueeze(0)
pixel_values = pixel_values.to(torch.bfloat16).cuda()

# =========================================================
# PROMPT
# =========================================================

question = """<image>

Generate executable FreeCAD Python code.

STRICT RULES:
- Output Python code only
- Use ONLY these APIs:

import FreeCAD
import Part

doc = FreeCAD.newDocument()

Part.makeBox(length,width,height)

Part.makeCylinder(radius,height)

Part.makeSphere(radius)

Part.show(shape)

doc.recompute()

- DO NOT invent functions
- DO NOT use markdown
- DO NOT explain anything

"""

# =========================================================
# INFERENCE
# =========================================================

print("Running InternVL inference...")

with torch.no_grad():

    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config={
            "max_new_tokens": 512,
            "do_sample": False
        }
    )

# =========================================================
# CLEAN RESPONSE
# =========================================================

response = response.strip()

# remove markdown if exists
response = response.replace("```python", "")
response = response.replace("```", "")

# =========================================================
# APPEND SAFE EXPORT CODE
# =========================================================

append_code = """

# =====================================================
# AUTO EXPORT STEP
# =====================================================

import Import

objs = []

for obj in FreeCAD.ActiveDocument.Objects:
    try:
        if hasattr(obj, "Shape"):
            objs.append(obj)
    except:
        pass

Import.export(objs, "generated_model.step")

print("STEP exported:", "generated_model.step")
"""

response += append_code

# =========================================================
# SAVE PYTHON SCRIPT
# =========================================================

with open(OUTPUT_PY, "w", encoding="utf-8") as f:
    f.write(response)

print()
print("======================================")
print("FreeCAD script saved:")
print(OUTPUT_PY)
print("======================================")
print()

# =========================================================
# SHOW GENERATED CODE
# =========================================================

print(response)

# =========================================================
# RUN FREECAD
# =========================================================

print()
print("======================================")
print("Running FreeCAD...")
print("======================================")

cmd = ["freecad", OUTPUT_PY]

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)

# =========================================================
# CHECK STEP
# =========================================================

if os.path.exists(OUTPUT_STEP):
    print()
    print("======================================")
    print("SUCCESS!")
    print("STEP generated:")
    print(OUTPUT_STEP)
    print("======================================")
else:
    print()
    print("======================================")
    print("FAILED: STEP file not found")
    print("======================================")

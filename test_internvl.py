import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

IMAGE_PATH = "/home/zxwcax/Dataset/SketchSeg/1/0.images/2023-2024-1-329.png"
MODEL_PATH = "/home/zxwcax/models/Mini-InternVL-2B"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# model
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# image
image = Image.open(IMAGE_PATH).convert("RGB")

# resize
image = image.resize((448, 448))

# image -> tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

pixel_values = transform(image).unsqueeze(0)
pixel_values = pixel_values.to(torch.bfloat16).cuda()

# prompt
question = """<image>
Reconstruct 3D model from CAD drawing.

Output ONLY executable FreeCAD Python code.

### Reasoning
1. Identify orthographic views
2. Read dimensions
3. Recover 3D geometry
4. Generate FreeCAD code

### FreeCAD Code
"""

# inference
with torch.no_grad():
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config={
            "max_new_tokens": 256,
            "do_sample": False
        }
    )

print("===== OUTPUT =====")
print(response)








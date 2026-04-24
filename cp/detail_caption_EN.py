import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===========================
# Step 0. Environment Setup
# ===========================
os.environ["TRANSFORMERS_ATTENTION_IMPL"] = "eager"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===========================
# Step 1. Input Subject
# ===========================
subject = input("Enter subject name (e.g., physics, chemistry, math): ").strip()
if not subject:
    raise ValueError("❌ Subject name cannot be empty!")
print(f"📘 Current Subject: {subject}")

# ===========================
# Step 2. Path Configuration (Relative to Working Directory)
# ===========================
# Using current working directory to ensure portability
work_dir = os.getcwd() 

# Expected structure: ./data/data/{subject} and ./data/data/annotations/
base_dir = os.path.join(work_dir, "data", "raw")
image_dir = os.path.join(base_dir, subject)
jsonl_path = os.path.join(base_dir, "annotations", f"{subject}.jsonl")

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ Image directory not found: {image_dir}")
if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"❌ Annotation file not found: {jsonl_path}")

print(f"🖼️ Image Dir: {image_dir}")
print(f"📝 Annotations: {jsonl_path}")

# ===========================
# Step 3. Load Annotations
# ===========================
annotations = {}
with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            item = json.loads(line)
            annotations[item["id"]] = item
        except json.JSONDecodeError:
            continue

# ===========================
# Step 4. Model Loading
# ===========================
# You can also change this to a relative path if the model is inside the project
model_path = os.path.join(script_dir, "models/Qwen2.5-VL-32B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading model: {model_path}")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
print("✅ Model loaded successfully.")

# ===========================
# Step 5. Process Images
# ===========================
for img_name in tqdm(sorted(os.listdir(image_dir)), desc=f"Processing {subject}"):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    img_path = os.path.join(image_dir, img_name)
    
    # Matching image to annotation
    found_key = next(
        (k for k, v in annotations.items() if v.get("image_path", "").endswith(img_name)),
        None,
    )
    
    if not found_key:
        print(f"⚠️ {img_name} not found in JSONL, skipping.")
        continue

    ann = annotations[found_key]
    title = ann.get("title", "").strip()
    explanation = ann.get("explanation", "").strip()
    
    if not title or not explanation:
        print(f"⚠️ {img_name} missing title/explanation, skipping.")
        continue

    # ===========================
    # Step 6. Generate Description
    # ===========================
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": (
                f"Please write a fluent and detailed English description of this image. "
                f"Focus on visible elements, layout, and scientific structures. "
                f"Context for reference:\n"
                f"Title: {title}\nExplanation: {explanation}\n"
                f"Output ONLY the natural English description. No prefixes or special formatting."
            )},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=vision_inputs, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512) # Increased for detailed descriptions

    # Trim input from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    gen_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    annotations[found_key]["detail"] = gen_text
    # Print first 100 chars for verification
    print(f"✅ {img_name}: {gen_text[:100]}...")

# ===========================
# Step 7. Atomic Save
# ===========================
temp_path = jsonl_path + ".tmp"
with open(temp_path, "w", encoding="utf-8") as f:
    for item in annotations.values():
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
os.replace(temp_path, jsonl_path)

print(f"🎯 Task complete. Results saved to: {jsonl_path}")
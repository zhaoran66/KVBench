import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re

# =========================== 1. 命令行参数与环境配置 ===========================
parser = argparse.ArgumentParser(description="KVBench: Checklist Generation Pipeline")
parser.add_argument("--subject", type=str, required=True, help="Subject name")
parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-VL weights")
# 默认指向当前目录下的 data 文件夹
parser.add_argument("--base_dir", default=os.path.join(os.getcwd(), "data"), help="Project root data dir")
args = parser.parse_args()

# 强制显存优化配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 路径对齐：data/raw/{subject}/ 和 data/raw/annotations/{subject}.jsonl
image_dir = os.path.join(args.base_dir, "raw", args.subject)
jsonl_path = os.path.join(args.base_dir, "raw", "annotations", f"{args.subject}.jsonl")
output_jsonl_path = jsonl_path + ".tmp"

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ Image directory not found: {image_dir}")
if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"❌ Annotation file not found: {jsonl_path}")

# =========================== 2. 加载模型 ===========================
print(f"🚀 Loading Evaluator Model: {args.model_path}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path,
    attn_implementation="eager",
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(args.model_path)
print("✅ Model loaded successfully")

# =========================== 3. 工具函数 ===========================
checklist_item_pattern = re.compile(r"(?:\d{1,2}|•|-)\s*[:.、-]?\s*(.+)")

def load_and_resize_image(image_path, max_size=1024):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    return img

def generate_checklist(image_path, detail):
    try:
        torch.cuda.empty_cache()
        img = load_and_resize_image(image_path)
        
        # 构造专家级 Prompt
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": (
                    "As a scientific image expert, generate 3-6 concise English checklist points "
                    "to verify if the key elements in this image are accurately represented. "
                    "Focus on scientific facts, structures, or historical details. "
                    "Format each line starting with 01, 02... \n"
                    f"Context: {detail[:500]}"
                )},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.0, do_sample=False)

        output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # 提取并过滤
        items = checklist_item_pattern.findall(output_text)
        exclude_keywords = ["checklist", "please", "generate", "task", "instruction"]
        items = [i.strip(" 。:;-") for i in items if len(i.strip()) > 5]
        items = [i for i in items if not any(k in i.lower() for k in exclude_keywords)]

        if len(items) < 2: return None
        return {f"{i+1:02}": item for i, item in enumerate(items[:6])}

    except Exception as e:
        print(f"⚠️ Error on {os.path.basename(image_path)}: {e}")
        return None

# =========================== 4. 主处理流程 ===========================
with open(jsonl_path, "r", encoding="utf-8") as fin, \
     open(output_jsonl_path, "w", encoding="utf-8") as fout:

    lines = fin.readlines()
    for line in tqdm(lines, desc=f"🧩 Checklist: {args.subject}"):
        try:
            ann = json.loads(line)
        except:
            fout.write(line); continue

        # 路径逻辑：ann["id"] -> 001.png
        img_id = ann.get("id", "")
        id_num = img_id.split("_")[-1] if "_" in img_id else img_id
        img_path = os.path.join(image_dir, f"{id_num}.png")

        # 跳过条件：已存在或无描述
        if ann.get("checklist") or not ann.get("detail") or not os.path.exists(img_path):
            fout.write(json.dumps(ann, ensure_ascii=False) + "\n")
            continue

        checklist = generate_checklist(img_path, ann["detail"])
        if checklist:
            ann["checklist"] = checklist
        
        fout.write(json.dumps(ann, ensure_ascii=False) + "\n")

# 原子化替换文件
os.replace(output_jsonl_path, jsonl_path)
print(f"🎯 Completed! Updated: {jsonl_path}")
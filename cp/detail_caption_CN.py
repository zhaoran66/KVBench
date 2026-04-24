import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===========================
# Step 0. 环境设置
# ===========================
os.environ["FLASH_ATTENTION_2_DISABLED"] = "1"
os.environ["TRANSFORMERS_ATTENTION_IMPL"] = "eager"
os.environ["USE_TRITON_KERNEL"] = "0"
os.environ["DISABLE_AWQ_KERNEL"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===========================
# Step 1. 输入学科名称
# ===========================
subject = input("请输入学科名称（例如 physics / chemistry / math ...）：").strip()
print(f"🔍 当前学科: {subject}")

# ===========================
# Step 2. 路径配置
# ===========================
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "data/raw")
image_dir = os.path.join(base_dir, subject)
jsonl_path = os.path.join(base_dir, "annotations", f"{subject}.jsonl")

print(f"📂 图片目录: {image_dir}")
print(f"📂 标注文件: {jsonl_path}")

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ 找不到图片目录: {image_dir}")
if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"❌ 找不到标注文件: {jsonl_path}")

# ===========================
# Step 3. 读取 JSONL 文件到字典
# ===========================
annotations = {}
with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            item = json.loads(line)
            annotations[item["id"]] = item
        except Exception as e:
            print(f"⚠️ 解析 {jsonl_path} 时出错: {e}")

# ===========================
# Step 4. 加载模型
# ===========================
model_path = os.path.join(script_dir, "models/Qwen2.5-VL-32B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 正在加载模型: {model_path}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    attn_implementation="eager",
    device_map="auto",
    quantization_config=None,
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(model_path)
print("✅ 模型加载成功")

# ===========================
# Step 5. 生成描述并实时保存
# ===========================
output_path = jsonl_path  # 直接覆盖原文件

with open(output_path, "w", encoding="utf-8") as fout:
    for img_name in tqdm(sorted(os.listdir(image_dir)), desc=f"🖼 正在生成 {subject} 的图像描述"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(image_dir, img_name)

        # 查找对应的 JSON 条目
        found_key = None
        for ann_id, ann in annotations.items():
            if ann.get("image_path", "").endswith(img_name):
                found_key = ann_id
                break

        if not found_key:
            print(f"⚠️ {img_name} 未在 JSONL 中找到，跳过。")
            continue

        # 跳过已有 detail 的
        if "detail" in annotations[found_key] and annotations[found_key]["detail"].strip():
            print(f"⏩ {img_name} 已存在描述，跳过。")
            fout.write(json.dumps(annotations[found_key], ensure_ascii=False) + "\n")
            fout.flush()
            continue

        title = annotations[found_key].get("title", "").strip()
        explanation = annotations[found_key].get("explanation", "").strip()
        if not title or not explanation:
            print(f"⚠️ {img_name} 缺少标题或解释，跳过。")
            fout.write(json.dumps(annotations[found_key], ensure_ascii=False) + "\n")
            fout.flush()
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {
                        "type": "text",
                        "text": (
                            f"请根据以下信息生成该图像的详细自然语言描述：\n"
                            f"标题：{title}\n"
                            f"解释：{explanation}\n"
                            f"请输出完整连贯的中文描述，不要加任何前缀、标题以及符号。"
                        ),
                    },
                ],
            }
        ]

        try:
            # 处理输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    inputs[k] = v.half()

            # 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    return_dict_in_generate=False,
                    output_scores=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            output_text = output_texts[0].strip()

            annotations[found_key]["detail"] = output_text
            print(f"✅ {img_name}: {output_text[:80]}...")

        except Exception as e:
            annotations[found_key]["detail"] = f"⚠️ 生成失败: {e}"
            print(f"❌ 生成 {img_name} 出错: {e}")

        # ✅ 实时写入（每次覆盖写入）
        fout.write(json.dumps(annotations[found_key], ensure_ascii=False) + "\n")
        fout.flush()  # 确保立即写入磁盘

print(f"🏁 所有图片已处理完成，结果保存至: {output_path}")

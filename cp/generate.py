import os
import json
import torch
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))
def main():
    parser = argparse.ArgumentParser(description="KVBench Generation ")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--model", type=str, default="seedream", help="seedream (API) or flux (Local)")
    parser.add_argument("--source", type=str, choices=["title", "detail", "explanation"], default="detail")
    parser.add_argument("--local", action="store_true", help="If set, load model from local path instead of API")
    parser.add_argument("--model_path", type=str, default=os.path.join(script_dir, "models/Qwen2.5-VL-32B-Instruct"))
    args = parser.parse_args()

    # --- 路径配置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data", "raw", "annotations", f"{args.subject}.jsonl")
    output_dir = os.path.join(script_dir, "data", args.model, f"{args.model}_{args.source}", args.subject)
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 核心区分：加载模型逻辑
    # ==========================================
    if args.local:
        # --- 本地开源模型分支 (以 Diffusers 为例) ---
        from diffusers import FluxPipeline
        print(f"Loading local model from: {args.model_path}")
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        
        def generate_fn(prompt):
            # 本地模型推理
            image = pipe(prompt, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256).images[0]
            return image # 返回 PIL 对象

    else:
        # --- API 线上模型分支 ---
        from openai import OpenAI
        import requests
        from PIL import Image
        from io import BytesIO
        client = OpenAI(base_url=os.getenv("ARK_BASE_URL"), api_key=os.getenv("ARK_API_KEY"))
        
        def generate_fn(prompt):
            response = client.images.generate(model="doubao-seedream-4-0-250828", prompt=prompt)
            image_url = response.data[0].url
            img_data = requests.get(image_url).content
            return Image.open(BytesIO(img_data))

    # --- 统一遍历逻辑 ---
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)
        img_id = item["id"]
        prompt = item[args.source] + ", scientific style, high quality"
        save_path = os.path.join(output_dir, f"{img_id}.png")

        if os.path.exists(save_path): continue

        try:
            image = generate_fn(prompt)
            image.save(save_path)
        except Exception as e:
            print(f"Error on {img_id}: {e}")

if __name__ == "__main__":
    main()
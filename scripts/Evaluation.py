import os
import json
import re
import gc
import argparse
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ================= 1. Core Utility Functions =================
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def clean_comment(text):
    """Remove newlines, tabs, and special characters for a clean JSON string."""
    text = re.sub(r"[\n\t]+", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s。，、？！：；,.!?-]", "", text)
    return text.strip()

def parse_json_block(resp):
    """Find and parse the JSON block in the model response."""
    matches = re.findall(r"\{.*?\}", resp, re.DOTALL)
    for block in reversed(matches):
        try: return json.loads(block)
        except: continue
    return None

def extract_score_comment(response_text, model_name):
    """Extract score and comment. Threshold logic: score >= 0.5 results in 1.0."""
    parsed = parse_json_block(response_text or "")
    comment, score = "", None
    if parsed:
        dynamic_keys = (f"score_{model_name}", "score", "rating", "value", "得分")
        for k in dynamic_keys:
            if k in parsed:
                try: score = float(parsed[k]); break
                except: pass
        comment = parsed.get("comment", parsed.get("reason", parsed.get("理由", "No reason provided")))
    else:
        m = re.search(r"[Ss]core\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", response_text or "")
        if m: score = float(m.group(1))
        comment = response_text
    
    final_score = 1.0 if (score is not None and score >= 0.5) else 0.0
    return final_score, clean_comment(comment)

def generate_prompt(question_text):
    """Construct the English evaluation prompt."""
    return (
        "You are an expert image auditor. Please compare the following two images:\n"
        "Image 1: Ground Truth (Reference), Image 2: AI-Generated Image.\n"
        "Evaluation Criterion: " + question_text + "\n"
        "Instruction: Does Image 2 satisfy the requirement mentioned above? "
        "Please output the result strictly in the following JSON format:\n"
        "{\"score\": 0/1, \"comment\": \"Your detailed reason here\"}"
    )

# ================= 2. Evaluation Task Execution =================
def run_subject_eval(subject, model, processor, device, base_dir, model_name):
    print(f"\n🔍 Auditing Subject: {subject} | Target Model: {model_name}")
    
    # --- Strict Relative Path Logic ---
    # Input: data/raw/annotations/
    # Reference Images: data/raw/{subject}/
    # Generated Images: data/results/{model_name}/{subject}/
    raw_dir = os.path.join(base_dir, "raw")
    results_dir = os.path.join(base_dir, "results")

    jsonl_path = os.path.join(raw_dir, "annotations", f"{subject}.jsonl")
    gt_dir = os.path.join(raw_dir, subject)
    gen_dir = os.path.join(results_dir, model_name, subject)
    
    output_dir = os.path.join(results_dir, model_name, "annotations")
    output_jsonl = os.path.join(output_dir, f"{subject}_{model_name}_scored.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(jsonl_path):
        print(f"⚠️ Error: Annotation not found at {jsonl_path}")
        return

    # Checkpoint logic
    existing_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try: existing_ids.add(json.loads(line)["id"])
                except: continue
        print(f"⏩ Resuming: {len(existing_ids)} items already scored.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        all_entries = [json.loads(l) for l in f if json.loads(l).get("id") not in existing_ids]

    for entry in tqdm(all_entries, desc=f"Evaluating {subject}"):
        img_id = entry.get("id", "")
        # Extract ID part (e.g., physics_001 -> 001)
        id_num = img_id.split("_")[-1] if "_" in img_id else img_id
        
        gt_img = os.path.join(gt_dir, f"{id_num}.png")
        gen_img = os.path.join(gen_dir, f"{id_num}.png")

        if not (os.path.exists(gt_img) and os.path.exists(gen_img)):
            continue
            
        checklist = entry.get("checklist", {})
        if not checklist: continue

        per_scores, per_comments = [], []
        for q_key, question in checklist.items():
            messages = [{"role": "user", "content": [
                {"type": "image", "image": gt_img}, 
                {"type": "image", "image": gen_img}, 
                {"type": "text", "text": generate_prompt(question)}
            ]}]
            
            try:
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=256)
                
                response = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                score, comment = extract_score_comment(response, model_name)
                per_scores.append(score)
                per_comments.append(f"{q_key}: {comment}")
            except Exception as e:
                print(f"Inference error on {img_id}: {e}")
                torch.cuda.empty_cache()

        if per_scores:
            entry.update({
                f"score_{model_name}_list": per_scores,
                f"score_{model_name}_avg": round(sum(per_scores)/len(per_scores), 4),
                f"comments_{model_name}": "; ".join(per_comments)
            })
            with open(output_jsonl, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ================= 3. Main Entry Point =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVBench: Automated Image Audit Pipeline")
    
    parser.add_argument("--subjects", nargs="+", required=True, help="Subjects to evaluate (e.g. physics chemistry)")
    parser.add_argument("--model_name", type=str, required=True, help="Target generated model folder name")
    
    # Required to pass via command line to avoid hardcoded absolute paths
    parser.add_argument("--evaluator_path", type=str, required=True, help="Path to Qwen2.5-VL weights")
    
    # Default to 'data' folder in the current working directory
    current_dir = os.getcwd()
    parser.add_argument("--base_dir", default=os.path.join(current_dir, "data"), help="Project data root")
    
    args = parser.parse_args()

    # Model Loading
    print(f"Loading Evaluator: {args.evaluator_path}")
    processor = AutoProcessor.from_pretrained(args.evaluator_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.evaluator_path, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    # Execution
    for sub in args.subjects:
        run_subject_eval(sub, model, processor, "cuda", args.base_dir, args.model_name)
        clear_gpu()

    print(f"\n✨ All auditing tasks for [{args.model_name}] are complete.")
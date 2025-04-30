# inference.py
"""
Multi-view chest X-ray inference with chained few-shot LLM prompting and multi-case chunking.

1. From validation set, sample up to `samples_per_class` images per class.
2. Split each class’s sampled images into cases of `views_per_case` images.
3. For each case:
     - Aggregate model logits over the views → top-3 predictions.
     - Stage 1 LLM prompt → “medical_report”.
     - Stage 2 LLM prompt → “summary_and_next_steps”.
     - Stage 3 LLM prompt → “simplified_report”.
4. Save each case into output_dir/results/<case_id>/:
     • images/ (the 3 view files)
     • report.json (with predictions, the three sections, and all prompts)

Usage:
    python inference.py \
      --val_dir /path/to/val \
      --train_data_dir /path/to/full_dataset \
      --model_checkpoint /path/to/outputs/model_epoch_20.pth \
      --output_dir ./inference_out \
      --api_key YOUR_GEMINI_API_KEY \
      --img_size 224 \
      --samples_per_class 9 \
      --views_per_case 3
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai

from model import CombinedModel
from dataloader import get_dataloaders

def create_subset(val_dir: Path, subset_dir: Path, samples_per_class: int, views_per_case: int):
    """
    Sample up to `samples_per_class` images per class from validation set,
    then split into cases of `views_per_case` images each.
    """
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True)
    random.seed(42)

    for cls_dir in val_dir.iterdir():
        if not cls_dir.is_dir():
            continue

        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        if not imgs:
            continue

        chosen = imgs if len(imgs) <= samples_per_class else random.sample(imgs, samples_per_class)
        # split into chunks of views_per_case
        chunks = [chosen[i:i + views_per_case] for i in range(0, len(chosen), views_per_case)]

        for idx, chunk in enumerate(chunks):
            case_name = f"{cls_dir.name}_case{idx}"
            out_case = subset_dir / case_name
            out_case.mkdir(parents=True, exist_ok=True)
            for p in chunk:
                shutil.copy2(p, out_case / p.name)

def load_model(checkpoint_path, device, num_classes):
    """Load CombinedModel from checkpoint."""
    model = CombinedModel(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()

def preprocess_image(p: Path, transform):
    return transform(Image.open(p).convert('RGB'))

# Few-shot examples for Stage 1
FEW_SHOT_EXAMPLES = [
    {
        "case_id": "EX001",
        "views": ["PA_EX001.png", "LAT_EX001.png", "AP_EX001.png"],
        "predictions": [("Cardiomegaly", 0.82), ("Pulmonary Edema", 0.12), ("Normal", 0.06)],
        "medical_report": (
            "Findings:\n"
            "- Enlarged cardiac silhouette with CTR >0.55.\n"
            "- Kerley B lines and interstitial markings consistent with pulmonary edema.\n\n"
            "Impression:\n"
            "Moderate cardiomegaly with interstitial pulmonary edema."
        )
    },
    {
        "case_id": "EX002",
        "views": ["PA_EX002.png", "LAT_EX002.png", "AP_EX002.png"],
        "predictions": [("COVID-19 Pneumonia", 0.90), ("ARDS", 0.05), ("Normal", 0.05)],
        "medical_report": (
            "Findings:\n"
            "- Bilateral peripheral ground-glass opacities.\n"
            "- No pleural effusion or cardiomegaly.\n\n"
            "Impression:\n"
            "Findings consistent with early COVID-19 pneumonia."
        )
    }
]

def build_stage1_prompt(case_id, image_names, labels, probs):
    """Few-shot prompt for detailed medical_report only."""
    prompt = (
        "You are a board-certified radiologist. "
        "For each case, produce ONLY a JSON object with key \"medical_report\":\n"
    )
    for ex in FEW_SHOT_EXAMPLES:
        prompt += f"\nCase ID: {ex['case_id']}\nViews:\n"
        for v in ex["views"]:
            prompt += f"- {v}\n"
        prompt += "Predictions:\n"
        for lbl, pr in ex["predictions"]:
            prompt += f"- {lbl}: {pr*100:.1f}%\n"
        json_rep = json.dumps({"medical_report": ex["medical_report"]})
        prompt += f"\nResponse:\n{json_rep}\n"
    prompt += f"\nCase ID: {case_id}\nViews:\n"
    for v in image_names:
        prompt += f"- {v}\n"
    prompt += "Predictions:\n"
    for lbl, pr in zip(labels, probs):
        prompt += f"- {lbl}: {pr*100:.1f}%\n"
    prompt += "\nResponse (JSON):\n"
    return prompt

def build_stage2_prompt(medical_report: str):
    """Chain prompt: from medical_report to summary_and_next_steps."""
    return (
        "Based on the detailed medical report below, extract a concise JSON with key "
        "\"summary_and_next_steps\" as a list of bullet points:\n\n"
        f"{medical_report}\n\nResponse (JSON):\n"
    )

def build_stage3_prompt(medical_report: str, summary: str):
    """Chain prompt: from report & summary to simplified_report."""
    return (
        "Now, given the detailed findings and summary below, produce a JSON with key "
        "\"simplified_report\": a plain-language explanation for a non-expert:\n\n"
        f"{medical_report}\n\nSummary:\n{summary}\n\nResponse (JSON):\n"
    )

def parse_json(text: str):
    """Extract JSON object from model output."""
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        return json.loads(text[start:end])
    except Exception:
        return {}

def main(args):
    # 1) Sample & chunk validation images into cases
    val_dir = Path(args.val_dir)
    subset_dir = Path(args.output_dir) / "subset_val"
    create_subset(val_dir, subset_dir, args.samples_per_class, args.views_per_case)

    # 2) Load device, model, class names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, classes = get_dataloaders(
        args.train_data_dir, batch_size=1, num_workers=0, img_size=args.img_size
    )
    model = load_model(args.model_checkpoint, device, len(classes))

    # 3) Configure Gemini
    genai.configure(api_key=args.api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash')

    # 4) Prepare transforms & output folders
    transform = transforms.Compose([
        transforms.Resize(int(args.img_size*1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 5) Inference per chunked case
    for case_dir in tqdm(sorted(subset_dir.iterdir()), desc="Cases"):
        if not case_dir.is_dir():
            continue

        imgs = sorted(case_dir.glob("*.[pj][pn]g"))
        if not imgs:
            continue

        # aggregate model predictions
        batch = torch.stack([preprocess_image(p, transform) for p in imgs]).to(device)
        with torch.no_grad():
            logits = model(batch)           # (views_per_case, C)
            probs = F.softmax(logits, dim=1).cpu()
            agg = probs.mean(dim=0)         # (C,)

        topk = torch.topk(agg, k=3)
        labels = [classes[i] for i in topk.indices.tolist()]
        probs3 = topk.values.tolist()

        # setup per-case folder & copy images
        out_case = results_dir / case_dir.name
        out_imgs = out_case / "images"
        out_case.mkdir(exist_ok=True)
        out_imgs.mkdir(exist_ok=True)
        for p in imgs:
            shutil.copy2(p, out_imgs / p.name)

        # Stage 1: medical_report
        p1 = build_stage1_prompt(case_dir.name, [p.name for p in imgs], labels, probs3)
        r1 = llm.generate_content(
            p1,
            generation_config=genai.GenerationConfig(
                max_output_tokens=300, temperature=0.3, top_p=0.9, top_k=30
            )
        )
        j1 = parse_json(r1.text)
        med_report = j1.get("medical_report", "")

        # Stage 2: summary_and_next_steps
        p2 = build_stage2_prompt(med_report)
        r2 = llm.generate_content(
            p2,
            generation_config=genai.GenerationConfig(
                max_output_tokens=100, temperature=0.2, top_p=0.9, top_k=20
            )
        )
        j2 = parse_json(r2.text)
        summary = j2.get("summary_and_next_steps", "")

        # Stage 3: simplified_report
        p3 = build_stage3_prompt(med_report, summary)
        r3 = llm.generate_content(
            p3,
            generation_config=genai.GenerationConfig(
                max_output_tokens=100, temperature=0.2, top_p=0.9, top_k=20
            )
        )
        j3 = parse_json(r3.text)
        simplified = j3.get("simplified_report", "")

        # save report.json
        result = {
            "case_id": case_dir.name,
            "predictions": [{"label": l, "probability": float(p)} for l, p in zip(labels, probs3)],
            "medical_report": med_report,
            "summary_and_next_steps": summary,
            "simplified_report": simplified,
            "prompts": {"stage1": p1, "stage2": p2, "stage3": p3}
        }
        with open(out_case / "report.json", "w") as f:
            json.dump(result, f, indent=2)

    print("✅ Inference & reports saved to", results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir",           required=True,
                        help="Validation data root (class subfolders).")
    parser.add_argument("--train_data_dir",    required=True,
                        help="Full data root (train/ & val/) for class names.")
    parser.add_argument("--model_checkpoint",  required=True,
                        help="Path to CombinedModel checkpoint .pth")
    parser.add_argument("--output_dir",        required=True,
                        help="Where to store subset and results/")
    parser.add_argument("--api_key",           required=True,
                        help="Gemini API key")
    parser.add_argument("--img_size",    type=int, default=224,
                        help="Image size used during training.")
    parser.add_argument("--samples_per_class", type=int, default=9,
                        help="Max images per class to sample for inference.")
    parser.add_argument("--views_per_case",    type=int, default=3,
                        help="Number of images per case (views).")
    args = parser.parse_args()
    main(args)

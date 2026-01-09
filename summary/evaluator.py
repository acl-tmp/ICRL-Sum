import os
import sys
import re
import json
import numpy as np
import torch
import argparse
import logging
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from transformers import AutoTokenizer, AutoModelForSequenceClassification

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

logging.getLogger("transformers").setLevel(logging.ERROR)


class MetricEvaluator:
    def __init__(self, nli_model="cross-encoder/nli-deberta-v3-base", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("[System] NPU/other devices are not supported here; using GPU if available.")
        if self.device != "cuda":
            print("[System] CUDA not available; falling back to CPU.")

        print(f"[Evaluator] Initializing on {self.device}...")

        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(self.device)
            self.nli_model.eval()
        except Exception as e:
            print(f"[Warning] NLI model load failed: {e}")
            self.nli_model = None
            self.nli_tokenizer = None

    def evaluate_batch(self, generated_summaries, ground_truths):
        results = {"time_iou": [], "rouge_l": [], "bert_score": [], "grounding_score": []}

        gen_texts, ref_texts = [], []

        for gen, ref in zip(generated_summaries, ground_truths):
            results["time_iou"].append(self._compute_temporal_iou(gen, ref))

            gen_flat = self._flatten_summary(gen)
            ref_flat = self._flatten_summary(ref)

            r_scores = self.rouge.score(ref_flat, gen_flat)
            results["rouge_l"].append(r_scores["rougeL"].fmeasure)

            results["grounding_score"].append(self._compute_nli_score(gen_flat, ref_flat))

            gen_texts.append(gen_flat)
            ref_texts.append(ref_flat)

        if gen_texts:
            try:
                _, _, F1 = bert_score_func(
                    gen_texts,
                    ref_texts,
                    lang="en",
                    verbose=False,
                    device=self.device,
                    batch_size=32,
                )
                results["bert_score"] = F1.tolist()
            except Exception as e:
                print(f"[Error] BERTScore failed: {e}")
                results["bert_score"] = [0.0] * len(gen_texts)

        aggregated = {k: float(np.mean(v)) for k, v in results.items() if v}
        return aggregated, results

    def _compute_temporal_iou(self, gen_json, ref_json):
        def extract_ranges(data):
            text = json.dumps(data)
            matches = re.findall(r"\[(\d+):(\d+)-(\d+):(\d+)\]", text)
            ranges = []
            for m in matches:
                start = int(m[0]) * 60 + int(m[1])
                end = int(m[2]) * 60 + int(m[3])
                if end > start:
                    ranges.append((start, end))
            return ranges

        gen_ranges = extract_ranges(gen_json)
        ref_ranges = extract_ranges(ref_json)
        if not gen_ranges or not ref_ranges:
            return 0.0

        intersection = 0.0
        union = 0.0

        for g_start, g_end in gen_ranges:
            union += (g_end - g_start)
            max_ov = 0.0
            for r_start, r_end in ref_ranges:
                inter_s = max(g_start, r_start)
                inter_e = min(g_end, r_end)
                max_ov = max(max_ov, max(0.0, inter_e - inter_s))
            intersection += max_ov

        for r_start, r_end in ref_ranges:
            union += (r_end - r_start)

        union = union - intersection
        return (intersection / union) if union > 0 else 0.0

    def _compute_nli_score(self, hypothesis, premise):
        if not self.nli_model or not self.nli_tokenizer or not hypothesis or not premise:
            return 0.5

        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        return float(probs[0][1].item())

    def _flatten_summary(self, data):
        if isinstance(data, str):
            return data
        parts = []
        for _, v in data.items():
            if isinstance(v, list):
                parts.extend([str(x) for x in v])
            elif isinstance(v, str):
                parts.append(v)
        return " ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_path", type=str, default="sample_preds.json")
    parser.add_argument("--gt_path", type=str, default="sample_gt.json")
    parser.add_argument("--out_report", type=str, default="sample_eval_metrics.json")

    args = parser.parse_args()

    evaluator = MetricEvaluator()

    if os.path.exists(args.pred_path) and os.path.exists(args.gt_path):
        with open(args.pred_path, "r") as f:
            preds = json.load(f)
        with open(args.gt_path, "r") as f:
            gts = json.load(f)

        if isinstance(preds, dict):
            preds = [preds]
        if isinstance(gts, dict):
            gts = [gts]

        min_len = min(len(preds), len(gts))
        preds, gts = preds[:min_len], gts[:min_len]

        print(f"[Evaluator] Computing metrics for {min_len} samples...")

        agg_metrics, _ = evaluator.evaluate_batch(preds, gts)

        print("\n=== Evaluation Results ===")
        for k, v in agg_metrics.items():
            print(f"{k.upper()}: {v:.4f}")

        os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
        with open(args.out_report, "w") as f:
            json.dump(agg_metrics, f, indent=4)

        print(f"[Output] Report saved to {args.out_report}")
    else:
        print("[Error] Input files not found.")

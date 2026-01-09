import os
import sys
import json
import logging
import difflib
import torch
import unicodedata
import argparse
from typing import Dict, Any
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import warnings
warnings.simplefilter("ignore")


class SummaryFormatter:
    def __init__(self, model_name="Qwen/Qwen2.5-72B-Instruct", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            logging.getLogger("SummaryFormatter").info(
                "[Formatter] NPU/other devices are not supported here; using GPU if available."
            )
        if self.device != "cuda":
            logging.getLogger("SummaryFormatter").info("[Formatter] CUDA not available; falling back to CPU.")

        self.logger = logging.getLogger("SummaryFormatter")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.logger.info(f"[Formatter] Tokenizer loaded on {self.device}")
        except Exception as e:
            self.logger.warning(f"[Formatter] Tokenizer load failed: {e}. Token counts will be unavailable.")
            self.tokenizer = None

    def format_for_llm(self, state_dict: Dict[str, Any]) -> str:
        cleaned = self._sanitize_dict(state_dict)
        return json.dumps(cleaned, indent=2, ensure_ascii=False)

    def format_for_display(self, state_dict: Dict[str, Any]) -> str:
        lines = ["# Summary Report", ""]

        topic = state_dict.get("topic_or_subject", "N/A")
        lines.append(f"## Topic: {topic}")
        lines.append("")

        sections = [
            ("Background", "background_or_context"),
            ("Problem", "problem_or_motivation"),
            ("Methodology", "method_or_progression"),
            ("Results", "results_or_highlights"),
        ]

        for title, key in sections:
            lines.append(f"### {title}")
            items = state_dict.get(key, [])
            if not items:
                lines.append("  No content")
            else:
                for item in items:
                    lines.append(f"- {item}")
            lines.append("")

        concl = state_dict.get("conclusion_or_impact", "N/A")
        lines.append("## Conclusion")
        lines.append(str(concl))

        return "\n".join(lines)

    def format_linear(self, state_dict: Dict[str, Any]) -> str:
        parts = []
        topic = state_dict.get("topic_or_subject")
        if topic:
            parts.append(f"Topic: {topic}")

        for key in ["method_or_progression", "results_or_highlights"]:
            items = state_dict.get(key, [])
            if items:
                content = "; ".join([str(x) for x in items])
                parts.append(f"{key.split('_')[0].capitalize()}: {content}")

        return " | ".join(parts)

    def compute_token_count(self, text: str) -> int:
        if not self.tokenizer:
            return len(text.split())

        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text)

    def generate_diff(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> str:
        old_str = json.dumps(old_state, indent=2, sort_keys=True).splitlines()
        new_str = json.dumps(new_state, indent=2, sort_keys=True).splitlines()

        diff = difflib.unified_diff(old_str, new_str, fromfile="S_{t-1}", tofile="S_t", lineterm="")
        return "\n".join(diff)

    def _sanitize_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._sanitize_dict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_dict(v) for v in data]
        if isinstance(data, str):
            return unicodedata.normalize("NFKC", data).strip()
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_prev", type=str, default="sample_state_prev.json")
    parser.add_argument("--json_curr", type=str, default="sample_state_curr.json")
    parser.add_argument("--out_report", type=str, default="sample_summary.md")

    args = parser.parse_args()

    formatter = SummaryFormatter(model_name="gpt2")

    s1 = {"topic_or_subject": "Alpha", "method_or_progression": ["[00:10] Init"]}
    s2 = {"topic_or_subject": "Alpha", "method_or_progression": ["[00:10] Init", "[00:20] Update"]}

    if os.path.exists(args.json_prev) and os.path.exists(args.json_curr):
        with open(args.json_prev, "r") as f:
            s1 = json.load(f)
        with open(args.json_curr, "r") as f:
            s2 = json.load(f)

    llm_str = formatter.format_for_llm(s2)
    tokens = formatter.compute_token_count(llm_str)
    print(f"[Formatter] LLM JSON Tokens: {tokens}")

    diff_output = formatter.generate_diff(s1, s2)
    print(f"\n[Formatter] State Update Diff:\n{diff_output}")

    md_report = formatter.format_for_display(s2)

    os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
    with open(args.out_report, "w") as f:
        f.write(md_report)

    print(f"[Success] Markdown report saved to {args.out_report}")

import os
import sys
import json
import logging
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from rl.reward import TriAlignmentReward

logger = logging.getLogger("CriticAgent")
logging.basicConfig(level=logging.INFO)


class CriticAgent:
    def __init__(self, device="cuda", acceptance_threshold=0.8, strict_mode=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            logger.info("[Critic] NPU/other devices are not supported here; using GPU if available.")
        if self.device != "cuda":
            logger.info("[Critic] CUDA not available; falling back to CPU.")

        self.threshold = acceptance_threshold
        self.strict_mode = strict_mode

        logger.info(f"[Critic] Initializing Reward Engine on {self.device}...")
        self.reward_engine = TriAlignmentReward()

        self.score_history = []

    def evaluate(self, draft_json, evidence_context, iteration_idx=0):
        total_score, raw_feedback = self.reward_engine.compute_reward(draft_json, evidence_context)
        self.score_history.append(total_score)

        if iteration_idx > 1:
            delta = self.score_history[-1] - self.score_history[-2]
            if abs(delta) < 0.01 and total_score > (self.threshold * 0.8):
                logger.info(f"[Critic] Score converged ({total_score:.3f}). Accepting.")
                return total_score, "Converged.", True, {}

        if total_score >= self.threshold:
            logger.info(f"[Critic] Threshold met ({total_score:.3f} >= {self.threshold}).")
            return total_score, "Accept.", True, {}

        diagnosis, intent = self._analyze_failure(raw_feedback)
        return total_score, diagnosis, False, intent

    def _analyze_failure(self, raw_feedback: str):
        diagnosis_parts = []
        retrieval_intent = {"query_type": "general", "focus_modality": "both"}

        feedback_lower = (raw_feedback or "").lower()

        if "[hallucination]" in feedback_lower or "unsupported" in feedback_lower:
            diagnosis_parts.append(
                "Critical factual error: entities or numbers are not supported by the evidence."
            )
            retrieval_intent["query_type"] = "verification"
            retrieval_intent["focus_modality"] = "visual" if "visual" in feedback_lower else "text"

        elif "[misalignment]" in feedback_lower or "temporal" in feedback_lower:
            diagnosis_parts.append("Alignment error: timestamps or visual descriptions do not match events.")
            retrieval_intent["query_type"] = "temporal"
            retrieval_intent["focus_modality"] = "visual"

        elif "[incoherence]" in feedback_lower:
            diagnosis_parts.append("Coherence issue: the narrative is repetitive or illogical.")
            retrieval_intent["query_type"] = "context"

        if not diagnosis_parts:
            diagnosis_parts.append(raw_feedback)

        return " | ".join(diagnosis_parts), retrieval_intent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_path", type=str, default="sample_draft.json")

    args = parser.parse_args()

    critic = CriticAgent(acceptance_threshold=0.85)

    evidence = {
        "text_content": "The experiment shows 85% accuracy.",
        "visual_desc": "A bar chart showing 85%.",
    }

    draft = {"results_or_highlights": ["The experiment shows 99% accuracy."]}

    score, diag, stop, intent = critic.evaluate(draft, evidence)

    print(f"Score: {score}")
    print(f"Diagnosis: {diag}")
    print(f"Stop Signal: {stop}")
    print(f"Retrieval Intent: {json.dumps(intent, indent=2)}")

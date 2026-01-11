import os
import sys
import json
import argparse
import logging
import torch
from tqdm import tqdm
from typing import Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from llm.llm_client import simple_chat
from llm.prompt import PromptManager
from llm.response_parser import ResponseParser
from rl.critic import CriticAgent
from knowledge_base.retriever import CriticRetriever
from summary.builder import SummaryBuilder
from rl.trajectory_logger import TrajectoryLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ICRL_Runner")


class ICRLRunner:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ICRL Runner on {self.device}...")

        self.prompt_mgr = PromptManager()
        self.parser = ResponseParser()
        self.builder = SummaryBuilder()

        self.critic = CriticAgent(
            device=self.device,
            acceptance_threshold=config.get("reward_threshold", 0.75),
        )

        db_path = config.get("evidence_db_path")
        if db_path and os.path.exists(db_path):
            self.retriever = CriticRetriever(
                db_path=db_path,
                device=self.device,
                score_threshold=0.25,
            )
        else:
            logger.warning(f"Evidence DB not found at {db_path}. Retrieval disabled.")
            self.retriever = None

        self.traj_logger = TrajectoryLogger(config.get("log_dir"))

        self.max_iters = config.get("max_refinement_steps", 2)
        self.model_name = config.get("llm_model") or "gemini-3"

    def run(self, merged_windows_path: str, output_path: str):
        if not os.path.exists(merged_windows_path):
            logger.error("Input file missing.")
            return

        with open(merged_windows_path, "r") as f:
            windows = json.load(f)

        logger.info(f"Starting Inference on {len(windows)} macro-windows...")

        for i, win in enumerate(tqdm(windows, desc="Processing Windows")):
            self._process_single_window(i, win)
            if (i + 1) % 5 == 0:
                self._save_checkpoint(output_path)

        self._save_checkpoint(output_path)
        self.traj_logger.save()
        logger.info("Pipeline Finished.")

    def _process_single_window(self, win_idx: int, window_data: Dict[str, Any]):
        current_global_state = self.builder.export_json()
        time_range = window_data.get("time_range")

        asr_text = window_data.get("text_content", "")
        visual_paths = window_data.get("image_paths", [])

        sys_p, user_p = self.prompt_mgr.build_update_prompt(
            json.loads(current_global_state),
            asr_text,
            "[Visuals provided via image inputs]",
            time_range,
        )

        response = simple_chat(
            model=self.model_name,
            prompt=user_p,
            system=sys_p,
            images=visual_paths,
            json_mode=True,
        )

        draft_json, _ = self.parser.parse_summary_update(response.text)
        if not draft_json:
            logger.warning(f"Win {win_idx}: Failed to parse initial draft. Skipping.")
            return

        final_draft = draft_json
        evidence_ctx = {
            "text_content": asr_text,
            "visual_desc": f"Visuals at {time_range}",
            "time_range": time_range,
        }

        for k in range(self.max_iters + 1):
            score, diagnosis, stop, ret_intent = self.critic.evaluate(
                final_draft, evidence_ctx, iteration_idx=k
            )

            self.traj_logger.log_iteration(
                window_id=str(win_idx),
                iteration_k=k,
                draft_state=final_draft,
                scores={"total": score},
                diagnosis=diagnosis,
            )

            if stop:
                break

            retrieved_info = ""
            if self.retriever and ret_intent:
                logger.info(f"Win {win_idx} Iter {k}: Retrieving evidence for {ret_intent['query_type']}...")
                retrieved_info = self.retriever.retrieve_evidence(
                    diagnosis,
                    final_draft,
                    time_range,
                )

            sys_fb, user_fb = self.prompt_mgr.build_feedback_prompt(
                json.dumps(final_draft, ensure_ascii=False),
                f"Diagnosis: {diagnosis}\nEVIDENCE FROM DATABASE:\n{retrieved_info}",
            )

            refine_resp = simple_chat(
                model=self.model_name,
                prompt=user_fb,
                system=sys_fb,
                json_mode=True,
            )

            refined_json, _ = self.parser.parse_summary_update(refine_resp.text)
            if refined_json:
                final_draft = refined_json
            else:
                logger.warning(f"Win {win_idx}: Refinement failed to parse. Keeping previous.")
                break

        self.builder.update_state(final_draft)

    def _save_checkpoint(self, path: str):
        data = self.builder.state.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICRL-Sum Inference Pipeline")

    parser.add_argument("--input_merged", type=str, default="sample_merged.json")
    parser.add_argument("--evidence_db", type=str, default="sample_index")
    parser.add_argument("--output_summary", type=str, default="sample_summary.json")
    parser.add_argument("--log_dir", type=str, default="sample_logs")

    parser.add_argument("--model", type=str, default="gemini-3", help="gemini-3, gpt-5")
    parser.add_argument("--iters", type=int, default=3)

    args = parser.parse_args()

    config = {
        "evidence_db_path": args.evidence_db,
        "log_dir": args.log_dir,
        "max_refinement_steps": args.iters,
        "llm_model": args.model,
        "reward_threshold": 0.85,
    }

    runner = ICRLRunner(config)
    runner.run(args.input_merged, args.output_summary)
 

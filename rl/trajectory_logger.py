import os
import sys
import json
import time
import torch
import logging
import uuid
import argparse
from typing import Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class TrajectoryLogger:
    def __init__(self, output_dir: str, session_id: str = None):
        self.output_dir = output_dir
        self.session_id = session_id if session_id else str(uuid.uuid4())[:8]
        self.log_buffer = {}
        self.start_time = time.time()

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger("TrajectoryLogger")
        self.logger.setLevel(logging.INFO)

    def log_iteration(
        self,
        window_id: str,
        iteration_k: int,
        draft_state: Dict[str, Any],
        scores: Dict[str, float],
        diagnosis: str,
        retrieved_evidence: Optional[str] = None,
    ):
        if window_id not in self.log_buffer:
            self.log_buffer[window_id] = {
                "window_id": window_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "iterations": [],
            }

        gpu_stats = self._get_gpu_stats()

        record = {
            "k": iteration_k,
            "time_offset": round(time.time() - self.start_time, 4),
            "gpu_mem_allocated_mb": gpu_stats["allocated"],
            "gpu_mem_reserved_mb": gpu_stats["reserved"],
            "metrics": scores,
            "critic_feedback": diagnosis,
            "retrieval_context_snapshot": (retrieved_evidence[:200] + "...") if retrieved_evidence else None,
            "draft_summary": draft_state,
        }

        self.log_buffer[window_id]["iterations"].append(record)

    def save(self):
        filename = f"sample_trajectory_{self.session_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        temp_path = filepath + ".tmp"

        try:
            with open(temp_path, "w") as f:
                json.dump(self.log_buffer, f, indent=2, ensure_ascii=False)

            os.replace(temp_path, filepath)
            self.logger.info(f"[Logger] Saved {len(self.log_buffer)} trajectories to {filepath}")
        except Exception as e:
            self.logger.error(f"[Logger] Save failed: {e}")

    def _get_gpu_stats(self) -> Dict[str, float]:
        stats = {"allocated": 0.0, "reserved": 0.0}
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                stats["allocated"] = round(torch.cuda.memory_allocated(dev) / 1024**2, 2)
                stats["reserved"] = round(torch.cuda.memory_reserved(dev) / 1024**2, 2)
            except Exception:
                pass
        return stats

    def get_summary_stats(self):
        total_wins = len(self.log_buffer)
        if total_wins == 0:
            return {}

        total_iters = sum(len(w["iterations"]) for w in self.log_buffer.values())
        avg_iters = total_iters / total_wins

        return {
            "total_windows": total_wins,
            "avg_iterations": round(avg_iters, 2),
            "gpu_active": torch.cuda.is_available(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="sample_logs")

    args = parser.parse_args()

    logger = TrajectoryLogger(output_dir=args.log_dir, session_id="test_sess_01")

    print("[Test] Simulating ICRL Loop on GPU...")

    logger.log_iteration(
        window_id="win_001",
        iteration_k=0,
        draft_state={"results": "99% accuracy"},
        scores={"total": 0.4, "alignment": 0.2},
        diagnosis="Hallucination detected.",
        retrieved_evidence="Visual: Bar chart shows 85%.",
    )

    time.sleep(0.1)

    logger.log_iteration(
        window_id="win_001",
        iteration_k=1,
        draft_state={"results": "85% accuracy"},
        scores={"total": 0.95, "alignment": 1.0},
        diagnosis="Accept.",
        retrieved_evidence=None,
    )

    logger.save()

    stats = logger.get_summary_stats()
    print(f"\n[Stats] {json.dumps(stats, indent=2)}")

import os
import sys
import json
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from knowledge_base.vector_store import VectorStore


class CriticRetriever:
    def __init__(self, db_path, model_name="all-mpnet-base-v2", device="cuda", score_threshold=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("[System] NPU/other devices are not supported here; using GPU if available.")
        if self.device != "cuda":
            print("[System] CUDA not available; falling back to CPU.")

        self.threshold = score_threshold

        print(f"[Retriever] Loading Query Encoder {model_name} on {self.device}...")
        try:
            self.encoder = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            print(f"[Fatal] Encoder load failed: {e}")
            sys.exit(1)

        print(f"[Retriever] Loading Evidence Database from {db_path}...")
        self.store = VectorStore(dim=768, use_gpu=(self.device == "cuda"))
        try:
            self.store.load(db_path)
        except Exception as e:
            print(f"[Fatal] DB load failed: {e}")
            sys.exit(1)

    def retrieve_evidence(self, diagnosis, draft_json, current_time_range, top_k=3):
        query_text = self._formulate_query(diagnosis, draft_json)

        with torch.no_grad():
            query_vec = self.encoder.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        raw_results = self.store.search(query_vec, k=top_k * 2)
        refined_results = self._temporal_rerank(raw_results, current_time_range)

        final_evidence = []
        for res in refined_results[:top_k]:
            if res["score"] < self.threshold:
                continue

            meta = res["metadata"]
            evidence_str = (
                f"[Source ID: {meta.get('id')}] "
                f"Time: {meta.get('time_range')} | "
                f"Visual: {meta.get('image_paths', ['No Image'])[0]} | "
                f"Text: {meta.get('text_content')}"
            )
            final_evidence.append(evidence_str)

        return "\n".join(final_evidence) if final_evidence else "No relevant high-confidence evidence found."

    def _formulate_query(self, diagnosis, draft_json):
        diag_lower = diagnosis.lower()
        draft_content = " ".join([str(v) for k, v in draft_json.items() if isinstance(v, str)])

        if "hallucination" in diag_lower or "unsupported" in diag_lower:
            return f"Visual evidence and factual details regarding: {draft_content}"

        if "misalignment" in diag_lower or "time" in diag_lower:
            return f"Timestamped events and scene changes related to: {draft_content}"

        return f"Context and background for: {draft_content}"

    def _temporal_rerank(self, results, current_time_range):
        if not current_time_range:
            return results

        try:
            curr_start, curr_end = map(float, current_time_range.split("-"))
            curr_mid = (curr_start + curr_end) / 2
        except Exception:
            return results

        reranked = []
        decay_factor = 0.01

        for res in results:
            meta = res["metadata"]
            res_range = meta.get("time_range", "0-0")
            try:
                r_start, r_end = map(float, res_range.split("-"))
                r_mid = (r_start + r_end) / 2
                dist = abs(curr_mid - r_mid)
                res["adjusted_score"] = res["score"] / (1 + decay_factor * dist)
            except Exception:
                res["adjusted_score"] = res["score"]
            reranked.append(res)

        reranked.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return reranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critic-Guided Retriever")

    parser.add_argument("--db_path", type=str, default="sample_index")
    parser.add_argument("--query", type=str, default="hallucination: model achieved 99% accuracy")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if os.path.exists(args.db_path):
        retriever = CriticRetriever(args.db_path, device=args.device)

        dummy_draft = {"results": "The model achieved 99% accuracy."}
        dummy_time = "10.00-20.00"

        print(f"[Test] Diagnosing: {args.query}")
        evidence = retriever.retrieve_evidence(args.query, dummy_draft, dummy_time)

        print("\n=== Retrieved Evidence ===")
        print(evidence)
    else:
        print(f"[Error] DB not found at {args.db_path}")

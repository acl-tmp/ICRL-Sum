import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from knowledge_base.vector_store import VectorStore


class EvidenceIndexer:
    def __init__(self, model_name="all-mpnet-base-v2", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("[System] NPU/other devices are not supported here; using GPU if available.")
        if self.device != "cuda":
            print("[System] CUDA not available; falling back to CPU.")

        print(f"[Indexer] Loading Encoder {model_name} on {self.device}...")

        try:
            self.encoder = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            print(f"[Fatal] Failed to load encoder: {e}")
            sys.exit(1)

        self.store = VectorStore(dim=768, use_gpu=(self.device == "cuda"))

    def build_index(self, data_path, output_dir, batch_size=32):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "r") as f:
            windows = json.load(f)

        print(f"[Indexer] Processing {len(windows)} evidence windows...")

        corpus_texts = []
        metadatas = []
        for win in windows:
            text_content = win.get("text_content", "")
            combined_text = f"Time: {win.get('time_range', '')} | Content: {text_content}"
            corpus_texts.append(combined_text)
            metadatas.append(win)

        total = len(corpus_texts)
        all_embeddings = []
        for i in tqdm(range(0, total, batch_size), desc="Encoding"):
            batch_texts = corpus_texts[i : i + batch_size]
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            all_embeddings.append(embeddings)

        if not all_embeddings:
            print("[Warning] No embeddings generated.")
            return

        final_embeddings = np.vstack(all_embeddings)

        print(f"[Indexer] Adding {final_embeddings.shape[0]} vectors to store...")
        self.store.add_batch(final_embeddings, metadatas)

        self.store.save(output_dir)
        print(f"[Success] Evidence Database saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Evidence Indexer")

    parser.add_argument("--json_in", type=str, default="sample_merged.json")
    parser.add_argument("--db_out", type=str, default="sample_index")
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if not os.path.exists(args.json_in) and "sample" in args.json_in:
        print("[System] Default paths not found. Please provide valid inputs.")
        sys.exit(0)

    indexer = EvidenceIndexer(model_name=args.model, device=args.device)
    indexer.build_index(args.json_in, args.db_out)

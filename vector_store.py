import os
import sys
import json
import pickle
import numpy as np
import faiss
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class VectorStore:
    def __init__(self, dim=768, use_gpu=True):
        self.dim = dim
        self.use_gpu = use_gpu
        self.metadata_map = {}
        self.current_id = 0

        self.cpu_index = faiss.IndexFlatIP(dim)

        if self.use_gpu:
            try:
                self.res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
                print(f"[VectorStore] Initialized FAISS on GPU (Dim={dim}).")
            except Exception as e:
                print(f"[VectorStore] GPU init failed: {e}. Falling back to CPU.")
                self.index = self.cpu_index
                self.use_gpu = False
        else:
            self.index = self.cpu_index
            print(f"[VectorStore] Initialized FAISS on CPU (Dim={dim}).")

    def add_batch(self, vectors, metadatas):
        if len(vectors) != len(metadatas):
            raise ValueError("Vectors and metadata count mismatch.")

        vectors = np.ascontiguousarray(vectors.astype("float32"))
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        for meta in metadatas:
            self.metadata_map[self.current_id] = meta
            self.current_id += 1

        return True

    def search(self, query_vector, k=5):
        query_vector = np.ascontiguousarray(query_vector.astype("float32"))
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for i, idx_row in enumerate(indices):
            row_results = []
            for j, idx in enumerate(idx_row):
                if idx == -1:
                    continue
                row_results.append(
                    {
                        "score": float(scores[i][j]),
                        "metadata": self.metadata_map.get(idx, {}),
                    }
                )
            results.append(row_results)

        return results[0] if results else []

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        index_cpu = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(index_cpu, os.path.join(save_dir, "index.faiss"))

        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(
                {"map": self.metadata_map, "curr_id": self.current_id, "dim": self.dim}, f
            )

        print(f"[VectorStore] Persisted to {save_dir}")

    def load(self, save_dir):
        idx_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "metadata.pkl")

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or Metadata missing.")

        self.cpu_index = faiss.read_index(idx_path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        else:
            self.index = self.cpu_index

        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.metadata_map = data["map"]
            self.current_id = data["curr_id"]
            self.dim = data["dim"]

        print(f"[VectorStore] Loaded index with {self.index.ntotal} vectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_json", type=str, default="sample_features.json")
    parser.add_argument("--db_out", type=str, default="sample_db")

    args = parser.parse_args()

    if os.path.exists(args.feature_json):
        store = VectorStore(dim=768, use_gpu=True)

        with open(args.feature_json, "r") as f:
            segments = json.load(f)

        vecs = []
        metas = []

        for seg in segments:
            npy_path = seg.get("audio_feature_path")
            if npy_path and os.path.exists(npy_path):
                try:
                    vec = np.load(npy_path)
                    if vec.shape[0] == 768:
                        vecs.append(vec)
                        metas.append(seg)
                except Exception:
                    pass

        if vecs:
            vec_np = np.stack(vecs)
            store.add_batch(vec_np, metas)
            store.save(args.db_out)
            print(f"[Success] Built DB with {len(vecs)} entries.")

            print("\n[Self-Test] Querying with first vector...")
            res = store.search(vec_np[0])
            for hit in res:
                meta = hit["metadata"]
                print(f"  Score: {hit['score']:.4f} | ID: {meta.get('id')}")
        else:
            print("[Warning] No valid vectors found.")
    else:
        print("[Error] Input JSON not found.")

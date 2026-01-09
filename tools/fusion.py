import os
import json
import numpy as np
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class WindowFuser:
    def __init__(self, min_tokens=500, max_duration=120.0, max_images=6):
        self.min_tokens = min_tokens
        self.max_duration = max_duration
        self.max_images = max_images

    def process_batch(self, segments):
        if not segments:
            return []

        merged_windows = []
        buffer = {"start": None, "end": 0.0, "text_parts": [], "img_paths": [], "src_ids": []}
        current_token_count = 0
        total_segs = len(segments)

        print(f"[Fusion] Processing {total_segs} segments...")

        for i, seg in enumerate(segments):
            if buffer["start"] is None:
                buffer["start"] = seg["start"]

            text = seg.get("asr_text", "").strip()
            if text:
                buffer["text_parts"].append(text)
                current_token_count += len(text.split())

            raw_imgs = seg.get("visual_frame_paths", [])
            valid_imgs = [p for p in raw_imgs if p and os.path.exists(p)]
            buffer["img_paths"].extend(valid_imgs)

            buffer["src_ids"].append(seg["id"])
            buffer["end"] = seg["end"]

            curr_duration = buffer["end"] - buffer["start"]
            cond_len = current_token_count >= self.min_tokens
            cond_time = curr_duration >= self.max_duration
            cond_last = i == total_segs - 1

            if cond_len or cond_time or cond_last:
                if not buffer["text_parts"] and not buffer["img_paths"] and not cond_last:
                    continue

                merged_windows.append(self._create_macro_window(buffer, len(merged_windows)))

                buffer = {"start": None, "end": 0.0, "text_parts": [], "img_paths": [], "src_ids": []}
                current_token_count = 0

        return merged_windows

    def _create_macro_window(self, buffer, new_id):
        final_text = " ".join(buffer["text_parts"])
        sampled_imgs = self._uniform_sample(buffer["img_paths"])

        return {
            "id": new_id,
            "time_range": f"{buffer['start']:.2f}-{buffer['end']:.2f}",
            "duration": round(buffer["end"] - buffer["start"], 2),
            "text_content": final_text if final_text else "[No Speech]",
            "image_paths": sampled_imgs,
            "source_window_ids": buffer["src_ids"],
        }

    def _uniform_sample(self, img_list):
        if not img_list:
            return []

        total = len(img_list)
        if total <= self.max_images:
            return img_list

        indices = np.linspace(0, total - 1, self.max_images, dtype=int)

        sampled = []
        seen = set()
        for idx in indices:
            path = img_list[idx]
            if path not in seen:
                sampled.append(path)
                seen.add(path)

        return sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal Window Fusion Tool")

    parser.add_argument("--json_in", type=str, default="sample_features.json")
    parser.add_argument("--json_out", type=str, default="sample_merged.json")

    parser.add_argument("--min_tokens", type=int, default=300)
    parser.add_argument("--max_duration", type=float, default=120.0)
    parser.add_argument("--max_images", type=int, default=6)

    args = parser.parse_args()

    if os.path.exists(args.json_in):
        with open(args.json_in, "r") as f:
            raw_data = json.load(f)

        fuser = WindowFuser(
            min_tokens=args.min_tokens,
            max_duration=args.max_duration,
            max_images=args.max_images,
        )

        merged_data = fuser.process_batch(raw_data)

        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(merged_data, f, indent=4)

        print(f"[Success] Fused {len(raw_data)} segments into {len(merged_data)} macro-windows.")
        print(f"[Output] {args.json_out}")

        if merged_data:
            w = merged_data[0]
            print(f"\n[Preview Window 0] Duration: {w['duration']}s | Images: {len(w['image_paths'])}")
    else:
        print(f"[Error] File not found: {args.json_in}")

import os
import cv2
import json
import argparse
import sys
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class FrameExtractor:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu

    def extract_batch(self, video_path, segments, output_dir):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        tasks = []
        for seg in segments:
            timestamps = seg.get("visual_frame_timestamps", [])
            if not timestamps:
                continue
            for i, ts in enumerate(timestamps):
                tasks.append({"timestamp": ts, "win_id": seg["id"], "seq_idx": i, "seg_ref": seg})

        tasks.sort(key=lambda x: x["timestamp"])

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_root = os.path.join(output_dir, video_name)
        os.makedirs(save_root, exist_ok=True)

        print(f"[FrameExtractor] Processing {len(tasks)} frames from {video_name}...")

        extracted_paths_map = {}

        try:
            for task in tqdm(tasks, unit="frame"):
                target_ts = task["timestamp"]
                target_frame_idx = int(target_ts * fps)

                if target_frame_idx >= total_frames_video:
                    target_frame_idx = total_frames_video - 1

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()

                if not ret and target_frame_idx < total_frames_video - 1:
                    ret, frame = cap.read()

                if not ret:
                    continue

                file_name = f"sample_win_{task['win_id']}_seq_{task['seq_idx']}.jpg"
                file_path = os.path.join(save_root, file_name)
                cv2.imwrite(file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                wid = task["win_id"]
                extracted_paths_map.setdefault(wid, {})[task["seq_idx"]] = file_path

        finally:
            cap.release()

        for seg in segments:
            wid = seg["id"]
            ts_count = len(seg.get("visual_frame_timestamps", []))
            if wid not in extracted_paths_map:
                seg["visual_frame_paths"] = []
                continue

            ordered_paths = []
            for i in range(ts_count):
                p = extracted_paths_map[wid].get(i)
                if p:
                    ordered_paths.append(p)
            seg["visual_frame_paths"] = ordered_paths

        return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient Video Frame Extractor")

    parser.add_argument("--video", type=str, default="sample_video.mp4")
    parser.add_argument("--json_in", type=str, default="sample_seg.json")
    parser.add_argument("--json_out", type=str, default="sample_frames.json")
    parser.add_argument("--img_dir", type=str, default="sample_frames_dir")

    args = parser.parse_args()

    if not os.path.exists(args.json_in) and "sample" in args.json_in:
        print("[System] Default paths not found. Please provide valid arguments.")
        sys.exit(0)

    extractor = FrameExtractor(use_gpu=True)

    if os.path.exists(args.json_in) and os.path.exists(args.video):
        with open(args.json_in, "r") as f:
            segments_data = json.load(f)

        updated_data = extractor.extract_batch(args.video, segments_data, args.img_dir)

        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(updated_data, f, indent=4)

        print(f"[Success] Extracted frames. Saved metadata to: {args.json_out}")
    else:
        print(f"[Error] Input files missing: {args.video} or {args.json_in}")

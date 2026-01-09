import os
import sys
import gc
import warnings
import cv2
import json
import numpy as np
import argparse
import librosa
import logging
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile.*", category=UserWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

logging.getLogger("librosa").setLevel(logging.ERROR)


class MultimediaSegmenter:
    def __init__(
        self,
        scene_threshold=30.0,
        min_duration=2.0,
        silence_thresh_db=-40.0,
        speech_break_duration=0.8,
        visual_buffer=1.0,
    ):
        self.scene_threshold = scene_threshold
        self.min_duration = min_duration
        self.silence_thresh_db = silence_thresh_db
        self.speech_break_duration = speech_break_duration
        self.visual_buffer = visual_buffer

    def _plan_visual_frames(self, start, end):
        duration = end - start
        b = self.visual_buffer

        if duration < (b * 2):
            return [round(start + duration / 2, 3)]

        if duration < 15.0:
            return [round(start + b, 3)]

        return [round(start + b, 3), round(end - b, 3)]

    def _load_audio_energy(self, video_path):
        try:
            y, sr = librosa.load(video_path, sr=16000)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            db = librosa.amplitude_to_db(rms, ref=np.max)
            times = librosa.times_like(rms, sr=sr, hop_length=512)
            return times, db
        except Exception as e:
            print(f"[Segmenter] Audio load warning: {e}. Processing as video only.")
            return None, None

    def process(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"[Segmenter] Processing: {video_path}")
        print("[Segmenter] Analyzing audio track for speech breaks...")
        audio_times, audio_db = self._load_audio_energy(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        windows = []
        prev_hist = None
        curr_scene_start_frame = 0

        consecutive_silence_time = 0.0
        last_time_sec = 0.0

        print(f"[Segmenter] Scanning {total_frames} frames...")

        check_interval = 5
        for curr_frame_idx in range(0, total_frames, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            curr_time_sec = curr_frame_idx / fps if fps > 0 else 0.0
            time_step = curr_time_sec - last_time_sec
            last_time_sec = curr_time_sec

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

            scene_changed = False
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                if diff > self.scene_threshold:
                    scene_changed = True
            prev_hist = hist

            is_silence_now = False
            if audio_db is not None and audio_times is not None:
                idx = np.searchsorted(audio_times, curr_time_sec)
                if idx < len(audio_db) and audio_db[idx] < self.silence_thresh_db:
                    is_silence_now = True

            if is_silence_now:
                consecutive_silence_time += max(0.0, time_step)
            else:
                consecutive_silence_time = 0.0

            segment_duration = curr_time_sec - (curr_scene_start_frame / fps if fps > 0 else 0.0)
            should_cut = False

            if segment_duration >= self.min_duration:
                if scene_changed or consecutive_silence_time > self.speech_break_duration:
                    should_cut = True

            if should_cut:
                self._add_segment(windows, curr_scene_start_frame, curr_frame_idx, fps)
                curr_scene_start_frame = curr_frame_idx
                consecutive_silence_time = 0.0

        self._add_segment(windows, curr_scene_start_frame, total_frames, fps)
        cap.release()

        self._refine_segments_silence(windows, audio_times, audio_db)
        return windows

    def _add_segment(self, windows_list, start_frame, end_frame, fps):
        if fps <= 0:
            return

        start_sec = float(round(start_frame / fps, 3))
        end_sec = float(round(end_frame / fps, 3))
        if end_sec <= start_sec:
            return

        visual_frames = [float(t) for t in self._plan_visual_frames(start_sec, end_sec)]
        windows_list.append(
            {
                "id": len(windows_list),
                "start": start_sec,
                "end": end_sec,
                "duration": round(end_sec - start_sec, 3),
                "is_silent": False,
                "visual_frame_timestamps": visual_frames,
            }
        )

    def _refine_segments_silence(self, windows, audio_times, audio_db):
        if audio_db is None or audio_times is None:
            for w in windows:
                w["is_silent"] = True
            return

        for w in windows:
            start_idx = np.searchsorted(audio_times, w["start"])
            end_idx = np.searchsorted(audio_times, w["end"])

            if start_idx >= end_idx:
                w["is_silent"] = True
                continue

            segment_dbs = audio_db[start_idx:end_idx]
            if len(segment_dbs) == 0:
                w["is_silent"] = True
                continue

            w["is_silent"] = bool(np.max(segment_dbs) < self.silence_thresh_db)

    def save_segments(self, windows, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(windows, f, indent=4)
        print(f"[Segmenter] Saved {len(windows)} segments to {output_path}")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] -> {dev}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="sample_video.mp4", help="Path to video")
    parser.add_argument("--out", type=str, default="sample_segments.json", help="Path to output json")
    parser.add_argument("--speech_break", type=float, default=0.8)
    parser.add_argument("--silence_db", type=float, default=-40.0)
    parser.add_argument("--visual_buffer", type=float, default=1.0)

    args = parser.parse_args()

    if args.out is None or args.out.strip() == "":
        video_base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.out = os.path.join("sample_out_dir", f"{video_base_name}_seg.json")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    segmenter = MultimediaSegmenter(
        scene_threshold=5000.0,
        min_duration=2.0,
        silence_thresh_db=args.silence_db,
        speech_break_duration=args.speech_break,
        visual_buffer=args.visual_buffer,
    )

    if os.path.exists(args.video):
        try:
            segments = segmenter.process(args.video)
            print(f"\n=== Processed {len(segments)} Windows ===")
            if segments:
                print(f"Example: Window 0 -> Frames: {segments[0]['visual_frame_timestamps']}")
            segmenter.save_segments(segments, args.out)
            print(f"[Output] Segmentation saved to: {args.out}")
        except Exception:
            import traceback

            traceback.print_exc()
    else:
        print(f"[Error] Video not found: {args.video}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

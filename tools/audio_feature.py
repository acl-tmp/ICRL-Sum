import os
import sys
import json
import torch
import torchaudio
import numpy as np
import argparse
import warnings
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

warnings.simplefilter("ignore")


class AudioFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            print("[System] NPU/other devices are not supported here; using GPU if available.")
        if self.device != "cuda":
            print("[System] CUDA not available; falling back to CPU.")

        print(f"[AudioFeature] Initializing Wav2Vec2 on {self.device}...")

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[Fatal] Model loading failed: {e}")
            sys.exit(1)

        self.target_sr = 16000

    def process_video(self, video_path, segments, output_dir):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"[AudioFeature] Loading audio track from {os.path.basename(video_path)}...")
        try:
            waveform, sr = torchaudio.load(video_path)
        except Exception as e:
            print(f"[Error] Audio decoding failed: {e}")
            return segments

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze()
        total_samples = waveform.shape[0]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        feature_save_dir = os.path.join(output_dir, video_name)
        os.makedirs(feature_save_dir, exist_ok=True)

        print(f"[AudioFeature] Extracting features for {len(segments)} segments...")

        with torch.no_grad():
            for seg in tqdm(segments, unit="seg"):
                if seg.get("is_silent", False):
                    dummy_vec = np.zeros((768,), dtype=np.float32)
                    seg["audio_feature_path"] = self._save_npy(dummy_vec, seg["id"], feature_save_dir)
                    continue

                start_sample = int(seg["start"] * self.target_sr)
                end_sample = int(seg["end"] * self.target_sr)

                start_sample = max(0, start_sample)
                end_sample = min(total_samples, end_sample)

                if end_sample - start_sample < 160:
                    seg["audio_feature_path"] = ""
                    continue

                chunk = waveform[start_sample:end_sample]

                input_values = self.processor(
                    chunk, sampling_rate=self.target_sr, return_tensors="pt"
                ).input_values.to(self.device)

                outputs = self.model(input_values)
                hidden_states = outputs.last_hidden_state

                pooled = torch.mean(hidden_states, dim=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                emb_np = pooled.detach().cpu().numpy().squeeze()
                seg["audio_feature_path"] = self._save_npy(emb_np, seg["id"], feature_save_dir)

        return segments

    def _save_npy(self, array, seg_id, folder):
        filename = f"sample_win_{seg_id}_audio_emb.npy"
        path = os.path.join(folder, filename)
        np.save(path, array)
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 Feature Extractor")

    parser.add_argument("--video", type=str, default="sample_video.mp4")
    parser.add_argument("--json_in", type=str, default="sample_features.json")
    parser.add_argument("--json_out", type=str, default="sample_features_plus_audio.json")
    parser.add_argument("--feature_dir", type=str, default="sample_audio_embeddings")

    args = parser.parse_args()

    if not os.path.exists(args.json_in) and "sample" in args.json_in:
        print("[System] Default paths not found. Please provide valid arguments.")
        sys.exit(0)

    if os.path.exists(args.json_in) and os.path.exists(args.video):
        extractor = AudioFeatureExtractor()

        with open(args.json_in, "r") as f:
            segments = json.load(f)

        updated_segments = extractor.process_video(args.video, segments, args.feature_dir)

        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(updated_segments, f, indent=4)

        print(f"[Success] Audio features extracted. Metadata saved to: {args.json_out}")
    else:
        print("[Error] Input files missing.")

import os
import sys
import argparse
import json
import gc
import time
import torch
import logging
from typing import Any

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from segmenter.segmenter import MultimediaSegmenter
from tools.frame_extractor import FrameExtractor
from tools.asr import ASRTool
from tools.audio_feature import AudioFeatureExtractor
from tools.fusion import WindowFuser
from knowledge_base.indexer import EvidenceIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Pipeline")


class PreprocessingPipeline:
    def __init__(self, output_root: str, force_rerun: bool = False):
        self.output_root = output_root
        self.force_rerun = force_rerun
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized Pipeline on {self.device}")

    def run(self, video_path: str):
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        dirs = {
            "root": os.path.join(self.output_root, video_name),
            "frames": os.path.join(self.output_root, "window_features", "frames"),
            "audio_emb": os.path.join(self.output_root, "window_features", "audio_embeddings"),
            "db": os.path.join(self.output_root, "evidence_db", video_name),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        artifacts = {
            "segments": os.path.join(dirs["root"], "1_segments.json"),
            "frames": os.path.join(dirs["root"], "2_frames_meta.json"),
            "asr": os.path.join(dirs["root"], "3_asr_meta.json"),
            "features": os.path.join(dirs["root"], "4_features_meta.json"),
            "merged": os.path.join(dirs["root"], "5_merged_windows.json"),
            "index_done": os.path.join(dirs["db"], "index.faiss"),
        }

        segments = self._run_stage(
            "Segmentation",
            artifacts["segments"],
            lambda: MultimediaSegmenter().process(video_path),
        )

        def run_visual():
            extractor = FrameExtractor(use_gpu=(self.device == "cuda"))
            return extractor.extract_batch(video_path, segments, dirs["frames"])

        segments = self._run_stage("Frame Extraction", artifacts["frames"], run_visual)

        def run_asr():
            asr = ASRTool(model_name="whisper-1")
            return asr.run_batch(video_path, segments)

        segments = self._run_stage("ASR Transcription", artifacts["asr"], run_asr)

        def run_audio_emb():
            self._cleanup_gpu()
            extractor = AudioFeatureExtractor(device=self.device)
            return extractor.process_video(video_path, segments, dirs["audio_emb"])

        segments = self._run_stage("Audio Embedding", artifacts["features"], run_audio_emb)

        def run_fusion():
            fuser = WindowFuser(min_tokens=300, max_duration=120.0, max_images=6)
            return fuser.process_batch(segments)

        _ = self._run_stage("Window Fusion", artifacts["merged"], run_fusion)

        if not os.path.exists(artifacts["index_done"]) or self.force_rerun:
            logger.info(">>> Stage 6: Building Evidence Database")
            self._cleanup_gpu()
            try:
                indexer = EvidenceIndexer(device=self.device)
                indexer.build_index(artifacts["merged"], dirs["db"])
                logger.info(f"Evidence Index persisted at {dirs['db']}")
            except Exception as e:
                logger.error(f"Indexing Failed: {e}")
                raise
        else:
            logger.info(f"[Skip] Evidence Index exists at {dirs['db']}")

        logger.info(f"Pipeline Complete. Final Output: {artifacts['merged']}")

    def _run_stage(self, name: str, output_path: str, func) -> Any:
        if os.path.exists(output_path) and not self.force_rerun:
            logger.info(f"[Skip] Stage '{name}' cached: {output_path}")
            with open(output_path, "r") as f:
                return json.load(f)

        logger.info(f">>> Stage: {name}")
        start_t = time.time()

        try:
            result = func()
            with open(output_path, "w") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            logger.info(f"Stage '{name}' finished in {time.time() - start_t:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Stage '{name}' Failed: {e}")
            self._cleanup_gpu()
            sys.exit(1)

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICRL-Sum Data Preprocessing Pipeline")

    parser.add_argument("--video", type=str, default="sample_video.mp4")
    parser.add_argument("--out_dir", type=str, default="sample_out_dir")
    parser.add_argument("--force", action="store_true", help="Force rerun existing stages")

    args = parser.parse_args()

    try:
        pipeline = PreprocessingPipeline(args.out_dir, force_rerun=args.force)
        pipeline.run(args.video)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled Exception: {e}")
        sys.exit(1)

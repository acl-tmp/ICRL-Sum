import os
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import shutil
import subprocess
import tempfile
import warnings
import time
from typing import List, Dict, Any

warnings.simplefilter("ignore")

try:
    from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
except ImportError:
    print("[ASRTool] Error: 'openai' library not installed. Please install it via pip.")
    sys.exit(1)

logger = logging.getLogger("ASRTool")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ASRTool:
    def __init__(self, api_key: str = None, model_name: str = "whisper-1", max_retries: int = 3):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("[ASRTool] Fatal: API Key not found. Set OPENAI_API_KEY env var.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.max_retries = max_retries

        logger.info(f"[ASRTool] Initialized. Using Remote Model: {self.model_name}")

    def run_batch(self, video_path: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"[ASRTool] Processing {len(segments)} segments using Whisper API...")

        temp_dir = tempfile.mkdtemp(prefix="sample_asr_")

        try:
            processed_count = 0
            total_cost_seconds = 0.0

            for i, seg in enumerate(segments):
                seg_id = seg.get("id", i)

                if seg.get("is_silent", False):
                    seg["asr_text"] = ""
                    continue

                if seg.get("asr_text"):
                    continue

                chunk_filename = f"sample_chunk_{seg_id}.mp3"
                chunk_path = os.path.join(temp_dir, chunk_filename)

                start = seg.get("start", 0.0)
                duration = seg.get("duration", 0.0)

                if duration < 0.1:
                    seg["asr_text"] = ""
                    continue

                if self._extract_audio_chunk(video_path, start, duration, chunk_path):
                    transcription = self._transcribe_with_retry(chunk_path)

                    if transcription:
                        seg["asr_text"] = transcription
                        processed_count += 1
                        total_cost_seconds += duration

                        if processed_count % 5 == 0:
                            logger.info(
                                f"  -> Processed {processed_count} chunks (Total audio: {total_cost_seconds:.1f}s)..."
                            )
                    else:
                        seg["asr_text"] = ""
                        logger.warning(f"Window {seg_id}: Transcription failed.")
                else:
                    seg["asr_text"] = ""
                    logger.error(f"Window {seg_id}: Audio extraction failed.")

        finally:
            logger.info(f"[ASRTool] Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

        return segments

    def _transcribe_with_retry(self, audio_path: str) -> str:
        retries = 0
        backoff = 2

        while retries <= self.max_retries:
            try:
                with open(audio_path, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model_name,
                        file=audio_file,
                        response_format="text",
                        temperature=0.0,
                    )
                return transcript.strip()

            except RateLimitError:
                logger.warning(f"[API] Rate limit hit. Retrying in {backoff}s...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2

            except APIConnectionError:
                logger.warning(f"[API] Connection error. Retrying in {backoff}s...")
                time.sleep(backoff)
                retries += 1

            except APIStatusError as e:
                logger.error(f"[API] Status error: {e}")
                return None

            except Exception as e:
                logger.error(f"[API] Fatal Error: {e}")
                return None

        logger.error("[API] Max retries exceeded.")
        return None

    def _extract_audio_chunk(self, video_path: str, start: float, duration: float, output_path: str) -> bool:
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(duration),
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "4",
                "-loglevel",
                "error",
                output_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
        except Exception as e:
            logger.error(f"[FFmpeg] Exception: {e}")
            return False

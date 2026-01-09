import json
import os
from copy import deepcopy

from llm.llm_client import call_local_vl_qwen, simple_chat
from llm.prompt import PromptManager
from summary.schema import AcademicSummary
from rl.reward import TriAlignmentReward


class InContextRL:
    def __init__(
        self,
        vl_model_path: str,
        brain_model_name: str = "gemini-3",
        max_retries: int = 2,
        reward_threshold: float = 0.75,
    ):
        self.vl_model_path = vl_model_path
        self.brain_model_name = brain_model_name
        self.max_retries = max_retries
        self.threshold = reward_threshold

        self.prompt_mgr = PromptManager()
        self.reward_engine = TriAlignmentReward()
        self.global_state = AcademicSummary().to_dict()

        print(
            f"[ICRL] Initialized. Eye: {os.path.basename(vl_model_path)} | Brain: {brain_model_name}"
        )

    def run(self, merged_windows, output_dir: str):
        trajectory_logs = []

        print(f"[ICRL] Processing {len(merged_windows)} macro-windows...")

        for i, window in enumerate(merged_windows):
            print(f"\n{'=' * 40}")
            print(f"Window {i} | Time: {window['time_range']} | Duration: {window['duration']}s")

            visual_desc = "[No visual input]"
            images = window.get("image_paths", [])

            if images:
                print(f"  [Eye] Analyzing {len(images)} frames on GPU...")
                try:
                    vis_prompt = self.prompt_mgr.build_visual_perception_prompt()
                    vl_out = call_local_vl_qwen(
                        model_dir=self.vl_model_path,
                        system="You are an expert visual observer.",
                        user=vis_prompt,
                        images=images,
                        max_new_tokens=256,
                        temperature=0.1,
                    )
                    visual_desc = vl_out.text
                    print(f"  [Eye] Observation: {visual_desc[:80]}...")
                except Exception as e:
                    print(f"  [Eye] Error: {e}")
                    visual_desc = "[Visual analysis failed]"

            evidence_context = {
                "text_content": window.get("text_content", ""),
                "visual_desc": visual_desc,
                "duration": window.get("duration"),
                "time_range": window.get("time_range"),
            }

            sys_p, user_p = self.prompt_mgr.build_update_prompt(
                self.global_state,
                evidence_context["text_content"],
                evidence_context["visual_desc"],
                evidence_context["time_range"],
            )

            print(f"  [Brain] Generating initial draft with {self.brain_model_name}...")
            draft_json = self._call_brain_generate(sys_p, user_p)
            if not draft_json:
                print("  [Brain] Failed to generate valid Schema. Skipping window.")
                continue

            final_json = draft_json

            for k in range(self.max_retries + 1):
                score, diagnosis = self.reward_engine.compute_reward(final_json, evidence_context)

                trajectory_logs.append(
                    {
                        "window_id": i,
                        "iteration": k,
                        "score": score,
                        "diagnosis": diagnosis,
                        "state_snapshot": deepcopy(final_json),
                    }
                )

                print(f"  [Eval] Iter {k} | Score: {score:.3f} | Diagnosis: {diagnosis}")

                if score >= self.threshold:
                    print("  [Success] Quality threshold met.")
                    break

                if k == self.max_retries:
                    print("  [Stop] Max iterations reached.")
                    break

                print("  [Refine] Triggering Critic-Guided Retrieval...")
                retrieved_evidence = self._retrieve_evidence(diagnosis, evidence_context)

                sys_fb, user_fb = self.prompt_mgr.build_feedback_prompt(
                    json.dumps(final_json, ensure_ascii=False),
                    f"Diagnosis: {diagnosis}\nEVIDENCE: {retrieved_evidence}",
                )

                refined_json = self._call_brain_generate(sys_fb, user_fb)
                if refined_json:
                    final_json = refined_json

            self.global_state = final_json
            print(f"  [Update] Global State Updated. Current Topic: {self.global_state.get('topic', 'Unknown')}")

            self._save_checkpoint(output_dir, f"sample_checkpoint_win_{i}.json")

        return self.global_state, trajectory_logs

    def _retrieve_evidence(self, diagnosis: str, context: dict) -> str:
        diagnosis_lower = (diagnosis or "").lower()

        if "hallucination" in diagnosis_lower or "visual" in diagnosis_lower:
            return f"Visual Observation: {context['visual_desc']}"

        if "misalignment" in diagnosis_lower or "timestamp" in diagnosis_lower:
            return f"Time Range: {context['time_range']} | ASR Transcript: {context['text_content']}"

        return f"Visual: {context['visual_desc']} | Text: {context['text_content']}"

    def _call_brain_generate(self, sys_prompt: str, user_prompt: str):
        try:
            resp = simple_chat(
                model=self.brain_model_name,
                prompt=user_prompt,
                system=sys_prompt,
                temperature=0.2,
                max_tokens=2048,
            )
            return self._parse_json_robust(resp.text)
        except Exception as e:
            print(f"  [Brain Error] {e}")
            return None

    def _parse_json_robust(self, text: str):
        text = (text or "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            if "```" in text:
                content = text.split("```json")[-1] if "```json" in text else text.split("```")[-1]
                content = content.split("```")[0].strip()
                return json.loads(content)

            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            return None

        return None

    def _save_checkpoint(self, out_dir: str, filename: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, filename), "w") as f:
            json.dump(self.global_state, f, indent=4, ensure_ascii=False)

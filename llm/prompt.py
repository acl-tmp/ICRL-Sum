import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from summary.schema import AcademicSummary


class PromptManager:
    def __init__(self):
        self.schema_keys = list(AcademicSummary().to_dict().keys())
        self.keys_formatted = "\n    - ".join(self.schema_keys)

    def build_visual_perception_prompt(self):
        user_prompt = (
            "Analyze the visual content of this academic video frame.\n"
            "Focus on extracting Textual Evidence and Scene Dynamics:\n"
            "1. Slides: Transcribe the main title and key bullet points verbatim.\n"
            "2. Figures: If charts/graphs exist, describe axes, trends, and key values.\n"
            "3. Actions: If a demo is shown, describe the action and objects used.\n"
            "4. Scene: Identify the setting.\n"
            "Output concise, objective observations. No filler."
        )
        return user_prompt

    def build_update_prompt(self, current_state, asr_text, visual_desc, time_range):
        state_str = json.dumps(current_state, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are an expert academic video summarizer. Maintain a structured JSON summary that evolves over time. "
            "Strictly follow the schema. Ground updates in the provided evidence. Do not invent details."
        )

        user_prompt = f"""
        ### Context
        We are processing a time window {time_range} of an academic presentation.

        ### 1. Multimodal Evidence
        Audio Stream:
        {asr_text}

        Visual Stream:
        {visual_desc}

        ### 2. Historical State
        ```json
        {state_str}
        3. Task

        Update the JSON state using the new evidence:

        Topic: refine if the slide title explicitly changes.

        Events: add to method_or_progression or results_or_highlights only if a distinct event occurs.

        Timestamps: prefix new event strings with {time_range}.

        Grounding: every added detail must be supported by audio or visuals.

        4. Constraints

        Output only a valid JSON object.

        Fields:

        {self.keys_formatted}
        """
        return system_prompt, user_prompt

def build_feedback_prompt(self, draft_json_str, critique_and_evidence):
    system_prompt = (
    "You are a strict editor for academic summaries. The draft failed the ICRL-Sum quality check. "
    "Fix it using only the diagnosis and evidence."
    )

    user_prompt = f"""

    Rejected Draft
    {draft_json_str}

    Critic Diagnosis and Evidence

    {critique_and_evidence}

    Instructions

    Remove anything not supported by the evidence.

    If there is a timestamp or visual mismatch, rewrite to match the evidence.

    Keep it coherent.

    Return the corrected JSON only.
    """
    return system_prompt, user_prompt
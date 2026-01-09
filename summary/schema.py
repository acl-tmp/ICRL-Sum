import re
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger("SchemaValidator")


class TimeSpan(BaseModel):
    start: float
    end: float
    raw_str: Optional[str] = None

    @model_validator(mode="after")
    def validate_order(self):
        if self.end < self.start:
            self.start, self.end = self.end, self.start
        return self

    @property
    def duration(self):
        return self.end - self.start


class AcademicSummary(BaseModel):
    topic_or_subject: str = Field(default="", description="High-level phrase summarizing the central theme.")

    background_or_context: List[str] = Field(default_factory=list, description="Establishing context or initial setup.")
    problem_or_motivation: List[str] = Field(default_factory=list, description="Triggering problem or motivation.")
    method_or_progression: List[str] = Field(default_factory=list, description="Core actions or procedural steps.")
    results_or_highlights: List[str] = Field(default_factory=list, description="Key outcomes or salient events.")

    conclusion_or_impact: str = Field(default="", description="Final conclusion or takeaway.")

    @staticmethod
    def parse_timestamp(text: str) -> Optional[TimeSpan]:
        match = re.search(r"\[(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\]", text)
        if match:
            m1, s1, m2, s2 = map(int, match.groups())
            start = m1 * 60 + s1
            end = m2 * 60 + s2
            return TimeSpan(start=float(start), end=float(end), raw_str=match.group(0))

        match_sec = re.search(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]", text)
        if match_sec:
            start, end = map(float, match_sec.groups())
            return TimeSpan(start=start, end=end, raw_str=match_sec.group(0))

        return None

    def get_event_timestamps(self, field_name: str) -> List[Optional[TimeSpan]]:
        if not hasattr(self, field_name):
            return []
        events = getattr(self, field_name)
        return [self.parse_timestamp(e) for e in events]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_prompt_string(self) -> str:
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False)

    @field_validator(
        "background_or_context",
        "problem_or_motivation",
        "method_or_progression",
        "results_or_highlights",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    def merge_update(self, partial_update: Dict[str, Any]):
        if partial_update.get("topic_or_subject"):
            self.topic_or_subject = partial_update["topic_or_subject"]

        if partial_update.get("conclusion_or_impact"):
            self.conclusion_or_impact = partial_update["conclusion_or_impact"]

        for field in [
            "background_or_context",
            "problem_or_motivation",
            "method_or_progression",
            "results_or_highlights",
        ]:
            new_items = partial_update.get(field, [])
            if isinstance(new_items, list):
                current_list = getattr(self, field)
                current_list.extend(new_items)
                setattr(self, field, current_list)

        return self


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_out", type=str, default="sample_schema_test.json")
    args = parser.parse_args()

    raw_llm_output = {
        "topic_or_subject": "Deep Learning Optimization",
        "method_or_progression": [
            "[00:10-00:20] The speaker introduces SGD.",
            "He then compares it with Adam [01:00-01:15].",
        ],
        "results_or_highlights": "Accuracy improved by 5%.",
    }

    print("[Schema] Validating raw input...")

    try:
        summary = AcademicSummary(**raw_llm_output)
        timestamps = summary.get_event_timestamps("method_or_progression")

        print(f"Topic: {summary.topic_or_subject}")
        print(f"Results: {summary.results_or_highlights}")

        for i, ts in enumerate(timestamps):
            if ts:
                print(f"Event {i} Duration: {ts.duration}s (Raw: {ts.raw_str})")
            else:
                print(f"Event {i} has no timestamp.")

        os.makedirs(os.path.dirname(args.test_out) or ".", exist_ok=True)
        with open(args.test_out, "w") as f:
            f.write(summary.to_prompt_string())

        print(f"[Success] Schema validated and saved to {args.test_out}")

    except Exception as e:
        print(f"[Error] Validation failed: {e}")

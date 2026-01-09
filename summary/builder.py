import os
import sys
import json
import logging
from difflib import SequenceMatcher
from typing import Dict, Any, List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from summary.schema import AcademicSummary

logger = logging.getLogger("SummaryBuilder")


class SummaryBuilder:
    def __init__(self, initial_state: Dict[str, Any] = None):
        if initial_state:
            try:
                self.state = AcademicSummary(**initial_state)
            except Exception as e:
                logger.error(f"State load failed: {e}. Resetting.")
                self.state = AcademicSummary()
        else:
            self.state = AcademicSummary()

    def update_state(self, partial_update: Dict[str, Any], window_meta: Dict[str, Any] = None):
        if not partial_update:
            return self.state.to_dict()

        self._merge_global_field("topic_or_subject", partial_update.get("topic_or_subject"))
        self._merge_global_field("conclusion_or_impact", partial_update.get("conclusion_or_impact"))

        event_keys = [
            "background_or_context",
            "problem_or_motivation",
            "method_or_progression",
            "results_or_highlights",
        ]

        for key in event_keys:
            new_items = partial_update.get(key, [])
            if not new_items:
                continue
            if isinstance(new_items, str):
                new_items = [new_items]

            current_list = getattr(self.state, key)
            setattr(self.state, key, self._merge_event_list(current_list, new_items))

        return self.state.to_dict()

    def _merge_global_field(self, field_name: str, new_value: str):
        if not new_value or not isinstance(new_value, str):
            return

        current_value = getattr(self.state, field_name)
        if not current_value or len(new_value) > len(current_value):
            setattr(self.state, field_name, new_value)
            return

        if len(new_value) > 10 and new_value != current_value:
            setattr(self.state, field_name, new_value)

    def _merge_event_list(self, current: List[str], new_items: List[str]) -> List[str]:
        merged = list(current)

        for item in new_items:
            if not self._is_duplicate(merged, item):
                merged.append(item)

        def get_start_time(text: str):
            ts = AcademicSummary.parse_timestamp(text)
            return ts.start if ts else 99999.0

        merged.sort(key=get_start_time)
        return merged

    def _is_duplicate(self, current_list: List[str], new_item: str, threshold=0.85) -> bool:
        if new_item in current_list:
            return True

        for existing in current_list:
            if SequenceMatcher(None, existing, new_item).ratio() > threshold:
                return True
        return False

    def export_json(self):
        return self.state.to_prompt_string()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="sample_builder_test.json")
    args = parser.parse_args()

    builder = SummaryBuilder()

    u1 = {
        "topic_or_subject": "Intro to Transformers",
        "method_or_progression": ["[00:00-00:10] Speaker defines Attention mechanism."],
    }
    builder.update_state(u1)

    u2 = {
        "topic_or_subject": "Transformers and Self-Attention",
        "method_or_progression": [
            "[00:00-00:10] Defining Attention mechanism.",
            "[00:15-00:20] Explaining Multi-head attention.",
        ],
        "results_or_highlights": ["[00:30-00:40] SOTA results achieved."],
    }
    final_state = builder.update_state(u2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final_state, f, indent=4)

    print(f"[Success] State updated. Topic: {final_state['topic_or_subject']}")
    print(f"[Events] Method count: {len(final_state['method_or_progression'])}")

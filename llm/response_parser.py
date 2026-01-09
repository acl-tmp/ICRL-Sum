import json
import re
import logging
from typing import Dict, Any, Optional, Tuple

try:
    from summary.schema import AcademicSummary

    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False

logger = logging.getLogger("ResponseParser")

CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


class ResponseParser:
    def __init__(self):
        self.json_decoder = json.JSONDecoder()

    def parse_summary_update(self, raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        reasoning, potential_json_str = self._separate_reasoning_and_content(raw_text)
        parsed_data = self._robust_json_extract(potential_json_str)

        if not parsed_data:
            logger.warning("Failed to extract valid JSON from response.")
            return None, reasoning

        if SCHEMA_AVAILABLE:
            parsed_data = self._enforce_schema_constraints(parsed_data)

        return parsed_data, reasoning

    def parse_feedback_instruction(self, raw_text: str) -> str:
        patterns = [
            r"(?i)Refinement Instructions?:?\s*(.*)",
            r"(?i)Instruction:?\s*(.*)",
            r"(?i)Correction:?\s*(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return raw_text.replace("**", "").replace("###", "").strip()

    def _separate_reasoning_and_content(self, text: str) -> Tuple[str, str]:
        m = CODE_BLOCK_RE.search(text)
        if m:
            reasoning = text[: m.start()].strip()
            return reasoning, m.group(1).strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[:start_idx].strip(), text[start_idx : end_idx + 1]

        return "", text

    def _robust_json_extract(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            fixed_text = self._heuristic_fix_json(text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass

        logger.error(f"JSON Parsing failed. Raw text snippet: {text[:50]}...")
        return None

    def _heuristic_fix_json(self, text: str) -> str:
        text = re.sub(r",\s*([\]}])", r"\1", text)
        text = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', text)
        return text

    def _enforce_schema_constraints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        template = AcademicSummary().to_dict()
        valid_keys = set(template.keys())

        cleaned = {}
        for key, value in data.items():
            if key not in valid_keys:
                continue

            expected_type = type(template[key])
            if expected_type is list and isinstance(value, str):
                cleaned[key] = [value]
            elif expected_type is str and isinstance(value, list):
                cleaned[key] = " ".join([str(v) for v in value])
            else:
                cleaned[key] = value

        for key in valid_keys:
            if key not in cleaned:
                cleaned[key] = template[key]

        return cleaned


if __name__ == "__main__":
    parser = ResponseParser()

    raw_response = """
    Based on the visual evidence, the speaker is showing a slide about accuracy.
    Here is the updated schema:
    ```json
    {
        "topic": "Accuracy Results",
        "results_or_highlights": ["Model achieved 85% accuracy."],
        "hallucinated_field": "ignore me"
    }
    ```
    """
    print("--- Test 1 ---")
    data, log = parser.parse_summary_update(raw_response)
    print(f"Log: {log}")
    print(f"Data: {json.dumps(data, indent=2)}")

    bad_json = """
    {
        "method_or_progression": ["Step 1", "Step 2",],
        "conclusion": "Done",
    }
    """
    print("\n--- Test 2 ---")
    data, _ = parser.parse_summary_update(bad_json)
    print(f"Fixed Data: {json.dumps(data, indent=2)}")

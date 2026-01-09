import re
import json
import math
from collections import Counter


class TriAlignmentReward:
    def __init__(self):
        print("[Reward] Initialized ICRL-Sum Reward Critic (Rule-based Proxy)")
        self.exclusion_list = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "of",
            "for",
            "to",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "it",
            "by",
            "with",
            "as",
            "be",
            "from",
            "at",
            "which",
            "have",
            "has",
            "we",
            "you",
            "can",
            "not",
            "but",
            "video",
            "clip",
            "shows",
            "scene",
        }
        self.visual_triggers = [
            "show",
            "shown",
            "slide",
            "figure",
            "chart",
            "graph",
            "image",
            "picture",
            "plot",
            "table",
            "scene",
            "shot",
            "view",
            "camera",
        ]

    def compute_reward(self, summary_json, source_context):
        if isinstance(summary_json, str):
            try:
                summary_json = json.loads(summary_json)
            except Exception:
                return 0.0, "Fatal: Invalid JSON format."

        summary_text = " ".join(
            [str(v) for _, v in summary_json.items() if isinstance(v, str) and v]
        )
        if not summary_text.strip():
            return 0.0, "Summary is empty."

        asr_text = source_context.get("text_content", "")
        visual_desc = source_context.get("visual_desc", "")
        full_evidence = f"{asr_text} {visual_desc}"

        r_ground, fb_ground = self._calc_factual_grounding(summary_text, full_evidence)
        r_align, fb_align = self._calc_spatio_temporal_alignment(summary_text, source_context)
        r_cohere, fb_cohere = self._calc_event_coherence(summary_text, asr_text)

        alpha, beta, gamma = 0.35, 0.3, 0.35
        total_score = (alpha * r_align) + (beta * r_cohere) + (gamma * r_ground)

        feedbacks = []
        if fb_ground:
            feedbacks.append(f"[Hallucination]: {fb_ground}")
        if fb_align:
            feedbacks.append(f"[Misalignment]: {fb_align}")
        if fb_cohere:
            feedbacks.append(f"[Incoherence]: {fb_cohere}")

        feedback_str = (
            "Excellent. The summary is grounded, aligned, and coherent."
            if not feedbacks
            else " ".join(feedbacks)
        )
        return round(total_score, 3), feedback_str

    def _calc_factual_grounding(self, summary, source):
        source_lower = source.lower()

        sum_nums = re.findall(r"\b\d+\.?\d*\b", summary)
        hallucinated_nums = []
        for num in sum_nums:
            if len(num) == 2 and int(num) < 60:
                continue
            if num not in source:
                hallucinated_nums.append(num)

        if hallucinated_nums:
            return 0.0, f"Unsupported numbers detected: {hallucinated_nums}."

        sentences = re.split(r"[.!?]", summary)
        hallucinated_entities = []
        for sent in sentences:
            words = sent.strip().split()
            for i, word in enumerate(words):
                clean_word = re.sub(r"[^\w]", "", word)
                if i > 0 and clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                    if clean_word.lower() not in source_lower:
                        hallucinated_entities.append(clean_word)

        if hallucinated_entities:
            penalty_score = max(0.0, 1.0 - len(set(hallucinated_entities)) * 0.3)
            return penalty_score, f"Unsupported entities found: {list(set(hallucinated_entities))}."

        sum_tokens = self._tokenize(summary)
        if len(sum_tokens) < 2:
            return 1.0, ""

        sum_bigrams = set(zip(sum_tokens, sum_tokens[1:]))
        src_tokens = self._tokenize(source)
        src_bigrams = set(zip(src_tokens, src_tokens[1:]))

        overlap = sum_bigrams.intersection(src_bigrams)
        entailment_score = len(overlap) / len(sum_bigrams) if sum_bigrams else 1.0

        if entailment_score < 0.2:
            return entailment_score, "Content phrases diverge significantly from evidence."

        return 1.0, ""

    def _calc_spatio_temporal_alignment(self, summary, source_context):
        time_pattern = r"\[(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\]"

        sum_times = re.findall(time_pattern, summary)
        src_text = source_context.get("text_content", "") + source_context.get("visual_desc", "")
        src_times = re.findall(time_pattern, src_text)

        if sum_times and src_times:
            def to_sec(m, s):
                return int(m) * 60 + int(s)

            total_iou = 0.0
            for st in sum_times:
                s_start, s_end = to_sec(st[0], st[1]), to_sec(st[2], st[3])
                best_span_iou = 0.0
                for gt in src_times:
                    g_start, g_end = to_sec(gt[0], gt[1]), to_sec(gt[2], gt[3])

                    inter_start = max(s_start, g_start)
                    inter_end = min(s_end, g_end)
                    inter_len = max(0.0, inter_end - inter_start)

                    union_start = min(s_start, g_start)
                    union_end = max(s_end, g_end)
                    union_len = max(1e-6, union_end - union_start)

                    iou = inter_len / union_len
                    if iou > best_span_iou:
                        best_span_iou = iou

                total_iou += best_span_iou

            avg_iou = total_iou / len(sum_times)
            if avg_iou < 0.5:
                return avg_iou, f"Low temporal alignment (tIoU={avg_iou:.2f})."
            return avg_iou, ""

        visual_desc = source_context.get("visual_desc", "")
        if not visual_desc:
            return 1.0, ""

        if any(t in summary.lower() for t in self.visual_triggers):
            sum_vec = Counter(self._tokenize(summary))
            vis_vec = Counter(self._tokenize(visual_desc))

            keys = set(sum_vec.keys()).union(vis_vec.keys())
            keys = {k for k in keys if k not in self.exclusion_list}

            dot = sum(sum_vec.get(k, 0) * vis_vec.get(k, 0) for k in keys)
            mag_sum = math.sqrt(sum(v * v for k, v in sum_vec.items() if k in keys))
            mag_vis = math.sqrt(sum(v * v for k, v in vis_vec.items() if k in keys))

            similarity = 0.0 if mag_sum == 0 or mag_vis == 0 else dot / (mag_sum * mag_vis)
            if similarity < 0.3:
                return similarity, "Visual description does not match the visual evidence."
            return 1.0, ""

        return 1.0, ""

    def _calc_event_coherence(self, summary, asr):
        sum_tokens = [t for t in self._tokenize(summary) if t not in self.exclusion_list and len(t) > 2]
        if not sum_tokens:
            return 0.0, "Empty semantic content."

        unique_ratio = len(set(sum_tokens)) / len(sum_tokens)
        if unique_ratio < 0.6:
            return 0.4, "Summary is highly repetitive."

        asr_tokens = [t for t in self._tokenize(asr) if t not in self.exclusion_list and len(t) > 2]
        if not asr_tokens:
            return 1.0, ""

        ctr = Counter(asr_tokens)
        num_keywords = max(3, int(len(ctr) * 0.2))
        top_keywords = {w for w, _ in ctr.most_common(num_keywords)}

        covered = sum(1 for w in top_keywords if w in sum_tokens)
        coverage_score = covered / len(top_keywords) if top_keywords else 1.0

        if coverage_score < 0.3:
            missing_kws = list(top_keywords - set(sum_tokens))[:3]
            return coverage_score, f"Summary misses key topics: {missing_kws}."

        return 1.0, ""

    def _tokenize(self, text):
        return re.findall(r"\w+", (text or "").lower())


if __name__ == "__main__":
    reward_model = TriAlignmentReward()

    print("--- Test Case 1 (High Quality) ---")
    good_sum = {
        "event_1": "[00:00-00:10] The Model X achieved 85% accuracy as shown in the blue chart.",
        "event_2": "[00:10-00:20] This result confirms the hypothesis.",
    }
    good_src = {
        "text_content": "[00:00-00:20] Our experiment with Model X shows 85 percent accuracy. The result confirms the hypothesis.",
        "visual_desc": "[00:00-00:10] A slide showing a blue bar chart of accuracy results.",
    }
    s, f = reward_model.compute_reward(good_sum, good_src)
    print(f"Score: {s}, Feedback: {f}")

    print("\n--- Test Case 2 (Hallucinated Entity & Wrong Visual) ---")
    bad_sum = {
        "description": "Professor Smith presents a red apple.",
        "result": "He claims 99% efficiency.",
    }
    bad_src = {
        "text_content": "The researcher presents the findings. We see 85% efficiency.",
        "visual_desc": "A person holding a green pear.",
    }
    s, f = reward_model.compute_reward(bad_sum, bad_src)
    print(f"Score: {s}, Feedback: {f}")

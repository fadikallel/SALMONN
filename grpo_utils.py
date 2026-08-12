import re
from collections import Counter

def extract_tag(text, tag):
    if text is None:
        return ""

    m = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    return m.group(1).strip() if m else ""


def extract_label(text):
    answer = extract_tag(text, "answer")

    if "bonafide" in answer or "Real" in answer:
        return "bonafide"

    if "spoof" in answer or "Fake" in answer:
        return "spoof"

    return None


def extract_reasoning(text):
    return extract_tag(text, "reasoning").strip().lower()

# ============================================================
# Reasoning Reward
# ============================================================

def token_f1(pred_tokens, gt_tokens):

    pred = Counter(pred_tokens)
    gt = Counter(gt_tokens)

    overlap = sum((pred & gt).values())

    if overlap == 0:
        return 0.0

    precision = overlap / max(sum(pred.values()), 1)
    recall = overlap / max(sum(gt.values()), 1)

    return 2 * precision * recall / (precision + recall)


def reasoning_reward(prediction, ground_truth):

    pred = extract_reasoning(prediction)
    gt = extract_reasoning(ground_truth)

    if pred == "" or gt == "":
        return 0.0

    pred_tokens = re.findall(r"\w+", pred)
    gt_tokens = re.findall(r"\w+", gt)

    return token_f1(pred_tokens, gt_tokens)

# ============================================================
# XML Formatting Reward
# ============================================================
import re

# Each tag must appear exactly once, in order, non-empty, non-nested.
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Full structure: optional leading whitespace, reasoning block, optional whitespace,
# answer block, optional trailing whitespace. Anchored so nothing extra sneaks in.
_STRUCTURE_RE = re.compile(
    r"^\s*<reasoning>(?P<reasoning>.+?)</reasoning>\s*"
    r"<answer>(?P<answer>.+?)</answer>\s*$",
    re.DOTALL,
)


def format_reward(prediction) -> float:
    pred = str(prediction).strip().lower()

    # 1. Each tag must occur exactly once (catches duplicates / missing closers)
    if len(_REASONING_RE.findall(pred)) != 1:
        return 0.0
    if len(_ANSWER_RE.findall(pred)) != 1:
        return 0.0

    # 2. Overall structure must match: reasoning block, then answer block, nothing else around them
    match = _STRUCTURE_RE.match(pred)
    if not match:
        return 0.0

    # 3. Both blocks must have non-empty content (not just whitespace)
    if not match.group("reasoning").strip() or not match.group("answer").strip():
        return 0.0

    return 1.0

def compute_reward(
    prediction_text,
    ground_truth_text,
):

    reward = 0.0

    pred_label = extract_label(prediction_text)
    gt_label = extract_label(ground_truth_text)

    label_correct = (pred_label == gt_label)

    # --------------------------------------------------
    # Correct prediction (highest priority)
    # --------------------------------------------------

    if label_correct:

        reward += 4.0
        reward += 1.0 * reasoning_reward(
            prediction_text,
            ground_truth_text,
        )

    else:

        reward -= 1.0

    # --------------------------------------------------
    # Always encourage valid XML
    # --------------------------------------------------

    reward += 0.5 * format_reward(
        prediction_text,
    )

    return float(reward)
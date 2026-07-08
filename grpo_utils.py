import re
from collections import Counter


# ============================================================
# XML Parsing
# ============================================================

def extract_tag(text, tag):
    if text is None:
        return ""

    m = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    return ""


def extract_label(text):
    answer = extract_tag(text, "answer").lower()

    if "bonafide" in answer:
        return "bonafide"

    if "spoof" in answer:
        return "spoof"

    return None


def extract_explanation(text):
    return extract_tag(text, "explanation").strip().lower()


def extract_reasons(text):
    reasons = extract_tag(text, "reasons")

    if reasons == "":
        return set()

    reasons = reasons.lower()

    parts = re.split(r"[,\n;]", reasons)

    return {
        p.strip()
        for p in parts
        if len(p.strip()) > 0
    }


# ============================================================
# Text Similarity
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


def explanation_reward(prediction, ground_truth):

    pred = extract_explanation(prediction)
    gt = extract_explanation(ground_truth)

    pred_tokens = re.findall(r"\w+", pred)
    gt_tokens = re.findall(r"\w+", gt)

    return token_f1(pred_tokens, gt_tokens)


# ============================================================
# Reason Reward
# ============================================================

def reason_f1(pred_set, gt_set):

    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0

    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & gt_set)

    precision = tp / len(pred_set)

    recall = tp / len(gt_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def reason_reward(prediction, ground_truth):

    gt_label = extract_label(ground_truth)

    pred_reasons = extract_reasons(prediction)
    gt_reasons = extract_reasons(ground_truth)

    # -------------------------
    # Bonafide samples
    # -------------------------

    if gt_label == "bonafide":

        # Bonafide should NOT have reasons.
        if len(pred_reasons) == 0:
            return 1.0

        return 0.0

    # -------------------------
    # Spoof samples
    # -------------------------

    return reason_f1(pred_reasons, gt_reasons)


# ============================================================
# XML Formatting
# ============================================================

def format_reward(prediction):

    pred = str(prediction).lower()

    reward = 0.0

    if "<answer>" in pred and "</answer>" in pred:
        reward += 0.25

    if "<explanation>" in pred and "</explanation>" in pred:
        reward += 0.25

    label = extract_label(prediction)

    # Reasons are required only for spoof.
    if label == "spoof":

        if "<reasons>" in pred and "</reasons>" in pred:
            reward += 0.5

    else:
        # Reward not hallucinating a reasons section.
        if "<reasons>" not in pred:
            reward += 0.5

    return reward


# ============================================================
# Explanation Length
# ============================================================

def length_reward(prediction):

    explanation = extract_explanation(prediction)

    n_words = len(explanation.split())

    if n_words < 5:
        return 0.0

    if n_words <= 20:
        return 0.5

    if n_words <= 60:
        return 1.0

    if n_words <= 100:
        return 0.8

    return 0.5


# ============================================================
# Consistency Reward
# ============================================================

def consistency_reward(prediction):

    label = extract_label(prediction)

    explanation = extract_explanation(prediction)

    reasons = extract_reasons(prediction)

    text = explanation + " " + " ".join(reasons)

    spoof_keywords = [
        "robotic",
        "metallic",
        "electronic",
        "synthetic",
        "artifact",
        "artifacts",
        "unnatural",
        "monotony",
        "monotonous",
        "strange voice",
        "strange intonation",
    ]

    score = 0

    for keyword in spoof_keywords:
        if keyword in text:
            score += 1

    if label == "spoof":

        if score > 0:
            return 1.0

        return 0.5

    # bonafide

    if len(reasons) > 0:
        return 0.0

    return 1.0


# ============================================================
# Final Reward
# ============================================================

def compute_reward(
    prediction_text,
    ground_truth_text,
    reward_funcs=None,
    reward_weights=None,
):

    reward = 0.0

    pred_label = extract_label(prediction_text)
    gt_label = extract_label(ground_truth_text)

    # --------------------------------------------------
    # 1. Correct answer (most important)
    # --------------------------------------------------

    if pred_label == gt_label:
        reward += 3.0

    # --------------------------------------------------
    # 2. Explanation similarity
    # --------------------------------------------------

    reward += 1.0 * explanation_reward(
        prediction_text,
        ground_truth_text,
    )

    # --------------------------------------------------
    # 3. Artifact reason matching
    # --------------------------------------------------

    reward += 3.0 * reason_reward(
        prediction_text,
        ground_truth_text,
    )

    # --------------------------------------------------
    # 4. Output formatting
    # --------------------------------------------------

    reward += 0.5 * format_reward(
        prediction_text,
    )

    # --------------------------------------------------
    # 5. Explanation quality
    # --------------------------------------------------

    reward += 0.5 * length_reward(
        prediction_text,
    )

    # --------------------------------------------------
    # 6. Logical consistency
    # --------------------------------------------------

    reward += 1.0 * consistency_reward(
        prediction_text,
    )

    return float(reward)
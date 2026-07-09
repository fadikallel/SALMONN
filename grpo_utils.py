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

    return m.group(1).strip() if m else ""


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

    parts = re.split(r"[,\n;]", reasons.lower())

    return {
        p.strip()
        for p in parts
        if p.strip()
    }


# ============================================================
# Explanation Reward
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

    if pred == "" or gt == "":
        return 0.0

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

    # Bonafide should not contain reasons
    if gt_label == "bonafide":
        return 1.0 if len(pred_reasons) == 0 else 0.0

    reward = reason_f1(pred_reasons, gt_reasons)

    # Penalize predicting too many reasons
    if len(gt_reasons) > 0 and len(pred_reasons) > len(gt_reasons):
        reward *= len(gt_reasons) / len(pred_reasons)

    return reward


# ============================================================
# XML Formatting Reward
# ============================================================

def format_reward(prediction):

    pred = str(prediction).lower()

    label = extract_label(prediction)

    has_answer = "<answer>" in pred and "</answer>" in pred
    has_explanation = "<explanation>" in pred and "</explanation>" in pred

    if label == "spoof":
        has_reasons = "<reasons>" in pred and "</reasons>" in pred
    else:
        has_reasons = "<reasons>" not in pred

    return float(
        has_answer and
        has_explanation and
        has_reasons
    )


def compute_reward(
    prediction_text,
    ground_truth_text,
    reward_funcs=None,
    reward_weights=None,
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

        reward += 2.5 * reason_reward(
            prediction_text,
            ground_truth_text,
        )

        reward += 1.0 * explanation_reward(
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
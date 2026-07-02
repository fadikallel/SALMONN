import re


def extract_label(text):
    if not text:
        return None

    text_str = str(text)
    answer_match = re.search(r"<answer>(.*?)</answer>", text_str, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip().lower()
    else:
        answer_text = text_str.lower().strip()

    if "bonafide" in answer_text:
        return 1
    if "spoof" in answer_text:
        return 0
    return None


def compute_reward(prediction_text, ground_truth_text, reward_funcs=None, reward_weights=None):
    reward_funcs = reward_funcs or ["correctness"]
    reward_weights = reward_weights or {}

    total_reward = 0.0
    for reward_name in reward_funcs:
        reward_name = reward_name.lower()
        if reward_name == "correctness":
            pred_label = extract_label(prediction_text)
            gt_label = extract_label(ground_truth_text)
            if pred_label is not None and gt_label is not None:
                reward = 1.0 if pred_label == gt_label else 0.0
            else:
                reward = 0.0
        elif reward_name == "format":
            has_answer_tag = "<answer>" in str(prediction_text).lower() and "</answer>" in str(prediction_text).lower()
            reward = 0.25 if has_answer_tag else 0.0
        elif reward_name == "length":
            reward = 0.0
        else:
            reward = 0.0

        weight = reward_weights.get(reward_name, 1.0)
        total_reward += reward * weight

    return total_reward

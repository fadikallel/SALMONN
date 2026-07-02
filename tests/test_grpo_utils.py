import importlib


def test_extract_label_and_reward():
    grpo_utils = importlib.import_module("grpo_utils")

    assert grpo_utils.extract_label("<answer>bonafide</answer>") == 1
    assert grpo_utils.extract_label("<answer>spoof</answer>") == 0
    assert grpo_utils.extract_label("The answer is spoof") == 0

    reward = grpo_utils.compute_reward(
        prediction_text="<answer>bonafide</answer>",
        ground_truth_text="<answer>bonafide</answer>",
        reward_funcs=["correctness", "format"],
    )
    assert reward == 2.0

    reward = grpo_utils.compute_reward(
        prediction_text="<answer>spoof</answer>",
        ground_truth_text="<answer>bonafide</answer>",
        reward_funcs=["correctness", "format"],
    )
    assert reward == 0.0

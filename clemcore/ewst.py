import json
import os
import math
# Some primary statistics from the first reflect run (adapted for wordle only)

def is_valid_number(x):
    return isinstance(x, (int, float)) and not math.isnan(x)

def compute_average_scores(root_dir):
    total_closeness = 0.0
    total_strategy = 0.0
    count = 0
    files_found = 0

    for root, _, files in os.walk(root_dir):
        if "scores.json" in files:
            files_found += 1
            filepath = os.path.join(root, "scores.json")

            with open(filepath, "r") as f:
                data = json.load(f)

            turn_scores = data.get("turn scores", {})

            for turn_data in turn_scores.values():
                closeness = turn_data.get("Closeness Score")
                strategy = turn_data.get("Strategy Score")

                if is_valid_number(closeness) and is_valid_number(strategy):
                    total_closeness += closeness
                    total_strategy += strategy
                    count += 1

    if count == 0:
        raise ValueError("No valid turn scores found (all values were NaN or missing).")

    return {
        "average_closeness": total_closeness / count,
        "average_strategy": total_strategy / count,
        "num_turns_used": count,
        "num_instances": files_found
    }


# Usage
root_path = r"C:\Users\white\Desktop\IM\Tests_for_presentation2\results_no_reflection\qwen3-coder-30b-t1.5\wordle\medium_frequency_words_no_clue_no_critic"

results = compute_average_scores(root_path)

print(f"Instances processed: {results['num_instances']}")
print(f"Turns used: {results['num_turns_used']}")
print(f"Average Closeness Score: {results['average_closeness']:.2f}")
print(f"Average Strategy Score: {results['average_strategy']:.2f}")


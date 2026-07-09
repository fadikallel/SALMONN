import json
from pathlib import Path

def transform_jsonl_to_training_format(input_jsonl_path, output_json_path, task="deepfake_detection_with_reasoning"):
    """
    Transform JSONL with response and audios to JSON with path, text, task
    
    Input format (JSONL):
    {"response": "...", "audios": ["/path/to/audio.wav"], ...}
    
    Output format (JSON):
    [
      {
        "path": "/path/to/audio.wav",
        "text": "...",
        "task": "deepfake_detection_with_reasoning"
      }
    ]
    """
    training_data = []
    
    with open(input_jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Extract the response text
                response = data.get('response', '')
                if not response:
                    print(f"Warning: Line {line_num} has no 'response' field, skipping...")
                    continue
                
                # Extract the audio path from the 'audios' list
                audios = data.get('audios', [])
                if not audios:
                    print(f"Warning: Line {line_num} has no 'audios' field, skipping...")
                    continue
                
                audio_path = audios[0]  # Take the first audio path
                
                # Create the training example
                example = {
                    "path": audio_path,
                    "text": f"{response}",
                    "task": task
                }
                
                training_data.append(example)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Transformed {len(training_data)} samples")
    print(f"   Input: {input_jsonl_path}")
    print(f"   Output: {output_json_path}")
    print(f"   Task: {task}")
    
    return training_data

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_file = "/netscratch/fkallel/reasoning_annotation/result/Qwen3-Omni-30B-A3B-Instruct/infer_result/reasoning_asv19train.jsonl"
    output_file = "data/asv_train_reasoning.json"
    
    # Transform the data
    training_data = transform_jsonl_to_training_format(
        input_jsonl_path=input_file,
        output_json_path=output_file,
        task="deepfake_detection_with_reasoning"
    )
    
    # Print a sample to verify
    if training_data:
        print("\n📝 Sample output:")
        print(json.dumps(training_data[0], indent=2))
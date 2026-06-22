import json
import pandas as pd
from pathlib import Path

def transform_parquet_to_training_format(input_parquet_path, output_json_path, task="deepfake_detection_with_reasoning"):
    """
    Transform Parquet with audio metadata to JSON with path, text, task
    
    Input format (Parquet):
    Columns: audio_id, path, source_corpus, source, reasoning, reasons, 
             is_bonafide, language, marker_id, consistency, 
             annotator_quality, complexity
    
    Output format (JSON):
    [
      {
        "path": "/ds-slt/audio/fkallel/HIR-SDD/audio_path_from_parquet",
        "text": "<answer>{bonafide or spoof}</answer><reasoning>{reasoning}</reasoning><reasons>{reasons}</reasons>",
        "task": "deepfake_detection_with_reasoning"
      }
    ]
    """
    training_data = []
    
    # Base path for audio files
    base_audio_path = "/ds-slt/audio/fkallel/HIR-SDD/"
    
    # Read the parquet file
    try:
        df = pd.read_parquet(input_parquet_path)
        print(f"✅ Loaded {len(df)} rows from parquet")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return []
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Extract required fields
            audio_path = row.get('path', '')
            if not audio_path:
                print(f"Warning: Row {idx} has no 'path' field, skipping...")
                continue
            
            # Convert to absolute path with base directory
            # Remove any leading './' or '../' or existing base paths
            audio_path = audio_path.lstrip('./').lstrip('../')
            # If the path already starts with the base path, use it as is
            absolute_path = base_audio_path + audio_path
            
            # Get bonafide status
            is_bonafide = row.get('is_bonafide', None)
            if is_bonafide is None:
                print(f"Warning: Row {idx} has no 'is_bonafide' field, skipping...")
                continue
            
            # Convert bonafide to answer text
            answer = "bonafide" if is_bonafide else "spoof"
            
            # Get reasoning and reasons
            reasoning = row.get('reasoning', '')
            reasons = row.get('reasons', [])
            
            # Convert reasons list to string if it's a list
            reasons_list = reasons.tolist()
            if len(reasons_list) == 0:
                formatted_text = f"<answer>{answer}</answer><explanation>{reasoning}</explanation>"
            else:
                processed_reasons = []
                for reason in reasons_list:
                    processed = reason.lower().replace('_', ' ')
                    processed_reasons.append(processed)
                reasons_str = ', '.join(processed_reasons) if processed_reasons else ''
                formatted_text = f"<answer>{answer}</answer><explanation>{reasoning}</explanation><reasons>{reasons_str}</reasons>"
            
            # Create the training example
            example = {
                "path": absolute_path,
                "text": formatted_text,
                "task": task
            }
            
            training_data.append(example)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Save to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n✅ Transformed {len(training_data)} samples")
    print(f"   Input: {input_parquet_path}")
    print(f"   Output: {output_json_path}")
    print(f"   Task: {task}")
    
    # Print some statistics
    if training_data:
        bonafide_count = sum(1 for item in training_data if '<answer>bonafide' in item['text'])
        spoof_count = sum(1 for item in training_data if '<answer>spoof' in item['text'])
        print(f"\n📊 Statistics:")
        print(f"   Bonafide samples: {bonafide_count}")
        print(f"   Spoof samples: {spoof_count}")
        
        # Print first few paths as examples
        print(f"\n📁 Sample absolute paths:")
        for i in range(min(3, len(training_data))):
            print(f"   {training_data[i]['path']}")
    
    return training_data

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_parquet = "/ds-slt/audio/fkallel/HIR-SDD/annotations/data.parquet"  # Change this to your parquet file
    output_file = "data/hir-sdd.json"
    
    # Transform the data
    training_data = transform_parquet_to_training_format(
        input_parquet_path=input_parquet,
        output_json_path=output_file,
        task="deepfake_detection_with_reasoning"
    )
    
    # Print a sample to verify
    if training_data:
        print("\n📝 Sample output:")
        print(json.dumps(training_data[0], indent=2))
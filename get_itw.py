import json
import pandas as pd
import numpy as np
from pathlib import Path

def transform_parquet_to_binary_classification(input_parquet_path, output_json_path, task="deepfake_detection"):
    """
    Transform Parquet with audio metadata to JSON with path, text, task
    for binary classification (bonafide vs spoof).
    
    Input format (Parquet):
    Columns: path, id, dataset_name, label (where label is binary 0/1 or string)
    
    Output format (JSON):
    [
      {
        "path": "/ds-slt/audio/fkallel/HIR-SDD/audio_path_from_parquet",
        "text": "<answer>{bonafide or spoof}</answer>",
        "task": "deepfake_detection"
      }
    ]
    """
    training_data = []
    
    
    # Read the parquet file
    try:
        df = pd.read_parquet(input_parquet_path)
        print(f"✅ Loaded {len(df)} rows from parquet")
        print(f"📋 Columns: {df.columns.tolist()}")
        print(f"📊 Data types:\n{df.dtypes}")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return []
    
    # Check if required columns exist
    required_cols = ['path', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {df.columns.tolist()}")
        return []
    
    # Optional: handle id and dataset_name if they exist
    has_id = 'id' in df.columns
    has_dataset_name = 'dataset_name' in df.columns
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Extract required fields
            audio_path = row.get('path')            
            # Get label and convert to answer text
            label = row.get('label')
            
            # Create the text with only the answer (binary classification)
            
            # Create the training example
            example = {
                "path": audio_path,
                "text": label,
                "task": task
            }
            
            # Add optional fields if they exist
            if has_id:
                example["id"] = str(row.get('id', idx))
            if has_dataset_name:
                example["dataset_name"] = str(row.get('dataset_name', ''))
            
            training_data.append(example)
            
            # Print progress every 10000 rows
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1} rows...")
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
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
        bonafide_count = sum(1 for item in training_data if 'bonafide' in item['text'])
        spoof_count = sum(1 for item in training_data if 'spoof' in item['text'])
        print(f"\n📊 Statistics:")
        print(f"   Bonafide samples: {bonafide_count}")
        print(f"   Spoof samples: {spoof_count}")
        
        # Check for dataset distribution if available
        if has_dataset_name:
            dataset_counts = {}
            for item in training_data:
                dataset = item.get('dataset_name', 'unknown')
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            print(f"\n📊 Dataset distribution:")
            for dataset, count in sorted(dataset_counts.items()):
                print(f"   {dataset}: {count}")
        
        # Print first few paths as examples
        print(f"\n📁 Sample absolute paths:")
        for i in range(min(3, len(training_data))):
            sample = training_data[i]
            path_str = sample['path']
            id_str = sample.get('id', 'N/A')
            print(f"   [{id_str}] {path_str}")
            
        # Print a sample to verify formatting
        print(f"\n📝 Sample output:")
        sample = training_data[0]
        print(json.dumps(sample, indent=2))
    
    return training_data


def transform_parquet_to_binary_classification_simple(input_parquet_path, output_json_path, 
                                                      base_audio_path="/ds-slt/audio/fkallel/HIR-SDD/",
                                                      task="deepfake_detection"):
    """
    Simplified version with customizable base path.
    """
    return transform_parquet_to_binary_classification(
        input_parquet_path=input_parquet_path,
        output_json_path=output_json_path,
        task=task
    )


# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_parquet = "/ds-slt/audio/universal_speech_df_data/parquets/InTheWild/itw_eval.parquet"
    output_file = "data/itw.json"
    
    # Transform the data
    training_data = transform_parquet_to_binary_classification(
        input_parquet_path=input_parquet,
        output_json_path=output_file,
        task="deepfake_detection"
    )
    
    # Print summary
    if training_data:
        print("\n✅ Transformation complete!")
        print(f"Total samples: {len(training_data)}")
        
        # Count labels
        bonafide = sum(1 for item in training_data if 'bonafide' in item['text'])
        spoof = len(training_data) - bonafide
        print(f"Bonafide: {bonafide}, Spoof: {spoof}")
        print(f"Ratio: {bonafide/spoof:.2f}" if spoof > 0 else "No spoof samples")
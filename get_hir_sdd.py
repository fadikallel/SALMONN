import json
import pandas as pd
import numpy as np
import ast
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
        "text": "<answer>{bonafide or spoof}</answer><explanation>{reasoning}</explanation><reasons>{reasons}</reasons>",
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
        print(f"📋 Columns: {df.columns.tolist()}")
        print(f"📊 Data types:\n{df.dtypes}")
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return []
    
    # Helper function to parse reasons
    def parse_reasons(reasons_val):
        """Parse reasons from various formats to a list of strings"""
        # Check for None
        if reasons_val is None:
            return []
        
        # Check for NaN using pandas, but handle numpy arrays carefully
        try:
            if isinstance(reasons_val, (np.ndarray, list)):
                # For arrays/lists, check if it's empty
                if len(reasons_val) == 0:
                    return []
                # If it's a numpy array with a single element that's NaN
                if isinstance(reasons_val, np.ndarray) and reasons_val.size == 1:
                    if pd.isna(reasons_val[0]):
                        return []
            elif pd.isna(reasons_val):
                return []
        except:
            pass
        
        # If it's already a list
        if isinstance(reasons_val, list):
            return reasons_val
        
        # If it's a numpy array
        if isinstance(reasons_val, np.ndarray):
            # Convert to list
            reasons_list = reasons_val.tolist()
            
            # If the list has one element that looks like a string representation of a list
            if len(reasons_list) == 1 and isinstance(reasons_list[0], str):
                try:
                    # Try to parse it as a list
                    parsed = ast.literal_eval(reasons_list[0])
                    if isinstance(parsed, list):
                        return parsed
                except:
                    # If it's a comma-separated string
                    if ',' in reasons_list[0]:
                        return [item.strip().strip('"\'') for item in reasons_list[0].split(',')]
                    return [reasons_list[0].strip('"\'')]
            else:
                # Convert each element to string
                return [str(item).strip('"\'') for item in reasons_list if item]
        
        # If it's a string, try to parse it as a list
        if isinstance(reasons_val, str):
            try:
                parsed = ast.literal_eval(reasons_val)
                if isinstance(parsed, list):
                    return parsed
            except:
                # If it's a comma-separated string
                if ',' in reasons_val:
                    return [item.strip().strip('"\'') for item in reasons_val.split(',')]
                return [reasons_val.strip('"\'')]
        
        # Try converting to string and parsing
        try:
            str_val = str(reasons_val)
            if str_val.startswith('[') and str_val.endswith(']'):
                parsed = ast.literal_eval(str_val)
                if isinstance(parsed, list):
                    return parsed
        except:
            pass
        
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
            audio_path = audio_path.lstrip('./').lstrip('../')
            absolute_path = base_audio_path + audio_path
            
            # Get bonafide status
            is_bonafide = row.get('is_bonafide', None)
            if is_bonafide is None:
                print(f"Warning: Row {idx} has no 'is_bonafide' field, skipping...")
                continue
            
            # Convert bonafide to answer text
            answer = "Real" if is_bonafide else "Fake"
            
            # Get reasoning
            reasoning = row.get('reasoning', '')
            if reasoning and not isinstance(reasoning, str):
                reasoning = str(reasoning)
            
            # Get and parse reasons
            reasons_raw = row.get('reasons', [])
            reasons_list = parse_reasons(reasons_raw)
            
            # Debug: print first few to see the format
            if idx < 5:
                print(f"\n🔍 Debug Row {idx}:")
                print(f"   Raw reasons type: {type(reasons_raw)}")
                print(f"   Raw reasons value: {reasons_raw}")
                print(f"   Parsed reasons: {reasons_list}")
            
            # Build the text with proper formatting
            if not reasons_list or len(reasons_list) == 0:
                formatted_text = f"<think>{reasoning}</think><answer>{answer}</answer>"
                # formatted_text = answer
            else:
                # Process each reason: lowercase and replace underscores with spaces
                processed_reasons = []
                for reason in reasons_list:
                    # Convert to string if needed
                    reason_str = str(reason)
                    # Remove any extra quotes
                    reason_str = reason_str.strip('"\'')
                    processed = reason_str.lower().replace('_', ' ')
                    processed_reasons.append(processed)
                
                # Join with commas
                reasons_str = ', '.join(processed_reasons)
                formatted_text = f"<think>{reasoning}</think><reasons>{reasons_str}</reasons><answer>{answer}</answer>"
                # formatted_text = answer
            
            # Create the training example
            example = {
                "path": absolute_path,
                "text": formatted_text,
                "task": task
            }
            
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
        bonafide_count = sum(1 for item in training_data if '<answer>Real' in item['text'])
        spoof_count = sum(1 for item in training_data if '<answer>Fake' in item['text'])
        print(f"\n📊 Statistics:")
        print(f"   Real samples: {bonafide_count}")
        print(f"   Fake samples: {spoof_count}")
        
        # Print first few paths as examples
        print(f"\n📁 Sample absolute paths:")
        for i in range(min(3, len(training_data))):
            print(f"   {training_data[i]['path']}")
            
        # Print a sample to verify formatting
        print(f"\n📝 Sample text formatting:")
        sample = training_data[0]
        print(f"   Path: {sample['path']}")
        print(f"   Text: {sample['text']}")
        print(f"   Task: {sample['task']}")
    
    return training_data

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_parquet = "/ds-slt/audio/fkallel/HIR-SDD/annotations/test_2.parquet"  # Change this to your parquet file
    output_file = "data/my_hir_sdd_test_2.json"

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
# import json
# from pathlib import Path

# def split_every_100_samples(input_json_path, output_dir="data_splits"):
#     """
#     Split the JSON data where each group has different sizes:
#     - Group 1: 100 samples (MLAAD real)
#     - Group 2: 96 samples (MLAAD fake)
#     - Group 3: 100 samples (ASVspoof real)
#     - Group 4: 97 samples (ASVspoof fake)
    
#     Each group is split into 80-10-10 train-val-test.
#     """
#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok=True)
    
#     # Load the JSON data
#     with open(input_json_path, 'r') as f:
#         all_samples = json.load(f)
    
#     print(f"📊 Loaded {len(all_samples)} samples")
#     print(f"Expected total: 100 + 96 + 100 + 97 = 393 samples\n")
    
#     # Define group sizes
#     group_sizes = [100, 96, 100, 97]
#     group_names = [
#         "MLAAD real",
#         "MLAAD fake", 
#         "ASVspoof real",
#         "ASVspoof fake"
#     ]
    
#     # Define splits
#     train_samples = []
#     val_samples = []
#     test_samples = []
    
#     # Process each group
#     start_idx = 0
#     for group_idx, group_size in enumerate(group_sizes):
#         end_idx = start_idx + group_size
#         group = all_samples[start_idx:end_idx]
        
#         # Calculate split sizes (80-10-10)
#         train_size = int(group_size * 0.8)  # 80% for training
#         val_size = int(group_size * 0.1)    # 10% for validation
#         test_size = group_size - train_size - val_size  # Remaining for test
        
#         # Split the group
#         train = group[:train_size]
#         val = group[train_size:train_size + val_size]
#         test = group[train_size + val_size:]
        
#         # Add to overall splits
#         train_samples.extend(train)
#         val_samples.extend(val)
#         test_samples.extend(test)
        
#         # Print group info
#         print(f"📦 Group {group_idx + 1}: {group_names[group_idx]} ({group_size} samples)")
#         print(f"  Train: {len(train)} samples ({train_size/group_size*100:.0f}%)")
#         print(f"  Val:   {len(val)} samples ({val_size/group_size*100:.0f}%)")
#         print(f"  Test:  {len(test)} samples ({test_size/group_size*100:.0f}%)")
        
#         # Verify we haven't lost any samples
#         if len(train) + len(val) + len(test) != group_size:
#             print(f"  ⚠️  WARNING: Split totals don't match group size!")
        
#         start_idx = end_idx
    
#     # Verify total
#     total_samples = len(train_samples) + len(val_samples) + len(test_samples)
#     print(f"\n📊 Total samples processed: {total_samples}")
    
#     # Save the three JSON files
#     print("\n💾 Saving splits...")
    
#     for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
#         output_file = output_dir / f"{split_name}.json"
#         with open(output_file, 'w') as f:
#             json.dump(samples, f, indent=2)
#         print(f"  ✅ Saved {len(samples)} samples to {split_name}.json")
    
#     # Print detailed statistics
#     print("\n" + "="*60)
#     print("📊 FINAL SPLIT STATISTICS")
#     print("="*60)
    
#     for split_name, samples in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
#         # Count labels
#         real_count = sum(1 for s in samples if '<answer>bona fide</answer>' in s.get('text', ''))
#         fake_count = sum(1 for s in samples if '<answer>spoof</answer>' in s.get('text', ''))
        
#         # Count sources
#         mlaad_count = sum(1 for s in samples if 'MLAAD' in s.get('path', ''))
#         asvspoof_count = sum(1 for s in samples if 'ASVspoof' in s.get('path', ''))
        
#         print(f"\n{split_name}:")
#         print(f"  Total:  {len(samples)}")
#         print(f"  Real:   {real_count} ({real_count/len(samples)*100:.1f}%)")
#         print(f"  Fake:   {fake_count} ({fake_count/len(samples)*100:.1f}%)")
#         print(f"  MLAAD:  {mlaad_count} ({mlaad_count/len(samples)*100:.1f}%)")
#         print(f"  ASVspoof: {asvspoof_count} ({asvspoof_count/len(samples)*100:.1f}%)")
    
#     return train_samples, val_samples, test_samples

# def verify_splits(train, val, test):
#     """Verify that the splits are correct"""
    
#     print("\n" + "="*60)
#     print("🔍 VERIFICATION")
#     print("="*60)
    
#     # Check for overlap (should be none)
#     train_ids = {s.get('id', id(s)) for s in train}
#     val_ids = {s.get('id', id(s)) for s in val}
#     test_ids = {s.get('id', id(s)) for s in test}
    
#     train_val_overlap = train_ids & val_ids
#     train_test_overlap = train_ids & test_ids
#     val_test_overlap = val_ids & test_ids
    
#     if train_val_overlap:
#         print(f"⚠️  Overlap between train and val: {len(train_val_overlap)} samples")
#     if train_test_overlap:
#         print(f"⚠️  Overlap between train and test: {len(train_test_overlap)} samples")
#     if val_test_overlap:
#         print(f"⚠️  Overlap between val and test: {len(val_test_overlap)} samples")
    
#     if not any([train_val_overlap, train_test_overlap, val_test_overlap]):
#         print("✅ No overlap between splits")
    
#     # Check expected sizes
#     expected_train = 80 + 77 + 80 + 78  # 80% of each group
#     expected_val = 10 + 10 + 10 + 10    # 10% of each group
#     expected_test = 10 + 9 + 10 + 9     # Remaining
    
#     print(f"\nExpected sizes:")
#     print(f"  Train: {expected_train} (80+77+80+78)")
#     print(f"  Val:   {expected_val} (10+10+10+10)")
#     print(f"  Test:  {expected_test} (10+9+10+9)")
#     print(f"  Total: {expected_train + expected_val + expected_test}")
    
#     print(f"\nActual sizes:")
#     print(f"  Train: {len(train)}")
#     print(f"  Val:   {len(val)}")
#     print(f"  Test:  {len(test)}")
#     print(f"  Total: {len(train) + len(val) + len(test)}")
    
#     # Check proportions
#     train_prop = len(train) / (len(train) + len(val) + len(test)) * 100
#     val_prop = len(val) / (len(train) + len(val) + len(test)) * 100
#     test_prop = len(test) / (len(train) + len(val) + len(test)) * 100
    
#     print(f"\nProportions:")
#     print(f"  Train: {train_prop:.1f}%")
#     print(f"  Val:   {val_prop:.1f}%")
#     print(f"  Test:  {test_prop:.1f}%")

# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual JSON file
#     input_file = "data/asv_mlaad_400.json"
    
#     # Split the data
#     train, val, test = split_every_100_samples(
#         input_json_path=input_file,
#         output_dir="data"
#     )
    
#     # Verify the splits
#     verify_splits(train, val, test)
    
#     # Show a sample
#     print("\n📝 Sample from train set:")
#     if train:
#         sample = train[0]
#         # Show key fields only for readability
#         sample_preview = {
#             'id': sample.get('id', 'N/A'),
#             'path': sample.get('path', 'N/A'),
#             'text': sample.get('text', '')[:100] + '...' if len(sample.get('text', '')) > 100 else sample.get('text', '')
#         }
#         print(json.dumps(sample_preview, indent=2))
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_json_data(input_json_path, output_dir, train_size=53580, val_size=5954, test_size=0):
    """
    Split JSON data into train, validation, and test sets
    
    Args:
        input_json_path: Path to the input JSON file
        output_dir: Directory to save the splits
        train_size: Number of training samples (default: 53580)
        val_size: Number of validation samples (default: 5954)
        test_size: Number of test samples (default: 0)
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the JSON data
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} samples from {input_json_path}")
    
    total_samples = len(data)
    total_requested = train_size + val_size + test_size
    
    # Adjust sizes if not enough samples
    if total_samples < total_requested:
        print(f"⚠️  Warning: Only {total_samples} samples available, which is less than requested total of {total_requested}")
        print(f"   Adjusting split sizes proportionally...")
        
        train_size = int(total_samples * (train_size / total_requested))
        val_size = int(total_samples * (val_size / total_requested))
        test_size = total_samples - train_size - val_size
        
        print(f"   New sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split the data
    # First split: separate test set
    # train_val_data, test_data = train_test_split(
    #     data, 
    #     test_size=test_size, 
    #     random_state=42,
    #     shuffle=True
    # )
    train_val_data = data  # Since test_size is 0, we keep all data for train/val
    
    # Second split: separate train and validation
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=val_size, 
        random_state=42,
        shuffle=True
    )
    
    # Save the splits
    splits = {
        'train': train_data,
        'val': val_data,
        # 'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_file = Path('data') / f"my_hir_sdd_binary_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Count bonafide vs spoof in each split
        bonafide_count = sum(1 for item in split_data if 'bonafide' in item['text'])
        spoof_count = sum(1 for item in split_data if 'spoof' in item['text'])
        
        print(f"\n📁 {split_name.upper()} split:")
        print(f"   Samples: {len(split_data)}")
        print(f"   Bonafide: {bonafide_count}")
        print(f"   Spoof: {spoof_count}")
        print(f"   Saved to: {output_file}")
    
    # Print sample from train set
    if train_data:
        print("\n📝 Sample from TRAIN set:")
        print(json.dumps(train_data[0], indent=2))
    
    print(f"\n✅ All splits saved to {output_dir}/")
    
    return splits

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_json = "data/my_hir_sdd_binary.json"  # Your prepared JSON file
    output_directory = "data/"  # Directory where splits will be saved
    
    # Split the data
    splits = split_json_data(
        input_json_path=input_json,
        output_dir=output_directory,
        train_size=53580,
        val_size=5954,
        test_size=0
    )
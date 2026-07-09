#!/usr/bin/env python3
"""
Split the dataset by source corpus (no corpus leakage) and verify
reason coverage.

Groups:
    TRAIN
        - asvspoof5
        - xtts
        - Espeech_spoofs
        - golos
        - sova
        - ruLS
        - SpeechLLM

    TEST
        - MLAAD
        - M-AILABS
        - LibriSeVoc
        - dfadd
        - final_dataset

Outputs:
    train.parquet
    test.parquet

Also prints:
    - exact number of samples
    - exact number of unique audio clips
    - corpus distribution
    - reason distribution
    - missing reasons (if any)
"""

import ast
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_PARQUET = "/ds-slt/audio/fkallel/HIR-SDD/annotations/data.parquet"

TRAIN_DATASETS = {
    "asvspoof5",
    "xtts",
    "Espeech_spoofs",
    "golos",
    "sova",
    "ruLS",
    "SpeechLLM",
    "LibriSeVoc",
    "dfadd",

}

TEST_DATASETS = {
    "MLAAD",
    "M-AILABS",
    "final_dataset",
}


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

df = pd.read_parquet(INPUT_PARQUET)

print("=" * 80)
print("TOTAL DATASET")
print("=" * 80)
print(f"Rows: {len(df):,}")

if "audio" in df.columns:
    print(f"Unique audio clips: {df['audio'].nunique():,}")

print()
source_col = "source_corpus"


# ---------------------------------------------------------------------
# Check datasets
# ---------------------------------------------------------------------

available = set(df[source_col].unique())

print("\nDatasets found:")
for d in sorted(available):
    print(" ", d)

unknown = available - TRAIN_DATASETS - TEST_DATASETS

if len(unknown):
    print("\nWARNING: Unknown datasets:")
    print(unknown)

missing = (TRAIN_DATASETS | TEST_DATASETS) - available

if len(missing):
    print("\nWARNING: Expected datasets not found:")
    print(missing)


# ---------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------

train = df[df[source_col].isin(TRAIN_DATASETS)].copy()
test = df[df[source_col].isin(TEST_DATASETS)].copy()

print("\n" + "=" * 80)
print("SPLIT")
print("=" * 80)

print(f"Train rows : {len(train):,}")
print(f"Test rows  : {len(test):,}")
print(f"Total      : {len(train)+len(test):,}")

print()

print(f"Train % : {100*len(train)/len(df):.2f}%")
print(f"Test %  : {100*len(test)/len(df):.2f}%")

if "audio" in df.columns:
    print()
    print(f"Train unique audio: {train['audio'].nunique():,}")
    print(f"Test unique audio : {test['audio'].nunique():,}")


# ---------------------------------------------------------------------
# Corpus counts
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAIN CORPUS COUNTS")
print("=" * 80)
print(train[source_col].value_counts())

print("\n" + "=" * 80)
print("TEST CORPUS COUNTS")
print("=" * 80)
print(test[source_col].value_counts())
reason_col = "reasons"
print("Reason column:", reason_col)
print("Type:", type(df[reason_col].iloc[0]))
print("Value:")
print(repr(df[reason_col].iloc[0]))
print()

for i in range(5):
    print("="*80)
    print(type(df[reason_col].iloc[i]))
    print(repr(df[reason_col].iloc[i]))
import numpy as np
# from collections.abc import Iterable
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


train_reasons = train["reasons"].apply(parse_reasons).explode().dropna()
test_reasons = test["reasons"].apply(parse_reasons).explode().dropna()

train_counts = train_reasons.value_counts().sort_index()
test_counts = test_reasons.value_counts().sort_index()
all_reasons = sorted(
    set(train_counts.index) | set(test_counts.index)
)

print("\n" + "=" * 80)
print("REASON DISTRIBUTION")
print("=" * 80)

summary = pd.DataFrame(
    {
        "train": train_counts,
        "test": test_counts,
    }
).fillna(0).astype(int)

print(summary)

missing_train = set(all_reasons) - set(train_counts.index)
missing_test = set(all_reasons) - set(test_counts.index)

print("\n" + "=" * 80)
print("REASON COVERAGE")
print("=" * 80)

if len(missing_train) == 0:
    print("✓ Every reason exists in TRAIN")
else:
    print("✗ Missing in TRAIN:")
    for r in sorted(missing_train):
        print("   ", r)

if len(missing_test) == 0:
    print("✓ Every reason exists in TEST")
else:
    print("✗ Missing in TEST:")
    for r in sorted(missing_test):
        print("   ", r)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

train.to_parquet("/ds-slt/audio/fkallel/HIR-SDD/annotations/train.parquet", index=False)
test.to_parquet("/ds-slt/audio/fkallel/HIR-SDD/annotations/test.parquet", index=False)

print("\nSaved:")
print("  train.parquet")
print("  test.parquet")
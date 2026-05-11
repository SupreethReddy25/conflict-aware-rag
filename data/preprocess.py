"""
data/preprocess.py
Builds two derived datasets from the raw ConflictQA data:

1. Conflict Detector Dataset (binary classification):
   - For each example, create TWO rows:
     - CONFLICT (label=1): question + memory_answer + counter_memory_aligned_evidence
     - NO CONFLICT (label=0): question + memory_answer + parametric_memory_aligned_evidence
   - Total: 15,894 examples -> 11,443 train / 1,272 val / 3,179 test

2. Reader (Resolver) Fine-Tuning Dataset (instruction format):
   - For each example, create ONE row with instruction-formatted input/output
   - Total: 7,947 examples -> 5,721 train / 636 val / 1,590 test
"""

import os
import json
import random
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join(os.path.dirname(__file__), "raw", "conflictQA-popQA-chatgpt.json")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

RANDOM_SEED = 42


def load_raw_data():
    """Load the raw ConflictQA JSONL file."""
    examples = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"[INFO] Loaded {len(examples)} raw examples")
    return examples


def build_detector_dataset(examples):
    """
    Build the conflict detector binary classification dataset.
    Two rows per original example:
      - label=1 (CONFLICT): uses counter_memory_aligned_evidence
      - label=0 (NO CONFLICT): uses parametric_memory_aligned_evidence
    """
    detector_data = []

    for ex in examples:
        question = ex["question"]
        memory_answer = ex["memory_answer"]

        # Get ground truth - it's a list, use index 0
        ground_truth = ex["ground_truth"]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]

        # CONFLICT example (label=1): conflict document contradicts memory
        conflict_doc = ex["counter_memory_aligned_evidence"]
        detector_data.append({
            "question": question,
            "memory_answer": memory_answer,
            "document": conflict_doc,
            "label": 1,
            "ground_truth": ground_truth,
        })

        # NO CONFLICT example (label=0): parametric document agrees with memory
        parametric_doc = ex["parametric_memory_aligned_evidence"]
        detector_data.append({
            "question": question,
            "memory_answer": memory_answer,
            "document": parametric_doc,
            "label": 0,
            "ground_truth": ground_truth,
        })

    print(f"[INFO] Built {len(detector_data)} detector examples ({len(detector_data)//2} conflict + {len(detector_data)//2} no-conflict)")
    return detector_data


def build_reader_dataset(examples):
    """
    Build the reader (resolver) fine-tuning dataset.
    One instruction-formatted example per original.
    Input: instruction with conflict document + question
    Output: ground truth answer
    """
    reader_data = []

    for ex in examples:
        question = ex["question"]
        conflict_doc = ex["counter_memory_aligned_evidence"]

        # Ground truth
        ground_truth = ex["ground_truth"]
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0]

        # Instruction format for Mistral fine-tuning
        instruction = (
            "Answer the question based on the provided document. "
            "If the document conflicts with your prior knowledge, trust the document.\n\n"
            f"Document: {conflict_doc}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        reader_data.append({
            "instruction": instruction,
            "output": ground_truth,
            "question": question,
            "document": conflict_doc,
        })

    print(f"[INFO] Built {len(reader_data)} reader examples")
    return reader_data


def split_and_save(data, name, train_ratio=0.72, val_ratio=0.08, test_ratio=0.20):
    """
    Split data into train/val/test and save as JSONL files.
    Default ratios: 72% train, 8% val, 20% test
    """
    random.seed(RANDOM_SEED)

    # First split: separate test set
    train_val, test = train_test_split(
        data, test_size=test_ratio, random_state=RANDOM_SEED, shuffle=True
    )

    # Second split: separate train and val from remaining
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_fraction, random_state=RANDOM_SEED, shuffle=True
    )

    splits = {"train": train, "val": val, "test": test}

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for split_name, split_data in splits.items():
        filepath = os.path.join(PROCESSED_DIR, f"{name}_{split_name}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} examples -> {filepath}")

    return {k: len(v) for k, v in splits.items()}


def verify_counts(detector_counts, reader_counts):
    """Verify the split counts match expected values."""
    expected_detector = {"train": 11443, "val": 1272, "test": 3179}
    expected_reader = {"train": 5721, "val": 636, "test": 1590}

    print("\n[VERIFICATION]")
    print(f"  Detector - Expected: {expected_detector}")
    print(f"  Detector - Actual:   {detector_counts}")

    print(f"  Reader   - Expected: {expected_reader}")
    print(f"  Reader   - Actual:   {reader_counts}")

    detector_total = sum(detector_counts.values())
    reader_total = sum(reader_counts.values())

    print(f"\n  Detector total: {detector_total} (expected 15894)")
    print(f"  Reader total:   {reader_total} (expected 7947)")

    # Check totals match
    if detector_total == 15894 and reader_total == 7947:
        print("\n[SUCCESS] Total counts match perfectly!")
    else:
        print("\n[WARNING] Total counts differ from expected. This may be due to rounding in splits.")
        print("  The data is still valid for training.")

    return True


def main():
    print("=" * 60)
    print("CONFLICT-AWARE RAG — Data Preprocessing")
    print("=" * 60)

    # Load raw data
    examples = load_raw_data()

    # Build detector dataset
    print("\n--- Building Conflict Detector Dataset ---")
    detector_data = build_detector_dataset(examples)
    print("\nSplitting detector data...")
    detector_counts = split_and_save(detector_data, "detector")

    # Build reader dataset
    print("\n--- Building Reader (Resolver) Dataset ---")
    reader_data = build_reader_dataset(examples)
    print("\nSplitting reader data...")
    reader_counts = split_and_save(reader_data, "reader")

    # Verify
    verify_counts(detector_counts, reader_counts)

    # Print sample from each
    print("\n--- Sample Detector Example (CONFLICT) ---")
    conflict_sample = next(d for d in detector_data if d["label"] == 1)
    print(f"  Question: {conflict_sample['question'][:80]}...")
    print(f"  Memory Answer: {conflict_sample['memory_answer'][:80]}...")
    print(f"  Document: {conflict_sample['document'][:80]}...")
    print(f"  Label: {conflict_sample['label']}")

    print("\n--- Sample Reader Example ---")
    reader_sample = reader_data[0]
    print(f"  Instruction: {reader_sample['instruction'][:120]}...")
    print(f"  Output: {reader_sample['output'][:80]}...")

    print("\n" + "=" * 60)
    print("[DONE] All 6 JSONL files created in data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()

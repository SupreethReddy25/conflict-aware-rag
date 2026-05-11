"""
data/load_data.py
Downloads the ConflictQA dataset from HuggingFace using hf_hub_download.
CRITICAL: load_dataset() does NOT work for this dataset. Must use hf_hub_download().
"""

import os
import json
from huggingface_hub import hf_hub_download

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_conflictqa():
    """Download conflictQA-popQA-chatgpt.json from osunlp/ConflictQA."""
    os.makedirs(RAW_DIR, exist_ok=True)

    output_path = os.path.join(RAW_DIR, "conflictQA-popQA-chatgpt.json")

    if os.path.exists(output_path):
        print(f"[INFO] Raw data already exists at {output_path}")
        # Verify it's valid
        with open(output_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        print(f"[INFO] Contains {count} examples")
        return output_path

    print("[INFO] Downloading conflictQA-popQA-chatgpt.json from osunlp/ConflictQA...")
    downloaded_path = hf_hub_download(
        repo_id="osunlp/ConflictQA",
        filename="conflictQA-popQA-chatgpt.json",
        repo_type="dataset",
        local_dir=RAW_DIR,
    )

    # Verify the download
    with open(output_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    print(f"[SUCCESS] Downloaded {count} examples to {output_path}")

    return output_path


if __name__ == "__main__":
    path = download_conflictqa()
    print(f"\n[DONE] Raw data ready at: {path}")

    # Print a sample to verify structure
    with open(path, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline().strip())

    print("\n[SAMPLE] First example fields:")
    for key in sample:
        val = str(sample[key])[:100]
        print(f"  {key}: {val}...")

# sample_subset.py

import os
import random
import shutil
import argparse
from pathlib import Path

def sample_and_copy(src_dir: Path, dst_dir: Path, num_samples: int):
    """
    For each class subfolder in src_dir, randomly sample up to num_samples files
    and copy them into the corresponding subfolder in dst_dir.
    """
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue

        dst_class_dir = dst_dir / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        # list all files (you can adjust the extension filter if needed)
        all_files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        if not all_files:
            print(f"⚠️  No image files found in {class_dir}")
            continue

        # sample
        if len(all_files) <= num_samples:
            chosen = all_files
            print(f"ℹ️  Only {len(all_files)} available in {class_dir.name}, copying all.")
        else:
            chosen = random.sample(all_files, num_samples)

        # copy
        for src_path in chosen:
            dst_path = dst_class_dir / src_path.name
            shutil.copy2(src_path, dst_path)

def main():
    parser = argparse.ArgumentParser(description="Create smaller train/val subsets by sampling per class.")
    parser.add_argument('--data_dir',   type=Path, required=True,
                        help="Root data dir with 'train/' and 'val/' subfolders.")
    parser.add_argument('--output_dir', type=Path, required=True,
                        help="Where to write 'train/' and 'val/' subsets.")
    parser.add_argument('--train_samples_per_class', type=int, default=500,
                        help="How many samples per class for training.")
    parser.add_argument('--val_samples_per_class',   type=int, default=50,
                        help="How many samples per class for validation.")
    args = parser.parse_args()

    random.seed(42)  # for reproducibility

    train_src = args.data_dir / 'train'
    val_src   = args.data_dir / 'val'
    train_dst = args.output_dir / 'train'
    val_dst   = args.output_dir / 'val'

    # make sure source dirs exist
    if not train_src.is_dir() or not val_src.is_dir():
        parser.error(f"Both {train_src} and {val_src} must exist and be directories.")

    # sample & copy
    print("Sampling training data...")
    sample_and_copy(train_src, train_dst, args.train_samples_per_class)

    print("Sampling validation data...")
    sample_and_copy(val_src, val_dst, args.val_samples_per_class)

    print(f"\nDone! Subsets written to {args.output_dir}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to clean training log files by keeping only the last line of each tqdm block.
Usage: python clean_training_log.py <input_log> [output_log]
"""

import re
import sys
from pathlib import Path


def clean_log_file(input_path: str, output_path: str = None):
    """Clean a training log by removing intermediate tqdm updates."""
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"Error: File {input_path} does not exist")
        sys.exit(1)

    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"

    print(f"Reading: {input_file}")
    print(f"Writing: {output_path}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Group train/val lines by epoch
    train_blocks = {}  # epoch -> list of lines
    val_blocks = {}    # epoch -> list of lines
    other_lines = []   # non-tqdm lines

    train_pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+\[Train\]:')
    val_pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+\[Val\]:')
    summary_pattern = re.compile(r'Epoch\s+\[\s*(\d+)/\d+\]')

    for line in lines:
        if match := train_pattern.search(line):
            epoch = int(match.group(1))
            train_blocks.setdefault(epoch, []).append(line)
        elif match := val_pattern.search(line):
            epoch = int(match.group(1))
            val_blocks.setdefault(epoch, []).append(line)
        else:
            other_lines.append(line)

    # Reconstruct: for each summary line, insert last train/val before it
    result = []
    for line in other_lines:
        if match := summary_pattern.search(line):
            epoch = int(match.group(1))
            if epoch in train_blocks:
                result.append(train_blocks[epoch][-1])
            if epoch in val_blocks:
                result.append(val_blocks[epoch][-1])
        result.append(line)

    with open(output_path, 'w') as f:
        f.writelines(result)

    print(f"\nOriginal: {len(lines):,} lines → Cleaned: {len(result):,} lines")
    print(f"Removed: {len(lines) - len(result):,} lines ({100*(len(lines)-len(result))/len(lines):.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_training_log.py <input_log> [output_log]")
        sys.exit(1)

    clean_log_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

#!/usr/bin/env python3
"""
Split a JSONL dataset into train/val/test subsets with configurable ratios.
Outputs files named <base>-train.jsonl, <base>-val.jsonl, <base>-test.jsonl
in the same directory (or an optional output directory).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Split a JSONL dataset into train/val/test subsets.")
	parser.add_argument(
		"--input",
		type=Path,
		required=True,
		help="Path to the input JSONL dataset.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory to write splits (defaults to input file directory).",
	)
	parser.add_argument(
		"--ratios",
		type=float,
		nargs=3,
		default=(0.6, 0.2, 0.2),
		metavar=("TRAIN", "VAL", "TEST"),
		help="Split ratios for train/val/test (must sum to 1.0).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for shuffling before splitting.",
	)
	return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
	if not path.exists():
		raise FileNotFoundError(f"Input file not found: {path}")
	records: List[dict] = []
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	if not records:
		raise ValueError(f"No records found in {path}")
	return records


def write_jsonl(path: Path, records: List[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for rec in records:
			handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
	args = parse_args()
	train_ratio, val_ratio, test_ratio = args.ratios
	total = train_ratio + val_ratio + test_ratio
	if abs(total - 1.0) > 1e-6:
		raise ValueError(f"Ratios must sum to 1.0; got {total:.6f}")

	records = load_jsonl(args.input)
	random.Random(args.seed).shuffle(records)

	num_total = len(records)
	train_end = int(num_total * train_ratio)
	val_end = train_end + int(num_total * val_ratio)

	train_records = records[:train_end]
	val_records = records[train_end:val_end]
	test_records = records[val_end:]

	output_dir = args.output_dir or args.input.parent
	base_name = args.input.stem
	if base_name.endswith(".json"):
		base_name = base_name[:-5]

	write_jsonl(output_dir / f"{base_name}-train.jsonl", train_records)
	write_jsonl(output_dir / f"{base_name}-val.jsonl", val_records)
	write_jsonl(output_dir / f"{base_name}-test.jsonl", test_records)

	print(f"Wrote {len(train_records)} train records to {(output_dir / f'{base_name}-train.jsonl')}")
	print(f"Wrote {len(val_records)} validation records to {(output_dir / f'{base_name}-val.jsonl')}")
	print(f"Wrote {len(test_records)} test records to {(output_dir / f'{base_name}-test.jsonl')}")


if __name__ == "__main__":
	main()

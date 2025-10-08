#!/usr/bin/env python3
"""Batch runner for test.py evaluations.

This script iterates over all pickle files in a results directory and
invokes ``test.py`` for each file with the provided judge. The outputs are
saved under ``final_result`` (configurable) with the ``eval_`` prefix.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test.py on every pickle file inside a results directory."
    )
    parser.add_argument(
        "--judge",
        required=True,
        help="Judge identifier passed through to test.py (e.g. gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/results",
        type=Path,
        help="Directory that contains *.pkl result files to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="final_result",
        type=Path,
        help="Directory where evaluation files will be written.",
    )
    parser.add_argument(
        "--test-script",
        default=Path("test.py"),
        type=Path,
        help="Path to test.py (or compatible evaluation script).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run evaluations even if the target output file already exists.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to test.py after a '--'.",
    )
    return parser.parse_args()


def list_result_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory is not a folder: {results_dir}")
    return sorted(path for path in results_dir.glob("*.pkl") if path.is_file())


def build_command(
    python_executable: str,
    test_script: Path,
    data_file: Path,
    judge: str,
    output_file: Path,
    extra_args: list[str] | None,
) -> list[str]:
    cmd = [
        python_executable,
        str(test_script),
        "-f",
        str(data_file),
        "--judge",
        judge,
        "--output",
        str(output_file),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_command(cmd: list[str]) -> None:
    display_cmd = shlex.join(cmd) if hasattr(shlex, "join") else " ".join(cmd)
    print(f"[INFO] Running: {display_cmd}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation command failed with exit code {completed.returncode}: {display_cmd}"
        )


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    test_script = args.test_script.resolve()
    extra_args = args.extra_args

    if extra_args:
        # argparse.REMAINDER retains the leading '--'; strip if present.
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
    else:
        extra_args = []

    if not test_script.exists():
        raise FileNotFoundError(f"test.py not found at: {test_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = list_result_files(results_dir)
    if not result_files:
        print(f"[WARN] No *.pkl files found in {results_dir}")
        return

    for data_file in result_files:
        output_file = output_dir / f"eval_{data_file.name}"
        if output_file.exists() and not args.overwrite:
            print(f"[SKIP] {output_file} already exists (use --overwrite to redo).")
            continue

        cmd = build_command(
            python_executable=sys.executable,
            test_script=test_script,
            data_file=data_file,
            judge=args.judge,
            output_file=output_file,
            extra_args=extra_args,
        )
        run_command(cmd)

    print("[DONE] Completed batch evaluation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit confidence calibration from JSONL: {"quality_score": ..., "correct": 0|1}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.confidence_calibration import DEFAULT_MODEL, fit_isotonic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = fit_isotonic(
        [record["quality_score"] for record in records],
        [record["correct"] for record in records],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(model, ensure_ascii=False))


if __name__ == "__main__":
    main()

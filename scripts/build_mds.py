                      


from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.causal.mds import build_mds_label

from scripts.cli import require_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_path = require_file(
        args.input,
        "Rollout data not found.\nPlease provide paired rollout JSONL with --input.",
    )
    with input_path.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                label = build_mds_label(row["original"], row["counterfactual"])
            except KeyError as exc:
                raise ValueError(f"line {line_number} must contain original and counterfactual") from exc
            target.write(json.dumps({**row, **label.as_dict()}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

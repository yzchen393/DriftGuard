                      


from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.evaluation.detection import evaluate_detection

from scripts.cli import require_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    scores, labels = [], []
    input_path = require_file(
        args.input,
        "Detection input not found.\nPlease provide evaluation JSONL with --input.",
    )
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scores.append(float(row["risk"]))
            labels.append(int(row.get("label", any(value == 1 for value in row.get("labels", {}).values()))))
    text = json.dumps(evaluate_detection(scores, labels), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

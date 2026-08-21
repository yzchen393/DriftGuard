                      


from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.evaluation.defense import summarize_defense

from scripts.cli import require_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    input_path = require_file(
        args.input,
        "Defense evaluation input not found.\nPlease provide evaluation JSONL with --input.",
    )
    with input_path.open(encoding="utf-8") as handle:
        result = summarize_defense(json.loads(line) for line in handle if line.strip())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

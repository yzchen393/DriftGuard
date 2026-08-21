from __future__ import annotations

import argparse

from driftguard.benchmarks.common import benchmark_data_path
from scripts.cli import fail, require_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = require_file(
        args.config,
        "Benchmark config not found.\nPlease provide a config such as configs/benchmark/minja.yaml.",
    )
    try:
        path = benchmark_data_path(config_path)
    except ValueError:
        fail(f"Benchmark data not configured.\nPlease set dataset.path in {args.config}.")
    except FileNotFoundError:
        fail(f"Benchmark data not found.\nPlease set dataset.path in {args.config}.")
    print(path)


if __name__ == "__main__":
    main()

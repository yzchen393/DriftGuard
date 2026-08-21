# DriftGuard

Official implementation of "DriftGuard: Online Memory-Drift Detection via Causal Attribution for LLM Agents".

## How to Use

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build MDS labels:

```bash
python -m scripts.build_mds \
  --input path/to/paired_rollouts.jsonl \
  --output path/to/mds_labels.jsonl
```

Train the ORM:

```bash
python -m scripts.train_orm --config configs/orm/minja.yaml
```

Evaluate detection:

```bash
python -m scripts.eval_detection \
  --input path/to/detection.jsonl \
  --output path/to/detection_metrics.json
```

Evaluate online defense:

```bash
python -m scripts.eval_defense --input path/to/defense.jsonl
```

Run the demo:

```bash
python -m scripts.run_demo
```

## External Dependencies

Full experiments require external benchmark and agent-environment resources:

- [MINJA](https://github.com/dsh3n77/MINJA)
- [AgentPoison](https://github.com/AI-secure/AgentPoison)
- [MPBench](https://github.com/Digital-Trust-Lab/mp-bench)
- [Mem2ActBench](https://github.com/Cantaloupe-M/Mem2ActBench)
- [STATE-Bench](https://github.com/microsoft/STATE-Bench)
- [DBBench](https://github.com/solomoon313/AgentMemoryBench)

Prepare the required resources locally and set their paths in the corresponding configuration files under configs/.

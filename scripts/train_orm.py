from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftguard.evaluation.metrics import auprc
from driftguard.orm.dataset import PreActionDataset, format_preaction_input
from driftguard.orm.model import RISK_HEADS, RiskModel, RiskModelConfig
from driftguard.orm.trainer import masked_bce_loss
from driftguard.utils.config import load_yaml
from driftguard.utils.seed import seed_everything

from scripts.cli import fail, require_file


def _load_training_data(config: dict, config_name: str) -> tuple[PreActionDataset, PreActionDataset]:
    train_value = config.get("train_jsonl")
    validation_value = config.get("validation_jsonl")
    if not train_value or not validation_value:
        fail(
            "Training data not configured.\n"
            f"Please set train_jsonl and validation_jsonl in {config_name}."
        )
    train_path = Path(train_value)
    validation_path = Path(validation_value)
    if not train_path.is_file() or not validation_path.is_file():
        fail(
            "Training data not found.\n"
            f"Please set train_jsonl and validation_jsonl in {config_name}."
        )
    train = PreActionDataset.from_jsonl(train_path)
    validation = PreActionDataset.from_jsonl(validation_path)
    if not train or not validation:
        fail(
            "Training data is empty.\n"
            f"Please provide non-empty JSONL files in {config_name}."
        )
    return train, validation


def _validation_metrics(model, loader, torch):
    validation_loss = 0.0
    risks = []
    labels = []
    with torch.no_grad():
        for encoded, batch_labels, masks in loader:
            output = model(**encoded)
            validation_loss += float(masked_bce_loss(output["logits"], batch_labels, masks).detach().cpu())
            batch_risks = output["risk"].detach().cpu().tolist()
            risks.extend(batch_risks)
            for index in range(len(batch_risks)):
                labels.append(
                    int(
                        any(
                            bool(masks[head][index]) and bool(batch_labels[head][index])
                            for head in RISK_HEADS
                        )
                    )
                )
    validation_loss /= max(len(loader), 1)
    try:
        validation_auprc = auprc(risks, labels)
    except ValueError:
        validation_auprc = 0.0
    return validation_loss, validation_auprc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = require_file(
        args.config,
        "ORM config not found.\nPlease provide a config such as configs/orm/minja.yaml.",
    )
    config = load_yaml(config_path)
    train, validation = _load_training_data(config, args.config)
    seed_everything(int(config.get("seed", 5316)))
    summary = {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "model": config.get("model_name_or_path", ""),
    }
    print(summary)
    if args.dry_run:
        print(format_preaction_input(train[0])[:500])
        return
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer
    except ImportError:
        fail("ORM dependencies not installed.\nPlease install requirements.txt.")
    model_name = config.get("model_name_or_path")
    if not model_name:
        fail(
            "Model checkpoint not configured.\n"
            f"Please set model_name_or_path in {args.config}."
        )
    if str(config.get("optimizer", "AdamW")).lower() != "adamw":
        fail(f"The paper configuration requires AdamW. Please set optimizer: AdamW in {args.config}.")
    if str(config.get("scheduler", "cosine")).lower() != "cosine":
        fail(f"The paper configuration requires cosine scheduling. Please set scheduler: cosine in {args.config}.")
    if str(config.get("early_stopping_metric", "validation_auprc")).lower() != "validation_auprc":
        fail(f"The paper configuration requires validation AUPRC early stopping in {args.config}.")
    local_files_only = bool(config.get("local_files_only", True))
    model_config = RiskModelConfig(
        model_name_or_path=model_name,
        max_length=int(config.get("max_length", 2048)),
        lora_rank=int(config.get("lora_rank", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        local_files_only=local_files_only,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = RiskModel(model_config)
    except OSError:
        fail(
            "Model checkpoint not found.\n"
            f"Please set model_name_or_path in {args.config} or make the checkpoint available locally."
        )

    def collate(rows):
        encoded = tokenizer(
            [format_preaction_input(row) for row in rows],
            padding=True,
            truncation=True,
            max_length=model_config.max_length,
            return_tensors="pt",
        )
        labels = {
            head: torch.tensor([float((row.labels or {}).get(head) or 0) for row in rows])
            for head in RISK_HEADS
        }
        masks = {
            head: torch.tensor(
                [bool((row.masks or {}).get(head, (row.labels or {}).get(head) is not None)) for row in rows]
            )
            for head in RISK_HEADS
        }
        return encoded, labels, masks

    batch_size = int(config.get("batch_size", config.get("per_device_train_batch_size", 16)))
    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    train_loader = DataLoader(train.examples, batch_size=batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation.examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = min(3, int(config.get("num_train_epochs", 3)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    output_dir = Path(config.get("output_dir", "checkpoints/orm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = float("-inf")
    stale_epochs = 0
    early_stopping_patience = max(1, int(config.get("early_stopping_patience", 1)))
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (encoded, labels, masks) in enumerate(train_loader):
            loss = masked_bce_loss(model(**encoded)["logits"], labels, masks) / accumulation
            loss.backward()
            running += float(loss.detach().cpu()) * accumulation
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        validation_loss, validation_auprc = _validation_metrics(model, validation_loader, torch)
        scheduler.step()
        print(
            {
                "epoch": epoch + 1,
                "train_loss": running / max(len(train_loader), 1),
                "validation_loss": validation_loss,
                "validation_auprc": validation_auprc,
            }
        )
        if validation_auprc > best_validation:
            best_validation = validation_auprc
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_risk_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
    torch.save(model.state_dict(), output_dir / "risk_model.pt")
    tokenizer.save_pretrained(output_dir)
    summary["best_validation_auprc"] = best_validation
    summary["optimizer"] = "AdamW"
    summary["scheduler"] = "cosine"
    summary["batch_size"] = batch_size * accumulation
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

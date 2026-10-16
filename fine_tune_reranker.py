"""Fine-tune SciFact claim-document relevance with an explicit PyTorch loop.

Inputs: data/derived/reranker_train.jsonl and reranker_val.jsonl
Outputs: checkpoints/scifact_reranker/ (best checkpoint + training_summary.json)

Run:
    python fine_tune_reranker.py
    python fine_tune_reranker.py --epochs 3 --batch-size 8 --max-length 512

Requires: pip install torch transformers sentence-transformers

Labels: 1 = annotated SUPPORT/CONTRADICT evidence; 0 = *presumed*
negative, not independently verified irrelevant. Only training-split claims
are used for weight updates; the 101 held-out training claims are used for
checkpoint selection. SciFact dev/test claims are not read by this script.
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
PRETRAINED_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_pairs(path: Path) -> list[dict]:
    """Read and validate the output of prepare_training_data.py."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"claim_id", "doc_id", "claim", "document", "label"}
            if not required.issubset(record):
                raise ValueError(f"{path}:{lineno}: missing {required - record.keys()}")
            if record["label"] not in (0, 1):
                raise ValueError(f"{path}:{lineno}: expected binary labels")
            if not isinstance(record["claim"], str) or not isinstance(record["document"], str):
                raise ValueError(f"{path}:{lineno}: claim/document must be strings")
            records.append(record)
    if not records:
        raise ValueError(f"No pairs found: {path}")
    keys = [(p["claim_id"], p["doc_id"]) for p in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate claim-document pairs: {path}")
    return records


class EvidencePairs(Dataset):
    """A single example is a claim/document pair with a binary relevance label."""

    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        return self.pairs[i]


def make_collate(tokenizer, max_length: int):
    """Tokenize and dynamically pad claim/document pairs within each batch."""
    def collate(batch: list[dict]) -> dict:
        features = tokenizer(
            [p["claim"] for p in batch],
            [p["document"] for p in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        features["labels"] = torch.tensor([p["label"] for p in batch], dtype=torch.float32)
        return features
    return collate


def preferred_device(choice: str) -> torch.device:
    if choice != "auto":
        device = torch.device(choice)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this Mac")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validation_ranking(pairs: list[dict], scores: list[float]) -> dict:
    """MRR@5 and top-1 hit rate over the sampled validation candidate pools.

    This is *not* Recall@k over the 5,183-document corpus or the dev benchmark.
    """
    if len(pairs) != len(scores):
        raise ValueError("One prediction is required for each validation pair")
    by_claim = defaultdict(list)
    for p, score in zip(pairs, scores):
        by_claim[p["claim_id"]].append((float(score), int(p["doc_id"]), p["label"]))
    reciprocal_ranks = []
    hits_at_1 = []
    for candidates in by_claim.values():
        ordered = sorted(candidates, key=lambda x: (-x[0], x[1]))
        first_positive = next(
            (i for i, (_, _, label) in enumerate(ordered, 1) if label == 1), None
        )
        reciprocal_ranks.append(1.0 / first_positive if first_positive and first_positive <= 5 else 0.0)
        hits_at_1.append(float(first_positive == 1))
    return {
        "sampled_val_mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "sampled_val_hit_at_1": sum(hits_at_1) / len(hits_at_1),
        "validation_claims": len(by_claim),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, pairs: list[dict], device: torch.device) -> dict:
    model.eval()
    loss_sum = 0.0
    all_scores = []
    correct = 0
    total = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {name: tensor.to(device) for name, tensor in batch.items()}
        logits = model(**inputs).logits.reshape(-1)
        loss_sum += nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        ).item()
        scores = logits.float().cpu().tolist()
        all_scores.extend(scores)
        correct += int(((logits >= 0).float() == labels).sum().item())
        total += len(labels)
    if total != len(pairs):
        raise RuntimeError("Validation DataLoader order/length doesn't match supplied pairs")
    return {
        "val_bce_loss": loss_sum / total,
        "val_binary_accuracy": correct / total,
        **validation_ranking(pairs, all_scores),
    }


def train_one_epoch(model, loader, optimizer, device, pos_weight: float) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    positive_weight = torch.tensor(pos_weight, device=device)
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {name: tensor.to(device) for name, tensor in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits.reshape(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=BASE_DIR / "data/derived/reranker_train.jsonl")
    parser.add_argument("--val", type=Path, default=BASE_DIR / "data/derived/reranker_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "checkpoints/scifact_reranker")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        parser.error("--epochs, --patience, and --batch-size must be positive")
    if args.learning_rate <= 0 or not (8 <= args.max_length <= 512):
        parser.error("--learning-rate must be positive; --max-length must be 8..512")
    if args.train.resolve() == args.val.resolve():
        parser.error("Training and validation files must differ")

    train_pairs = load_pairs(args.train)
    val_pairs = load_pairs(args.val)
    train_claims = {p["claim_id"] for p in train_pairs}
    val_claims = {p["claim_id"] for p in val_pairs}
    overlap = train_claims & val_claims
    if overlap:
        raise ValueError(f"Train/validation claim leakage: {list(overlap)[:5]}")
    if not any(p["label"] == 1 for p in train_pairs) or not any(p["label"] == 0 for p in train_pairs):
        raise ValueError("Training requires both positive and negative pairs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = preferred_device(args.device)
    print(f"Device: {device}")
    print(f"Training: {len(train_claims)} claims, {len(train_pairs)} pairs")
    print(f"Validation: {len(val_claims)} claims, {len(val_pairs)} pairs")
    print("Presumed negatives are not independently verified irrelevant")

    # Load the exact architecture and weights used for the pretrained reranker.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
    if model.config.num_labels != 1:
        raise ValueError("Expected a one-logit cross-encoder checkpoint")
    model.to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(EvidencePairs(train_pairs), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(EvidencePairs(val_pairs), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)

    n_pos = sum(p["label"] == 1 for p in train_pairs)
    n_neg = len(train_pairs) - n_pos
    pos_weight = n_neg / n_pos  # compensate for the 4-per-claim negative sampling
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(model, val_loader, val_pairs, device)
    print(f"Pretrained sampled-val: loss={baseline['val_bce_loss']:.4f}, "
          f"MRR@5={baseline['sampled_val_mrr_at_5']:.4f}, "
          f"Hit@1={baseline['sampled_val_hit_at_1']:.4f}")

    history = []
    best_loss = float("inf")
    best_epoch = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, val_pairs, device)
        history.append({"epoch": epoch, "train_weighted_bce": train_loss, **val_metrics})
        print(f"Epoch {epoch}: train weighted BCE={train_loss:.4f}, "
              f"val BCE={val_metrics['val_bce_loss']:.4f}, "
              f"val MRR@5={val_metrics['sampled_val_mrr_at_5']:.4f}, "
              f"val Hit@1={val_metrics['sampled_val_hit_at_1']:.4f}")

        if val_metrics["val_bce_loss"] < best_loss - 1e-5:
            best_loss = val_metrics["val_bce_loss"]
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  Saved best checkpoint: {args.output_dir}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation BCE did not improve")
                break

    summary = {
        "pretrained_checkpoint": PRETRAINED_MODEL,
        "best_epoch": best_epoch,
        "selection_metric": "minimum unweighted validation BCE",
        "baseline_sampled_validation": baseline,
        "history": history,
        "train_claims": len(train_claims),
        "validation_claims": len(val_claims),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "train_sha256": sha256_file(args.train),
        "validation_sha256": sha256_file(args.val),
        "positive_weight_for_training": pos_weight,
        "negative_label_caveat": "Zero labels are presumed negatives, not verified irrelevance",
        "candidate_pool_caveat": "Sampled validation pairs, not SciFact development hybrid top-20",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best validation epoch: {best_epoch}")
    print(f"Saved model + summary: {args.output_dir}")
    print("Next: compare saved model against pretrained reranker on the SAME cached hybrid candidates")


if __name__ == "__main__":
    main()

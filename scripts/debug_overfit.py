"""Quick diagnostic: can the RetroTransformer overfit 10 examples?

Tests different learning rates and model sizes to find the issue.
"""
from __future__ import annotations
import json
import sys
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def run_overfit_test(
    vocab_size, d_model, n_layers, d_ff, nhead,
    src_tensors, tgt_tensors, pad_id, lr, n_epochs, tag
):
    """Run a single overfit test and return best accuracy."""
    from rasyn.models.retro.model import RetroTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RetroTransformer(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead,
        num_encoder_layers=n_layers, num_decoder_layers=n_layers,
        dim_feedforward=d_ff, dropout=0.0, pad_token_id=pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n[{tag}] d={d_model}, layers={n_layers}, params={n_params:,}, lr={lr}")

    dataset = TensorDataset(src_tensors.to(device), tgt_tensors.to(device))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        for src_ids, tgt_ids in loader:
            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            logits = model(src_ids, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                for src_ids, tgt_ids in loader:
                    tgt_in = tgt_ids[:, :-1]
                    tgt_tgt = tgt_ids[:, 1:]
                    logits = model(src_ids, tgt_in)
                    preds = logits.argmax(dim=-1)
                    mask = tgt_tgt != pad_id
                    correct = ((preds == tgt_tgt) & mask).sum().item()
                    total = mask.sum().item()
                    acc = correct / max(total, 1)
                    best_acc = max(best_acc, acc)

            logger.info(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f} ({correct}/{total})")

            # Show a prediction
            if epoch == n_epochs - 1:
                from rasyn.models.retro.tokenizer import CharSmilesTokenizer
                tok = CharSmilesTokenizer()
                pred_str = tok.decode(preds[0].tolist())
                tgt_str = tok.decode(tgt_tgt[0].tolist())
                logger.info(f"  Pred: {pred_str[:80]}")
                logger.info(f"  True: {tgt_str[:80]}")

                # Check logits distribution for first token
                first_logits = logits[0, 0, :]  # First example, first position
                probs = torch.softmax(first_logits, dim=0)
                top5 = probs.topk(5)
                logger.info(f"  Top-5 probs at position 0: {[(tok.id2token.get(i.item(), '?'), f'{p.item():.3f}') for p, i in zip(top5.values, top5.indices)]}")

    return best_acc


def main():
    data_path = PROJECT_ROOT / "data/processed/uspto50k/edit_conditioned_train.jsonl"
    logger.info(f"Loading data from {data_path}...")

    from rasyn.models.retro.tokenizer import CharSmilesTokenizer
    tokenizer = CharSmilesTokenizer()

    # Load 10 examples
    import re
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", ex["prompt"])
            product = prod_match.group(1).strip() if prod_match else ""
            synth_match = re.search(r"<SYNTHONS>\s+(.+?)\s+<LG_HINTS>", ex["prompt"])
            synthons = synth_match.group(1).strip() if synth_match else ""
            reactants = ex["completion"]
            if product and reactants:
                examples.append((product, synthons, reactants))
            if len(examples) >= 10:
                break

    logger.info(f"Loaded {len(examples)} examples")

    # Encode to fixed tensors (with conditioning)
    src_list = []
    tgt_list = []
    for prod, synth, react in examples:
        if synth:
            src_text = f"{prod}|{synth}"
        else:
            src_text = prod
        src_ids = tokenizer.encode(src_text, max_len=256)
        tgt_ids = tokenizer.encode(react, max_len=128)
        src_list.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_list.append(torch.tensor(tgt_ids, dtype=torch.long))

    src_tensors = torch.stack(src_list)
    tgt_tensors = torch.stack(tgt_list)

    logger.info(f"SRC shape: {src_tensors.shape}, TGT shape: {tgt_tensors.shape}")
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # Test 1: Default config (6+6 layers, d=512) with lr=1e-3
    run_overfit_test(
        tokenizer.vocab_size, 512, 6, 2048, 8,
        src_tensors, tgt_tensors, tokenizer.pad_token_id,
        lr=1e-3, n_epochs=500, tag="Default-lr1e3"
    )

    # Test 2: Default config with higher lr
    run_overfit_test(
        tokenizer.vocab_size, 512, 6, 2048, 8,
        src_tensors, tgt_tensors, tokenizer.pad_token_id,
        lr=1e-2, n_epochs=500, tag="Default-lr1e2"
    )

    # Test 3: Tiny model (2 layers, d=128) with lr=1e-3
    run_overfit_test(
        tokenizer.vocab_size, 128, 2, 512, 4,
        src_tensors, tgt_tensors, tokenizer.pad_token_id,
        lr=1e-3, n_epochs=500, tag="Tiny-lr1e3"
    )

    # Test 4: Tiny model with higher lr
    run_overfit_test(
        tokenizer.vocab_size, 128, 2, 512, 4,
        src_tensors, tgt_tensors, tokenizer.pad_token_id,
        lr=5e-3, n_epochs=500, tag="Tiny-lr5e3"
    )

    # Test 5: Default config with SGD (no momentum adaption)
    logger.info("\n=== SGD Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from rasyn.models.retro.model import RetroTransformer
    model = RetroTransformer(
        vocab_size=tokenizer.vocab_size, d_model=512, nhead=8,
        num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.0, pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    dataset = TensorDataset(src_tensors.to(device), tgt_tensors.to(device))
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for epoch in range(500):
        model.train()
        for src_ids, tgt_ids in loader:
            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]
            logits = model(src_ids, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                for src_ids, tgt_ids in loader:
                    tgt_in = tgt_ids[:, :-1]
                    tgt_tgt = tgt_ids[:, 1:]
                    logits = model(src_ids, tgt_in)
                    preds = logits.argmax(dim=-1)
                    mask = tgt_tgt != tokenizer.pad_token_id
                    correct = ((preds == tgt_tgt) & mask).sum().item()
                    total = mask.sum().item()
                    acc = correct / max(total, 1)
            logger.info(f"  [SGD] Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

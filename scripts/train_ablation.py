"""Ablation study for RetroTransformer v2.

Trains multiple ablation variants, each with one component disabled,
to quantify the contribution of each v2 improvement:

  1. full          — All v2 components (baseline)
  2. no_regex      — Character-level tokenizer instead of regex
  3. no_copy       — Disable copy/pointer mechanism (generation only)
  4. no_augment    — Train on canonical-only data (no augmentation, augment_idx=0)
  5. no_rxn_class  — Remove <RXN_k> reaction class prefix from input
  6. no_segments   — Set all segment IDs to 0 (no product/synthon distinction)

All variants use the same hyperparameters as the full v2 model:
  lr=3e-4, warmup=2000, batch=64, d_model=512, nhead=8, n_layers=6, d_ff=2048

Usage:
    # Single ablation:
    python -u scripts/train_ablation.py --ablation no_copy

    # Run ALL ablations sequentially:
    python -u scripts/train_ablation.py --all

    # Run a specific subset:
    python -u scripts/train_ablation.py --ablation no_copy --ablation no_segments

    # Override hyperparameters:
    python -u scripts/train_ablation.py --ablation no_copy --epochs 100 --batch-size 32

    # RunPod background training:
    nohup python -u scripts/train_ablation.py --all > ablation_all.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import RDKit at module level so failures are loud
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent

# ── Ablation variant names ────────────────────────────────────────────
ABLATION_VARIANTS = [
    "full",
    "no_regex",
    "no_copy",
    "no_augment",
    "no_rxn_class",
    "no_segments",
]


# ── Logging ───────────────────────────────────────────────────────────

def setup_logger(variant: str, output_dir: Path) -> logging.Logger:
    """Create a logger that writes both to stdout and a variant-specific log file."""
    logger = logging.getLogger(f"ablation.{variant}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / f"ablation_{variant}.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── No-Copy Model: standard seq2seq without pointer mechanism ─────────

class RetroTransformerNoCopy(nn.Module):
    """Encoder-Decoder Transformer WITHOUT copy/pointer mechanism.

    Same architecture as RetroTransformerV2 but uses only generation head.
    The forward() signature stays compatible: returns (log_probs, dummy_copy_lambda).
    """

    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        num_segments: int = 2,
        num_rxn_classes: int = 11,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        # Token embedding (shared encoder/decoder)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Segment embedding (product=0, synthon=1)
        self.segment_embedding = nn.Embedding(num_segments, d_model)

        # Reaction class embedding
        self.rxn_class_embedding = nn.Embedding(num_rxn_classes, d_model)

        # Positional encoding
        from rasyn.models.retro.model_v2 import SinusoidalPositionalEncoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder (standard, no need for custom layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logging.getLogger(__name__).info(f"RetroTransformerNoCopy: {total_params:,} total params")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, src_ids, segment_ids=None):
        src_emb = self.embedding(src_ids) * math.sqrt(self.d_model)
        if segment_ids is not None:
            src_emb = src_emb + self.segment_embedding(segment_ids)
        src_emb = self.pos_encoding(src_emb)
        src_padding_mask = src_ids == self.pad_token_id
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        return memory, src_padding_mask

    def forward(self, src_ids, tgt_ids, segment_ids=None):
        memory, src_padding_mask = self._encode(src_ids, segment_ids)

        tgt_emb = self.pos_encoding(
            self.embedding(tgt_ids) * math.sqrt(self.d_model)
        )

        tgt_len = tgt_ids.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt_ids.device
        )
        tgt_padding_mask = tgt_ids == self.pad_token_id

        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        logits = self.output_proj(output)
        log_probs = F.log_softmax(logits, dim=-1)

        # Return dummy copy_lambda = 0 everywhere (no copy)
        copy_lambda = torch.zeros(
            src_ids.shape[0], tgt_ids.shape[1],
            device=src_ids.device,
        )

        return log_probs, copy_lambda

    @torch.no_grad()
    def generate_greedy(self, src_ids, bos_token_id, eos_token_id,
                        max_len=128, segment_ids=None):
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        memory, src_padding_mask = self._encode(src_ids, segment_ids)

        tgt_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoding(
                self.embedding(tgt_ids) * math.sqrt(self.d_model)
            )
            tgt_len = tgt_ids.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device
            )

            output = self.decoder(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_ids == self.pad_token_id,
            )

            logits = self.output_proj(output[:, -1, :])
            next_tokens = logits.argmax(dim=-1)

            finished = finished | (next_tokens == eos_token_id)
            next_tokens[finished] = eos_token_id
            tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

        results = []
        for i in range(batch_size):
            ids = tgt_ids[i].tolist()
            try:
                eos_pos = ids.index(eos_token_id, 1)
                ids = ids[1:eos_pos]
            except ValueError:
                ids = ids[1:]
            results.append(ids)
        return results

    @torch.no_grad()
    def generate_beam(self, src_ids, bos_token_id, eos_token_id,
                      beam_size=10, max_len=128, length_penalty=0.6,
                      diversity_penalty=0.0, segment_ids=None):
        """Beam search for the no-copy model."""
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        all_results = []

        for b in range(batch_size):
            single_src = src_ids[b:b+1]
            single_seg = segment_ids[b:b+1] if segment_ids is not None else None

            memory, src_padding_mask = self._encode(single_src, single_seg)

            memory = memory.repeat(beam_size, 1, 1)
            src_padding_mask = src_padding_mask.repeat(beam_size, 1)

            beam_ids = torch.full(
                (beam_size, 1), bos_token_id, dtype=torch.long, device=device
            )
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = -1e9

            finished_beams: list[tuple[list[int], float]] = []
            active_mask = torch.ones(beam_size, dtype=torch.bool, device=device)

            for step in range(max_len):
                n_active = active_mask.sum().item()
                if n_active == 0:
                    break

                active_idx = active_mask.nonzero(as_tuple=True)[0]
                active_beam_ids = beam_ids[active_idx]
                active_memory = memory[active_idx]
                active_src_mask = src_padding_mask[active_idx]
                active_scores = beam_scores[active_idx]

                tgt_emb = self.pos_encoding(
                    self.embedding(active_beam_ids) * math.sqrt(self.d_model)
                )
                tgt_len = active_beam_ids.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt_len, device=device
                )

                output = self.decoder(
                    tgt_emb, active_memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=active_src_mask,
                    tgt_key_padding_mask=active_beam_ids == self.pad_token_id,
                )

                logits = self.output_proj(output[:, -1, :])
                last_log_probs = F.log_softmax(logits, dim=-1)

                next_scores = active_scores.unsqueeze(1) + last_log_probs
                next_scores = next_scores.view(-1)
                k = min(beam_size, next_scores.numel())
                topk_scores, topk_flat = next_scores.topk(k, dim=-1)

                beam_indices = topk_flat // self.vocab_size
                token_indices = topk_flat % self.vocab_size

                new_beam_ids = torch.cat([
                    active_beam_ids[beam_indices],
                    token_indices.unsqueeze(1),
                ], dim=1)
                new_beam_scores = topk_scores

                new_active_mask = torch.ones(k, dtype=torch.bool, device=device)
                for i in range(k):
                    if token_indices[i].item() == eos_token_id:
                        seq_len = new_beam_ids.shape[1] - 1
                        norm_score = new_beam_scores[i].item() / max(seq_len, 1) ** length_penalty
                        ids = new_beam_ids[i].tolist()[1:-1]
                        finished_beams.append((ids, norm_score))
                        new_active_mask[i] = False

                still_active = new_active_mask.nonzero(as_tuple=True)[0]
                if len(still_active) == 0 or len(finished_beams) >= beam_size:
                    break

                n_keep = min(len(still_active), beam_size)
                keep_idx = still_active[:n_keep]

                beam_ids = torch.full(
                    (beam_size, new_beam_ids.shape[1]),
                    self.pad_token_id, dtype=torch.long, device=device,
                )
                beam_scores = torch.full((beam_size,), -1e9, device=device)
                active_mask = torch.zeros(beam_size, dtype=torch.bool, device=device)

                beam_ids[:n_keep] = new_beam_ids[keep_idx]
                beam_scores[:n_keep] = new_beam_scores[keep_idx]
                active_mask[:n_keep] = True

            if len(finished_beams) < beam_size:
                for i in range(min(beam_size, beam_ids.shape[0])):
                    if active_mask[i]:
                        ids = beam_ids[i].tolist()[1:]
                        ids = [t for t in ids if t != self.pad_token_id]
                        score = beam_scores[i].item() / max(len(ids), 1) ** length_penalty
                        finished_beams.append((ids, score))

            finished_beams.sort(key=lambda x: x[1], reverse=True)
            all_results.append(finished_beams[:beam_size])

        return all_results


# ── Dataset wrapper for no_segments ablation ──────────────────────────

class ZeroSegmentDataset(Dataset):
    """Wraps a RetroDatasetV2 and sets all segment_ids to 0."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["segment_ids"] = torch.zeros_like(item["segment_ids"])
        return item


# ── Evaluation ────────────────────────────────────────────────────────

def is_valid_smiles(smi: str) -> bool:
    if not RDKIT_AVAILABLE:
        return False
    mol = Chem.MolFromSmiles(smi.strip())
    return mol is not None


def check_validity(smiles_str: str) -> bool:
    if not RDKIT_AVAILABLE:
        return False
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    for p in parts:
        p = p.strip()
        if p and not is_valid_smiles(p):
            return False
    return len(parts) > 0


def canonicalize_and_sort(smiles_str: str) -> str:
    if not RDKIT_AVAILABLE:
        return ""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        mol = Chem.MolFromSmiles(p)
        if mol:
            canon.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon)) if canon else ""


def normalize_reactants(smiles_str: str) -> str:
    """Canonicalize and sort reactant SMILES for comparison."""
    if not RDKIT_AVAILABLE:
        return smiles_str
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        mol = Chem.MolFromSmiles(p)
        if mol:
            canon.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon)) if canon else ""


def evaluate_validation(model, val_loader, tokenizer, device, logger):
    """Validation: loss, token accuracy, exact match, validity, copy rate."""
    model.eval()
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_canon = 0
    total_valid = 0
    total_examples = 0
    total_copy_sum = 0.0
    total_copy_count = 0
    n_batches = 0

    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available -- validity/canon metrics will be 0")

    with torch.no_grad():
        for batch in val_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            seg_ids = batch["segment_ids"].to(device)

            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            log_probs, copy_lambda = model(src_ids, tgt_in, seg_ids)
            loss = loss_fn(
                log_probs.reshape(-1, log_probs.size(-1)),
                tgt_tgt.reshape(-1),
            )

            total_loss += loss.item()
            n_batches += 1

            # Token accuracy
            preds = log_probs.argmax(dim=-1)
            mask = tgt_tgt != tokenizer.pad_token_id
            total_correct += ((preds == tgt_tgt) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Copy gate stats
            copy_mask = mask.float()
            cl = copy_lambda[:, :mask.shape[1]]
            total_copy_sum += (cl * copy_mask).sum().item()
            total_copy_count += copy_mask.sum().item()

            # Greedy exact match + validity
            pred_ids_list = model.generate_greedy(
                src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id,
                max_len=128, segment_ids=seg_ids,
            )
            for i, pred_ids in enumerate(pred_ids_list):
                pred_str = tokenizer.decode(pred_ids)
                tgt_str = tokenizer.decode(tgt_ids[i].tolist())
                if pred_str == tgt_str:
                    total_exact += 1
                if check_validity(pred_str):
                    total_valid += 1
                pred_c = canonicalize_and_sort(pred_str)
                tgt_c = canonicalize_and_sort(tgt_str)
                if pred_c and tgt_c and pred_c == tgt_c:
                    total_canon += 1
                total_examples += 1

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_token_acc": total_correct / max(total_tokens, 1),
        "val_exact_match": total_exact / max(total_examples, 1),
        "val_canon_exact": total_canon / max(total_examples, 1),
        "val_validity": total_valid / max(total_examples, 1),
        "val_copy_rate": total_copy_sum / max(total_copy_count, 1),
        "val_examples": total_examples,
    }


def _get_segment_ids(tokenizer, token_ids: list[int]) -> list[int]:
    """Compute segment IDs, handling both regex and char tokenizers."""
    if hasattr(tokenizer, "get_segment_ids"):
        return tokenizer.get_segment_ids(token_ids)
    # Fallback for CharSmilesTokenizer: find '|' manually
    pipe_id = tokenizer.token2id.get("|", -1)
    segment_ids = []
    current_segment = 0
    for tid in token_ids:
        if tid == pipe_id:
            current_segment = 1
        segment_ids.append(current_segment)
    return segment_ids


def evaluate_beam(model, tokenizer, data_path, device, logger,
                  beam_size=10, max_samples=5000, use_rxn_class=True):
    """Test-set evaluation with beam search. Returns Top-1/3/5/10."""
    logger.info(f"Beam evaluation: beam_size={beam_size}, max_samples={max_samples}")

    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex.get("augment_idx", 0) == 0:
                examples.append(ex)
    if len(examples) > max_samples:
        examples = examples[:max_samples]

    logger.info(f"  Evaluating {len(examples)} test samples")

    top1 = top3 = top5 = top10 = 0
    total = 0
    valid_count = 0
    total_predictions = 0
    start = time.time()

    # Determine max_len based on tokenizer type
    is_char_tokenizer = not hasattr(tokenizer, "get_segment_ids")
    encode_max_len = 512 if is_char_tokenizer else 256
    decode_max_len = 256 if is_char_tokenizer else 128

    model.eval()
    for i, ex in enumerate(examples):
        src_text = ex["src_text"]
        tgt_text = ex["tgt_text"]
        rxn_class = ex.get("reaction_class", 0)

        gt = normalize_reactants(tgt_text)
        if not gt:
            continue

        # Prepend reaction class (if this variant uses it)
        if use_rxn_class and rxn_class and 1 <= rxn_class <= 10:
            src_input = f"<RXN_{rxn_class}> {src_text}"
        else:
            src_input = src_text

        encoded = tokenizer.encode(src_input, max_len=encode_max_len)
        src_ids = torch.tensor(
            [encoded],
            dtype=torch.long, device=device,
        )
        seg_ids = torch.tensor(
            [_get_segment_ids(tokenizer, encoded)],
            dtype=torch.long, device=device,
        )

        beam_results = model.generate_beam(
            src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id,
            beam_size=beam_size, max_len=decode_max_len,
            diversity_penalty=0.5, segment_ids=seg_ids,
        )[0]

        predictions = []
        for token_ids, score in beam_results:
            pred_str = tokenizer.decode(token_ids)
            total_predictions += 1
            if check_validity(pred_str):
                valid_count += 1
            norm = normalize_reactants(pred_str)
            if norm and norm not in predictions:
                predictions.append(norm)

        total += 1
        if predictions and predictions[0] == gt:
            top1 += 1
        if gt in predictions[:3]:
            top3 += 1
        if gt in predictions[:5]:
            top5 += 1
        if gt in predictions[:10]:
            top10 += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            logger.info(
                f"  [{i+1}/{len(examples)}] "
                f"Top-1={top1/total:.4f} Top-3={top3/total:.4f} "
                f"Top-5={top5/total:.4f} Top-10={top10/total:.4f} "
                f"Valid={valid_count/max(total_predictions,1)*100:.1f}% "
                f"({(i+1)/elapsed:.1f} s/s)"
            )

    validity_rate = valid_count / max(total_predictions, 1)
    results = {
        "total": total,
        "beam_size": beam_size,
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "top10": top10 / max(total, 1),
        "validity_rate": validity_rate,
    }

    logger.info(f"\n  BEAM RESULTS ({total} samples, beam={beam_size}):")
    logger.info(f"    Top-1:  {results['top1']:.4f} ({top1}/{total})")
    logger.info(f"    Top-3:  {results['top3']:.4f} ({top3}/{total})")
    logger.info(f"    Top-5:  {results['top5']:.4f} ({top5}/{total})")
    logger.info(f"    Top-10: {results['top10']:.4f} ({top10}/{total})")
    logger.info(f"    Valid:  {validity_rate*100:.1f}%")

    return results


# ── Training Loop (single variant) ───────────────────────────────────

def train_variant(
    model, train_loader, val_loader, tokenizer, device,
    output_dir: Path, logger,
    epochs=200, lr=3e-4, warmup_steps=2000, weight_decay=0.01,
    gradient_clip=1.0, label_smoothing=0.05,
    log_every=50, eval_every=5, sample_every=10,
    train_dataset=None, model_config=None,
    patience=15, variant_name="full",
):
    """Full training loop. Identical to train_retro_v2.py but self-contained."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    log_file = output_dir / "training_log.jsonl"
    best_val_loss = float("inf")
    best_val_exact = 0.0
    epochs_without_improvement = 0
    global_step = 0
    start_time = time.time()

    logger.info(f"Training variant={variant_name} for {epochs} epochs, {total_steps} total steps")
    logger.info(f"  LR: {lr}, warmup: {warmup_steps}, label_smoothing: {label_smoothing}")
    logger.info(f"  Early stopping patience: {patience} epochs")
    logger.info(f"  Output: {output_dir}")

    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        epoch_copy_sum = 0.0
        epoch_copy_count = 0
        epoch_batches = 0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            seg_ids = batch["segment_ids"].to(device)

            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            log_probs, copy_lambda = model(src_ids, tgt_in, seg_ids)

            # Label smoothing
            if label_smoothing > 0:
                nll_loss = loss_fn(
                    log_probs.reshape(-1, log_probs.size(-1)),
                    tgt_tgt.reshape(-1),
                )
                smooth_loss = -log_probs.reshape(-1, log_probs.size(-1)).mean(dim=-1)
                mask_flat = (tgt_tgt.reshape(-1) != tokenizer.pad_token_id).float()
                smooth_loss = (smooth_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
            else:
                loss = loss_fn(
                    log_probs.reshape(-1, log_probs.size(-1)),
                    tgt_tgt.reshape(-1),
                )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            # Token accuracy
            preds = log_probs.argmax(dim=-1)
            mask = tgt_tgt != tokenizer.pad_token_id
            epoch_correct += ((preds == tgt_tgt) & mask).sum().item()
            epoch_tokens += mask.sum().item()

            # Copy gate stats
            copy_mask = mask.float()
            cl = copy_lambda[:, :mask.shape[1]]
            epoch_copy_sum += (cl * copy_mask).sum().item()
            epoch_copy_count += copy_mask.sum().item()

            if global_step % log_every == 0:
                avg_loss = epoch_loss / epoch_batches
                token_acc = epoch_correct / max(epoch_tokens, 1)
                copy_rate = epoch_copy_sum / max(epoch_copy_count, 1)
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time

                logger.info(
                    f"[{variant_name}] Step {global_step} | Epoch {epoch} | "
                    f"loss={loss.item():.4f} | avg={avg_loss:.4f} | "
                    f"tok_acc={token_acc:.4f} | copy={copy_rate:.3f} | "
                    f"lr={current_lr:.6f} | {global_step/elapsed:.1f} s/s"
                )

                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "variant": variant_name,
                        "step": global_step, "epoch": epoch,
                        "loss": loss.item(), "avg_loss": avg_loss,
                        "token_acc": token_acc, "copy_rate": copy_rate,
                        "lr": current_lr,
                    }) + "\n")

        # End of epoch
        logger.info(
            f"\n--- [{variant_name}] Epoch {epoch}/{epochs} ---"
            f" avg_loss={epoch_loss/max(epoch_batches,1):.4f}"
            f" tok_acc={epoch_correct/max(epoch_tokens,1):.4f}"
            f" copy_rate={epoch_copy_sum/max(epoch_copy_count,1):.3f}"
        )

        # Validation
        if val_loader and epoch % eval_every == 0:
            metrics = evaluate_validation(model, val_loader, tokenizer, device, logger)
            n_exact = int(metrics['val_exact_match'] * metrics['val_examples'])
            n_canon = int(metrics['val_canon_exact'] * metrics['val_examples'])
            logger.info(
                f"  VAL: loss={metrics['val_loss']:.4f} | "
                f"tok_acc={metrics['val_token_acc']:.4f} | "
                f"exact={metrics['val_exact_match']:.4f} ({n_exact}/{metrics['val_examples']}) | "
                f"canon={metrics['val_canon_exact']:.4f} ({n_canon}/{metrics['val_examples']}) | "
                f"validity={metrics['val_validity']:.4f} | "
                f"copy={metrics['val_copy_rate']:.3f}"
            )

            # Track improvement
            improved = False
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                improved = True
            if metrics["val_exact_match"] > best_val_exact:
                best_val_exact = metrics["val_exact_match"]
                improved = True

            if improved:
                epochs_without_improvement = 0
                _save_checkpoint(
                    model, tokenizer, output_dir / "best",
                    config=model_config,
                    extra={
                        "epoch": epoch,
                        "val_metrics": metrics,
                        "variant": variant_name,
                    },
                )
                # Save training state for resume
                torch.save({
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_exact": best_val_exact,
                    "epochs_without_improvement": 0,
                }, output_dir / "best" / "training_state.pt")
                logger.info(
                    f"  New best! val_loss={best_val_loss:.4f} exact={best_val_exact:.4f}"
                )
            else:
                epochs_without_improvement += eval_every
                logger.info(
                    f"  No improvement for {epochs_without_improvement} epochs "
                    f"(patience={patience}, best_loss={best_val_loss:.4f}, "
                    f"best_exact={best_val_exact:.4f})"
                )

            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(
                    f"\n*** [{variant_name}] EARLY STOPPING at epoch {epoch} ***"
                    f"\n  No improvement for {epochs_without_improvement} epochs."
                    f"\n  Best val_loss={best_val_loss:.4f}, best_exact={best_val_exact:.4f}"
                )
                _save_checkpoint(
                    model, tokenizer, output_dir / "early_stopped",
                    config=model_config,
                    extra={
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "best_val_exact": best_val_exact,
                        "early_stopped": True,
                        "variant": variant_name,
                    },
                )
                return best_val_loss, best_val_exact

            with open(log_file, "a") as f:
                metrics["epoch"] = epoch
                metrics["type"] = "validation"
                metrics["variant"] = variant_name
                metrics["epochs_without_improvement"] = epochs_without_improvement
                f.write(json.dumps(metrics) + "\n")

        # Sample generations
        if train_dataset and epoch % sample_every == 0:
            model.eval()
            import random as _random
            indices = _random.sample(
                range(len(train_dataset)), min(3, len(train_dataset))
            )
            logger.info(f"\n  Samples (epoch {epoch}):")
            for idx in indices:
                item = train_dataset[idx]
                src = item["src_ids"].unsqueeze(0).to(device)
                seg = item["segment_ids"].unsqueeze(0).to(device)
                pred_ids = model.generate_greedy(
                    src, tokenizer.bos_token_id, tokenizer.eos_token_id,
                    max_len=128, segment_ids=seg,
                )
                pred = tokenizer.decode(pred_ids[0])
                true = tokenizer.decode(item["tgt_ids"].tolist())
                valid = check_validity(pred) if pred else False
                match = "EXACT" if pred == true else ("valid" if valid else "invalid")
                logger.info(f"    [{match}] P: {pred[:70]}")
                logger.info(f"           T: {true[:70]}")
            model.train()

        # Periodic checkpoint
        if epoch % 20 == 0:
            _save_checkpoint(
                model, tokenizer, output_dir / f"epoch_{epoch}",
                config=model_config,
                extra={"epoch": epoch, "variant": variant_name},
            )

    # Final save
    _save_checkpoint(
        model, tokenizer, output_dir / "final",
        config=model_config,
        extra={"epoch": epochs, "best_val_loss": best_val_loss, "variant": variant_name},
    )

    elapsed = time.time() - start_time
    logger.info(f"\n[{variant_name}] Training complete in {elapsed/3600:.1f} hours")
    logger.info(f"[{variant_name}] Best val_loss: {best_val_loss:.4f}, exact: {best_val_exact:.4f}")

    return best_val_loss, best_val_exact


def _save_checkpoint(model, tokenizer, output_dir, config=None, extra=None):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "vocab": tokenizer.get_vocab_dict(),
        "version": "v2_ablation",
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, output_dir / "model.pt")


# ── Build variant: tokenizer, dataset, model ──────────────────────────

def build_variant(
    variant: str,
    data_path: Path,
    device: str,
    d_model: int,
    nhead: int,
    n_layers: int,
    d_ff: int,
    max_src_len: int,
    max_tgt_len: int,
    batch_size: int,
    conditioning_dropout: float,
    val_split: float,
    logger,
):
    """Build tokenizer, datasets, data loaders, and model for a given ablation variant.

    Returns:
        (model, tokenizer, train_loader, val_loader, train_dataset, val_dataset, model_config)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Building variant: {variant}")
    logger.info(f"{'='*60}")

    # ── Determine effective data path ──────────────────────────────
    # no_augment: filter to augment_idx=0 only
    effective_data_path = data_path
    if variant == "no_augment":
        logger.info("  Filtering to canonical-only (augment_idx=0)...")
        filtered_path = data_path.parent / "augmented_train_no_augment.jsonl"
        n_kept = 0
        n_total = 0
        with open(data_path) as fin, open(filtered_path, "w") as fout:
            for line in fin:
                n_total += 1
                ex = json.loads(line.strip())
                if ex.get("augment_idx", 0) == 0:
                    fout.write(line)
                    n_kept += 1
        logger.info(f"  Filtered {n_total} -> {n_kept} examples (canonical only)")
        effective_data_path = filtered_path

    # ── Determine use_rxn_class ────────────────────────────────────
    use_rxn_class = variant != "no_rxn_class"

    # ── Build tokenizer ────────────────────────────────────────────
    if variant == "no_regex":
        logger.info("  Building CHARACTER-LEVEL tokenizer (ablation: no_regex)...")
        from rasyn.models.retro.tokenizer import CharSmilesTokenizer

        all_texts = []
        with open(effective_data_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                src = ex["src_text"]
                tgt = ex["tgt_text"]
                rxn_class = ex.get("reaction_class", 0)
                # Include reaction class prefix in text to ensure chars are in vocab
                if use_rxn_class and rxn_class and 1 <= rxn_class <= 10:
                    src = f"<RXN_{rxn_class}> {src}"
                all_texts.append(src)
                all_texts.append(tgt)

        # Char tokenizer needs all individual characters
        all_chars = set()
        for t in all_texts:
            all_chars.update(t)

        # Build char tokenizer with these chars
        tokenizer = CharSmilesTokenizer(chars=sorted(all_chars))
        logger.info(f"  CharSmilesTokenizer: {tokenizer.vocab_size} tokens")

        # Need max_src_len and max_tgt_len to be larger for char-level
        # Char-level sequences are ~2x longer
        effective_max_src_len = min(max_src_len * 2, 512)
        effective_max_tgt_len = min(max_tgt_len * 2, 256)
        logger.info(
            f"  Adjusted max lengths for char-level: "
            f"src={effective_max_src_len}, tgt={effective_max_tgt_len}"
        )
    else:
        logger.info("  Building REGEX tokenizer...")
        from rasyn.models.retro.tokenizer_v2 import RegexSmilesTokenizer

        all_texts = []
        with open(effective_data_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                all_texts.append(ex["src_text"])
                all_texts.append(ex["tgt_text"])

        tokenizer = RegexSmilesTokenizer.build_from_data(all_texts)
        logger.info(f"  RegexSmilesTokenizer: {tokenizer.vocab_size} tokens")
        effective_max_src_len = max_src_len
        effective_max_tgt_len = max_tgt_len

    # ── Build datasets ─────────────────────────────────────────────
    # CharSmilesTokenizer has a different API: no get_segment_ids, no get_rxn_class_token
    # We need a compatible dataset for the char tokenizer case.

    if variant == "no_regex":
        # For char-level, we use a custom dataset that handles the differences
        train_dataset, val_dataset = _load_data_char_tokenizer(
            data_path=effective_data_path,
            tokenizer=tokenizer,
            val_split=val_split,
            max_src_len=effective_max_src_len,
            max_tgt_len=effective_max_tgt_len,
            conditioning_dropout=conditioning_dropout,
            use_rxn_class=use_rxn_class,
        )
    else:
        from rasyn.models.retro.data_v2 import load_retro_data_v2
        train_dataset, val_dataset = load_retro_data_v2(
            data_path=effective_data_path,
            tokenizer=tokenizer,
            val_split=val_split,
            max_src_len=effective_max_src_len,
            max_tgt_len=effective_max_tgt_len,
            conditioning_dropout=conditioning_dropout,
            use_reaction_class=use_rxn_class,
        )

    # no_segments: wrap dataset to zero out segment_ids
    if variant == "no_segments":
        logger.info("  Wrapping dataset with ZeroSegmentDataset (ablation: no_segments)")
        train_dataset = ZeroSegmentDataset(train_dataset)
        val_dataset = ZeroSegmentDataset(val_dataset)

    logger.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Build model ────────────────────────────────────────────────
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": n_layers,
        "num_decoder_layers": n_layers,
        "dim_feedforward": d_ff,
        "max_seq_len": max(effective_max_src_len, effective_max_tgt_len),
        "pad_token_id": tokenizer.pad_token_id,
        "num_segments": 2,
        "num_rxn_classes": 11,
        "variant": variant,
    }

    if variant == "no_copy":
        logger.info("  Building RetroTransformerNoCopy (ablation: no_copy)")
        model = RetroTransformerNoCopy(**{
            k: v for k, v in model_config.items() if k != "variant"
        }).to(device)
    else:
        from rasyn.models.retro.model_v2 import RetroTransformerV2
        logger.info(f"  Building RetroTransformerV2 (variant={variant})")
        model = RetroTransformerV2(**{
            k: v for k, v in model_config.items() if k != "variant"
        }).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {total_params:,}")

    # ── Data loaders ───────────────────────────────────────────────
    from rasyn.models.retro.data_v2 import collate_fn_v2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_v2, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=2, pin_memory=True,
    )

    return model, tokenizer, train_loader, val_loader, train_dataset, val_dataset, model_config


# ── Character-level dataset loader ────────────────────────────────────

class CharRetroDataset(Dataset):
    """Dataset for the no_regex ablation: uses CharSmilesTokenizer.

    Provides the same interface as RetroDatasetV2 (src_ids, tgt_ids, segment_ids).
    Segment IDs are computed by finding the '|' character position.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 256,
        conditioning_dropout: float = 0.2,
        use_rxn_class: bool = True,
    ):
        import random as _random
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.conditioning_dropout = conditioning_dropout
        self.use_rxn_class = use_rxn_class

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        import random as _random
        ex = self.examples[idx]
        src_text = ex["src_text"]
        tgt_text = ex["tgt_text"]
        rxn_class = ex.get("reaction_class", 0)

        # Parse product and synthons
        if "|" in src_text:
            product, synthons = src_text.split("|", 1)
        else:
            product = src_text
            synthons = ""

        # Conditioning dropout
        use_conditioning = synthons and (_random.random() > self.conditioning_dropout)

        if use_conditioning:
            src_input = f"{product}|{synthons}"
        else:
            src_input = product

        # Prepend reaction class if available
        if self.use_rxn_class and rxn_class and 1 <= rxn_class <= 10:
            src_input = f"<RXN_{rxn_class}> {src_input}"

        # Tokenize
        src_ids = self.tokenizer.encode(src_input, max_len=self.max_src_len)
        tgt_ids = self.tokenizer.encode(tgt_text, max_len=self.max_tgt_len)

        # Compute segment IDs by finding '|' character position
        pipe_id = self.tokenizer.token2id.get("|", -1)
        segment_ids = []
        current_segment = 0
        for tid in src_ids:
            if tid == pipe_id:
                current_segment = 1
            segment_ids.append(current_segment)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
        }


def _load_data_char_tokenizer(
    data_path, tokenizer, val_split=0.1,
    max_src_len=512, max_tgt_len=256,
    conditioning_dropout=0.2, use_rxn_class=True,
    seed=42,
):
    """Load data for the char tokenizer case, using the same rxn_id-based split."""
    import random as _random

    all_examples = []
    with open(data_path) as f:
        for line in f:
            all_examples.append(json.loads(line.strip()))

    # Group by rxn_id
    rxn_groups: dict[str, list[dict]] = {}
    for ex in all_examples:
        rxn_id = ex.get("rxn_id", str(id(ex)))
        if rxn_id not in rxn_groups:
            rxn_groups[rxn_id] = []
        rxn_groups[rxn_id].append(ex)

    # Split
    unique_rxn_ids = sorted(rxn_groups.keys())
    rng = _random.Random(seed)
    rng.shuffle(unique_rxn_ids)

    n_val = int(len(unique_rxn_ids) * val_split)
    val_rxn_ids = set(unique_rxn_ids[:n_val])

    train_examples = []
    val_examples = []
    for rxn_id, examples in rxn_groups.items():
        if rxn_id in val_rxn_ids:
            for ex in examples:
                if ex.get("augment_idx", 0) == 0:
                    val_examples.append(ex)
                    break
            else:
                val_examples.append(examples[0])
        else:
            train_examples.extend(examples)

    train_dataset = CharRetroDataset(
        examples=train_examples, tokenizer=tokenizer,
        max_src_len=max_src_len, max_tgt_len=max_tgt_len,
        conditioning_dropout=conditioning_dropout,
        use_rxn_class=use_rxn_class,
    )
    val_dataset = CharRetroDataset(
        examples=val_examples, tokenizer=tokenizer,
        max_src_len=max_src_len, max_tgt_len=max_tgt_len,
        conditioning_dropout=0.0,
        use_rxn_class=use_rxn_class,
    )

    return train_dataset, val_dataset


# ── Main ──────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--ablation", "ablation_list", multiple=True,
    type=click.Choice(ABLATION_VARIANTS, case_sensitive=True),
    help="Ablation variant(s) to train. Use multiple times for several.",
)
@click.option("--all", "run_all", is_flag=True, help="Run ALL ablation variants sequentially.")
@click.option("--data", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--test-data", default="data/processed/uspto50k/augmented_train.jsonl",
              help="Test set for beam evaluation. Use augmented_test.jsonl if available.")
@click.option("--output-dir", default="checkpoints/retro_v2/ablation")
@click.option("--epochs", default=200, type=int)
@click.option("--batch-size", default=64, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--d-model", default=512, type=int)
@click.option("--nhead", default=8, type=int)
@click.option("--n-layers", default=6, type=int)
@click.option("--d-ff", default=2048, type=int)
@click.option("--max-src-len", default=256, type=int)
@click.option("--max-tgt-len", default=128, type=int)
@click.option("--warmup-steps", default=2000, type=int)
@click.option("--label-smoothing", default=0.05, type=float)
@click.option("--conditioning-dropout", default=0.2, type=float)
@click.option("--val-split", default=0.1, type=float)
@click.option("--patience", default=15, type=int, help="Early stopping patience (0=disable)")
@click.option("--eval-beam-size", default=10, type=int, help="Beam size for test evaluation")
@click.option("--eval-max-samples", default=5000, type=int, help="Max test samples for beam eval")
@click.option("--skip-eval", is_flag=True, help="Skip beam evaluation after training")
@click.option("--device", default="auto")
def main(
    ablation_list, run_all, data, test_data, output_dir,
    epochs, batch_size, lr, d_model, nhead, n_layers, d_ff,
    max_src_len, max_tgt_len, warmup_steps, label_smoothing,
    conditioning_dropout, val_split, patience,
    eval_beam_size, eval_max_samples, skip_eval, device,
):
    """Train ablation variants of RetroTransformer v2."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = PROJECT_ROOT / data
    test_data_path = PROJECT_ROOT / test_data
    base_output_dir = PROJECT_ROOT / output_dir

    # Determine which variants to run
    if run_all:
        variants = ABLATION_VARIANTS
    elif ablation_list:
        variants = list(ablation_list)
    else:
        click.echo(
            "Error: specify --ablation <variant> or --all. "
            f"Valid variants: {ABLATION_VARIANTS}"
        )
        sys.exit(1)

    # ── Summary ────────────────────────────────────────────────────
    root_logger = logging.getLogger("ablation")
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root_logger.addHandler(console)

    root_logger.info("=" * 70)
    root_logger.info("RETROTRANSFORMER V2 ABLATION STUDY")
    root_logger.info("=" * 70)
    root_logger.info(f"Device:     {device}")
    root_logger.info(f"Data:       {data_path}")
    root_logger.info(f"Test data:  {test_data_path}")
    root_logger.info(f"Output:     {base_output_dir}")
    root_logger.info(f"Variants:   {variants}")
    root_logger.info(f"Epochs:     {epochs}")
    root_logger.info(f"Batch:      {batch_size}")
    root_logger.info(f"LR:         {lr}")
    root_logger.info(f"Architecture: d_model={d_model}, nhead={nhead}, "
                     f"n_layers={n_layers}, d_ff={d_ff}")
    root_logger.info(f"Patience:   {patience}")
    root_logger.info(f"RDKit:      {RDKIT_AVAILABLE}")
    root_logger.info("=" * 70)

    # ── Run each variant ───────────────────────────────────────────
    all_results = {}
    overall_start = time.time()

    for variant_idx, variant in enumerate(variants):
        variant_start = time.time()
        variant_output = base_output_dir / variant
        variant_logger = setup_logger(variant, variant_output)

        variant_logger.info(f"\n{'#'*70}")
        variant_logger.info(f"# ABLATION VARIANT {variant_idx+1}/{len(variants)}: {variant}")
        variant_logger.info(f"{'#'*70}")

        try:
            # Build
            (model, tokenizer, train_loader, val_loader,
             train_dataset, val_dataset, model_config) = build_variant(
                variant=variant,
                data_path=data_path,
                device=device,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                d_ff=d_ff,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                batch_size=batch_size,
                conditioning_dropout=conditioning_dropout,
                val_split=val_split,
                logger=variant_logger,
            )

            # Train
            best_val_loss, best_val_exact = train_variant(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                output_dir=variant_output,
                logger=variant_logger,
                epochs=epochs,
                lr=lr,
                warmup_steps=warmup_steps,
                label_smoothing=label_smoothing,
                train_dataset=train_dataset,
                model_config=model_config,
                patience=patience,
                variant_name=variant,
            )

            # Beam evaluation on test set
            beam_results = None
            if not skip_eval and test_data_path.exists():
                variant_logger.info(f"\n{'='*60}")
                variant_logger.info(f"TEST SET BEAM EVALUATION: {variant}")
                variant_logger.info(f"{'='*60}")

                # Load best checkpoint for evaluation
                best_ckpt = variant_output / "best" / "model.pt"
                if best_ckpt.exists():
                    variant_logger.info(f"Loading best checkpoint from {best_ckpt}")
                    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()

                beam_results = evaluate_beam(
                    model=model,
                    tokenizer=tokenizer,
                    data_path=test_data_path,
                    device=device,
                    logger=variant_logger,
                    beam_size=eval_beam_size,
                    max_samples=eval_max_samples,
                    use_rxn_class=(variant != "no_rxn_class"),
                )

            variant_elapsed = time.time() - variant_start

            # Store results
            all_results[variant] = {
                "best_val_loss": best_val_loss,
                "best_val_exact": best_val_exact,
                "beam_results": beam_results,
                "training_hours": variant_elapsed / 3600,
            }

            variant_logger.info(
                f"\n[{variant}] Done in {variant_elapsed/3600:.2f} hours"
            )

            # Save individual variant results
            results_file = variant_output / "ablation_results.json"
            with open(results_file, "w") as f:
                json.dump(all_results[variant], f, indent=2, default=str)
            variant_logger.info(f"Results saved to {results_file}")

        except Exception as e:
            variant_logger.error(f"FAILED: {variant}: {e}", exc_info=True)
            all_results[variant] = {"error": str(e)}

        # Free GPU memory between variants
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final summary ──────────────────────────────────────────────
    overall_elapsed = time.time() - overall_start

    root_logger.info(f"\n\n{'='*70}")
    root_logger.info("ABLATION STUDY RESULTS SUMMARY")
    root_logger.info(f"{'='*70}")
    root_logger.info(f"Total time: {overall_elapsed/3600:.2f} hours")
    root_logger.info("")

    # Table header
    root_logger.info(
        f"{'Variant':<16} {'Val Loss':>10} {'Val Exact':>10} "
        f"{'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Top-10':>8} "
        f"{'Hours':>8}"
    )
    root_logger.info("-" * 90)

    for variant in variants:
        res = all_results.get(variant, {})
        if "error" in res:
            root_logger.info(f"{variant:<16} ERROR: {res['error']}")
            continue

        val_loss = res.get("best_val_loss", float("nan"))
        val_exact = res.get("best_val_exact", float("nan"))
        hours = res.get("training_hours", 0)

        beam = res.get("beam_results") or {}
        t1 = beam.get("top1", float("nan"))
        t3 = beam.get("top3", float("nan"))
        t5 = beam.get("top5", float("nan"))
        t10 = beam.get("top10", float("nan"))

        root_logger.info(
            f"{variant:<16} {val_loss:>10.4f} {val_exact:>10.4f} "
            f"{t1:>8.4f} {t3:>8.4f} {t5:>8.4f} {t10:>8.4f} "
            f"{hours:>8.2f}"
        )

    root_logger.info("-" * 90)

    # Save combined summary
    summary_path = base_output_dir / "ablation_summary.json"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "variants": variants,
            "results": {k: v for k, v in all_results.items()},
            "total_hours": overall_elapsed / 3600,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "d_model": d_model,
                "nhead": nhead,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "patience": patience,
                "warmup_steps": warmup_steps,
                "label_smoothing": label_smoothing,
            },
        }, f, indent=2, default=str)

    root_logger.info(f"\nSummary saved to {summary_path}")
    root_logger.info("Done.")


if __name__ == "__main__":
    main()

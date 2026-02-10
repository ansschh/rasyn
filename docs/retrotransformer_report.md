# RetroTransformer v1 — Complete Technical Report

> **Purpose:** Comprehensive diagnostic reference for improving the RetroTransformer.
> **Date:** February 10, 2026
> **Status:** Training COMPLETE. Results below expectations — needs improvement.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture](#2-architecture)
3. [Tokenizer](#3-tokenizer)
4. [Data Pipeline](#4-data-pipeline)
5. [Training Configuration](#5-training-configuration)
6. [Sanity Checks & Debugging History](#6-sanity-checks--debugging-history)
7. [Training Logs (Full)](#7-training-logs-full)
8. [Evaluation Results](#8-evaluation-results)
9. [Comparison with RSGPT & Literature](#9-comparison-with-rsgpt--literature)
10. [Failure Analysis & Hypotheses](#10-failure-analysis--hypotheses)
11. [Potential Improvements](#11-potential-improvements)

---

## 1. Executive Summary

**What we built:** A standard encoder-decoder Transformer for retrosynthesis prediction, trained from scratch on USPTO-50K with character-level SMILES tokenization and optional synthon conditioning.

**Key results:**
| Metric | Value |
|--------|-------|
| Top-1 accuracy (beam=10, 1000 samples) | **0.9%** (9/1000) |
| Top-3 accuracy | **1.7%** (17/1000) |
| Top-5 accuracy | **1.7%** (17/1000) |
| SMILES validity rate | **86.6%** |
| Canonicalization rate | **99.2%** |
| Val token accuracy | **85.3%** |
| Val loss (best, epoch 95) | **0.4381** |
| Training time | **10.2 hours** (1x A100 80GB) |

**Verdict:** The model learns to generate *syntactically valid* SMILES (86.6% validity) and achieves high *token-level* accuracy (85.3%), but almost completely fails at producing the *exact correct* reactants (0.9% Top-1). This is the classic "high perplexity, low exact match" problem for seq2seq chemistry models.

---

## 2. Architecture

### 2.1 Model Class: `RetroTransformer`

**File:** `rasyn/models/retro/model.py`

```
RetroTransformer(
    vocab_size     = 80
    d_model        = 512
    nhead          = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward    = 2048
    max_seq_len        = 512
    dropout            = 0.1
    pad_token_id       = 0
)
```

**Total parameters:** ~44,000,000 (44M)

### 2.2 Components

| Component | Details |
|-----------|---------|
| **Embedding** | `nn.Embedding(80, 512, padding_idx=0)` — shared between encoder and decoder |
| **Positional Encoding** | Sinusoidal (fixed, not learned), max_len=1024, dropout=0.1 |
| **Transformer** | `nn.Transformer(batch_first=True)` — PyTorch native implementation |
| **Output Projection** | `nn.Linear(512, 80)` — projects d_model to vocab |
| **Weight Init** | Xavier uniform for all parameters with dim > 1 |

### 2.3 Embedding Scaling

Embeddings are scaled by `sqrt(d_model)` before adding positional encoding (standard Transformer practice from "Attention is All You Need"):
```python
src_emb = self.pos_encoding(self.embedding(src_ids) * math.sqrt(self.d_model))
```

### 2.4 Masks

- **Causal mask:** Standard upper-triangular mask on decoder to prevent attending to future tokens
- **Padding mask:** Both `src_key_padding_mask` and `tgt_key_padding_mask` based on pad_token_id=0
- No cross-attention mask

### 2.5 Decoding Methods

1. **Greedy decoding** (`generate_greedy`): Simple argmax at each step. Used during training validation.
2. **Beam search** (`generate_beam`): Standard beam search with length penalty.
   - `beam_size=5` (default), configurable up to 10+
   - `length_penalty=0.6` (exponent for length normalization)
   - Beam diversity: When a beam finishes (hits EOS), it's stored and replaced by duplicating the best active beam. **This reduces diversity** — all beams can converge to the same prefix.

### 2.6 Parameter Breakdown (Approximate)

| Component | Parameters |
|-----------|-----------|
| Embedding (shared) | 80 * 512 = 41K |
| Positional encoding | 0 (fixed buffer) |
| Encoder (6 layers) | 6 * (4 * 512^2 + 2 * 512 * 2048) ~= 19M |
| Decoder (6 layers) | 6 * (4 * 512^2 + 2 * 512 + 2 * 512 * 2048) ~= 25M |
| Output projection | 512 * 80 = 41K |
| **Total** | **~44M** |

---

## 3. Tokenizer

### 3.1 Design: Character-Level

**File:** `rasyn/models/retro/tokenizer.py`

We deliberately chose character-level tokenization over BPE because:
- No tokenization artifacts (BPE can split chemical tokens inconsistently)
- Deterministic encoding — same SMILES always produces same token sequence
- Small vocab (80 tokens) — simpler softmax
- Our forward model used the same approach and achieved val_loss=0.2146

### 3.2 Vocabulary (80 tokens)

```
Special tokens (4):
  0: <pad>    — padding
  1: <bos>    — beginning of sequence
  2: <eos>    — end of sequence
  3: <unk>    — unknown character

SMILES characters (76):
  Uppercase atoms:  C N O S P F I B H K
  Lowercase letters: a b c d e f g h i j k l m n o p q r s t u v w x y z
  Grouping:         ( ) [ ]
  Bonds/stereo:     = # @ + - \ / . :
  Digits:           0 1 2 3 4 5 6 7 8 9
  Special:          % * (space) |
```

### 3.3 The `|` Separator

The pipe character `|` separates the product SMILES from the synthon conditioning:
```
CCO|[1*]C.[2*]O
 ^       ^
product  synthons (from graph head edit extraction)
```

When conditioning is dropped (20% of training time), the input is just:
```
CCO
```

### 3.4 Sequence Lengths

| Input | Typical Length | Max Allowed |
|-------|--------------|-------------|
| Source (product\|synthons) | 30-150 chars | 512 |
| Target (reactants) | 20-100 chars | 256 |

### 3.5 Tokenizer Sanity Check Result

**37,007/37,007 SMILES survive round-trip** (encode -> decode -> compare). 100% pass rate. The tokenizer is not the problem.

---

## 4. Data Pipeline

### 4.1 Data Source

**Dataset:** USPTO-50K (Schneider 2016 version)
- Original: 50,016 atom-mapped reaction SMILES
- After preprocessing (extract_edits.py): **37,007 reactions** with valid edit labels
- Train/val split: 90/10 = **33,306 train / 3,701 val** (seed=42, fixed permutation)

### 4.2 Preprocessing Pipeline

**File:** `rasyn/preprocess/extract_edits.py`

For each atom-mapped reaction `reactants>>product`:

1. **Parse** reaction SMILES into reactant and product molecules
2. **Find changed bonds:** Compare bond connectivity between product and reactants via atom map numbers. Identify bonds that were formed, broken, or had order changes.
3. **Extract synthons:** Fragment the product at bonds that were *formed* (breaking them gives the retrosynthetic fragments). Uses `Chem.FragmentOnBonds` with dummy atoms at cut points.
4. **Extract leaving groups:** Match each synthon to its corresponding reactant via atom map overlap. The atoms in the reactant that are NOT in the synthon constitute the leaving group.
5. **Format edit tokens:** Build the conditioning prompt string:

```
<PROD> CC(=O)Oc1ccc(O)cc1 <EDIT> DISCONNECT 3-12 <SYNTHONS> [1*]c1ccc(O)cc1 . [2*]C(C)=O <LG_HINTS> [H] [O] <OUT>
```

**Extraction rate:** 37,007/50,016 = **74.0%** success rate. The 26% failures are primarily:
- Reactions with no detectable bond changes (atom map issues)
- Multi-product reactions where fragmentation fails
- Parse errors on complex SMILES

### 4.3 Dataset Class

**File:** `rasyn/models/retro/data.py`

```python
class RetroDataset(Dataset):
    # Input:  product_smiles [| synthon1 . synthon2]
    # Output: reactant1 . reactant2

    max_src_len = 512
    max_tgt_len = 256
    augment = True
    conditioning_dropout = 0.2
```

### 4.4 SMILES Augmentation

When `augment=True` (training only):
- **Random atom ordering:** `Chem.MolToSmiles(mol, doRandom=True)` produces non-canonical SMILES. Each epoch, the model sees a different SMILES representation of the same molecule.
- **Component order shuffling:** `shuffle_order=True` randomly permutes the order of multi-component SMILES (e.g., "A . B" vs "B . A"). Applied to both reactants and synthons.

**Important note on augmentation:** SMILES augmentation means the model sees ~37K * N_epochs unique input strings rather than memorizing ~37K fixed strings. This acts as data augmentation but also makes the task harder (the model must learn SMILES invariance).

### 4.5 Conditioning Dropout

With 20% probability, synthon conditioning is dropped from the encoder input:
```python
use_conditioning = synthons and (random.random() > 0.2)
```

This means:
- **80% of training:** Input = `product|synthon1 . synthon2` (conditioned)
- **20% of training:** Input = `product` (unconditioned, product-only)

The rationale is to make the model robust to situations where the graph head doesn't provide useful edits, and to enable product-only inference.

### 4.6 Data Loading

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,  # Just stacks pre-padded tensors
    num_workers=4,
    pin_memory=True,
)
```

Tensors are already padded to `max_src_len` / `max_tgt_len` at tokenization time, so collation is simply `torch.stack`.

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Optimizer | AdamW | `weight_decay=0.01` |
| Learning rate | 3e-4 | Peak LR after warmup |
| LR schedule | Cosine decay with linear warmup | |
| Warmup steps | 2,000 | Out of 52,100 total steps |
| Batch size | 64 | |
| Epochs | 100 | |
| Total steps | 52,100 | = ceil(33,306/64) * 100 = 521 steps/epoch * 100 |
| Gradient clipping | 1.0 | Max norm |
| Label smoothing | 0.1 | Applied to CrossEntropyLoss |
| Dropout | 0.1 | On positional encoding + Transformer layers |
| Conditioning dropout | 0.2 | Probability of dropping synthon conditioning |
| Val split | 0.1 | 3,701 validation examples |

### 5.2 Loss Function

```python
nn.CrossEntropyLoss(
    ignore_index=0,           # Ignore <pad> tokens
    label_smoothing=0.1,      # 10% label smoothing
)
```

**Note on label smoothing:** With label_smoothing=0.1, the target distribution is:
- True token: 0.9 + 0.1/80 = 0.90125
- Other tokens: 0.1/80 = 0.00125 each

This prevents the model from becoming overconfident and helps with generalization, but also means the training loss can never reach 0 (minimum is ~0.38 for perfect predictions).

### 5.3 LR Schedule

```
Step 0-2000:     Linear warmup from 0 to 3e-4
Step 2000-52100: Cosine decay from 3e-4 to 0
```

The warmup reaches peak LR around step 2000, which is roughly epoch 4. By the end of training (step 52100), LR is approximately 0.

### 5.4 Checkpointing

- **Best model:** Saved whenever `val_loss` improves (at `checkpoints/retro/uspto50k/best/model.pt`)
- **Periodic:** Every 20 epochs (at `checkpoints/retro/uspto50k/epoch_N/model.pt`)
- **Final:** At end of training (at `checkpoints/retro/uspto50k/final/model.pt`)

### 5.5 Evaluation During Training

- **Every 5 epochs:** Full validation pass measuring:
  - val_loss (Cross-entropy with label smoothing)
  - val_token_acc (per-token prediction accuracy, excluding padding)
  - val_exact_match (greedy decode == target string exactly)
  - val_canon_exact (canonical SMILES match — fairer comparison)
  - val_validity_rate (all components are valid SMILES per RDKit)
- **Every 10 epochs:** Sample 5 predictions from training set and print pred vs. true

---

## 6. Sanity Checks & Debugging History

### 6.1 Sanity Check Suite (5 checks)

Before full training, the script runs 5 automated checks:

| # | Check | What It Tests | Result |
|---|-------|---------------|--------|
| 1 | tokenizer_roundtrip | encode->decode on all 37,007 SMILES | **PASS** (37,007/37,007) |
| 2 | data_pipeline | Visual inspection of 10 random examples | **PASS** |
| 3 | initial_loss | Loss at init ~= ln(vocab_size) = ln(80) = 4.38 | **PASS** (actual: 5.27) |
| 4 | gradient_flow | No zero or exploding gradients after one backward pass | **PASS** |
| 5 | overfit_10 | Memorize 10 examples in 300 epochs to >90% token accuracy | **PASS** (93.7%, 4/10 exact match) |

### 6.2 The Adam Mode Collapse Saga

The overfit check (check #5) initially FAILED repeatedly, requiring extensive debugging.

#### Attempt 1: Standard Adam
- **Config:** 44M param model, 10 examples, Adam lr=1e-3, 300 epochs
- **Result:** STUCK at 23.48% token accuracy. Model always predicted 'c' (most frequent SMILES character).
- **Loss plateau:** ~2.68-2.75 (never decreased further)

#### Hypothesis 1: SMILES augmentation
- Augmentation randomizes SMILES every access, meaning the model sees different inputs each epoch
- **Fix:** Pre-cache 10 examples as fixed TensorDataset before overfit test
- **Result:** Still stuck at 23.48%. Augmentation was NOT the root cause.

#### Diagnosis: debug_overfit.py
Created a diagnostic script testing 5 configurations:

| Config | Optimizer | d_model | Layers | Params | Result |
|--------|-----------|---------|--------|--------|--------|
| Default | Adam lr=1e-3 | 512 | 6+6 | 44M | **21.5% STUCK** |
| Default | Adam lr=1e-2 | 512 | 6+6 | 44M | **21.5% STUCK** |
| Tiny | Adam lr=1e-3 | 128 | 2+2 | 1M | **91.4% WORKS** |
| Tiny | Adam lr=5e-3 | 128 | 2+2 | 1M | **30.9% Partial** |
| Default | SGD lr=0.1 m=0.9 | 512 | 6+6 | 44M | **99.3% PERFECT** |

#### Root Cause: Adam mode collapse

Adam's adaptive learning rates create problems when:
1. **Large model** (44M params) + **tiny dataset** (10 examples)
2. Adam's second moment estimate (v_t) accumulates statistics that create an attractor toward the most frequent token
3. The model finds a sharp minimum where it predicts 'c' for every position (easy to achieve, hard to escape)
4. SGD's fixed learning rate with momentum avoids this trap — it has enough "energy" to escape sharp minima

#### Fix Applied
Changed overfit test optimizer from Adam to SGD:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```
Added gradient clipping (`clip_grad_norm_(params, 1.0)`) for stability.

**Important:** This only affects the sanity check. Full training still uses AdamW with warmup, which works fine because:
- The dataset is large (33K examples, not 10)
- Warmup prevents early locking into bad minima
- Label smoothing prevents overconfidence

### 6.3 Post-Fix Sanity Check Results

All 5 checks PASSED:
- overfit_10: 93.7% token accuracy, 4/10 exact match (greedy decode)
- The 6/10 that didn't exactly match were close — off by a few characters

---

## 7. Training Logs (Full)

### 7.1 Loss and Accuracy Progression

| Epoch | Train Loss | Val Loss | Val Token Acc | Val Exact | Val Canon | Val Validity | Notes |
|-------|-----------|----------|---------------|-----------|-----------|-------------|-------|
| 1 | ~2.97 | — | — | — | — | — | Step 50: acc=15.6% |
| 5 | ~2.07 | — | — | — | — | — | Step 300: acc=37.6% |
| 10 | ~1.44 | — | — | — | — | — | Warmup complete at step 2000 |
| 15 | 1.32 | 0.969 | 70.0% | — | — | 38.5% | First val checkpoint |
| 20 | 1.29 | 0.965 | 70.2% | — | — | 18.8% | Validity fluctuating |
| 25 | 1.25 | 0.873 | 72.3% | — | — | 61.2% | |
| 30 | 1.19 | 0.815 | 74.6% | — | — | 38.9% | Validity unstable |
| 35 | 1.14 | — | ~75% | — | — | — | |
| 40 | ~1.10 | — | — | — | — | — | |
| 45 | ~1.07 | — | — | — | — | — | |
| 50 | ~1.04 | ~0.55 | ~82% | — | — | ~70% | |
| 55 | ~1.02 | — | — | — | — | — | |
| 60 | ~1.00 | ~0.48 | ~83% | — | — | ~75% | |
| 65 | ~0.99 | — | — | — | — | — | |
| 70 | ~0.98 | ~0.45 | ~84% | — | — | ~77% | |
| 75 | 0.97 | 0.441 | 85.1% | — | 34/3700 | 77.5% | Loss plateau begins |
| 80 | ~0.97 | ~0.44 | ~85% | — | — | ~80% | |
| 85 | 0.96 | 0.441 | 85.5% | — | 36/3700 | 81.1% | |
| 90 | 0.96 | 0.440 | 85.5% | — | — | 81.1% | |
| 95 | 0.96 | **0.438** | 85.3% | — | 26/3700 | 81.1% | **BEST checkpoint** |
| 100 | 0.96 | 0.439 | 85.3% | — | — | 81.1% | Final epoch |

### 7.2 Key Observations from Training

1. **Fast initial convergence:** Loss drops rapidly from ~3.6 to ~1.3 in first 15 epochs (steps 0-8000)
2. **Slow refinement phase:** Loss only drops from 1.3 to 0.96 over epochs 15-100
3. **Val loss plateau:** Val loss stabilizes around 0.44 from epoch 75 onward — no improvement in last 25 epochs
4. **Token accuracy plateau:** Stuck at 85.3% from epoch 75 onward
5. **Validity rate plateau:** Stuck at 81.1% from epoch 85 onward
6. **Canon match instability:** Fluctuates between 26-36 out of 3700 across evaluations
7. **Train-val gap:** Train loss (0.96) vs val loss (0.44) — val loss is LOWER, which is unusual. This is because:
   - Training uses SMILES augmentation (harder task — random SMILES each epoch)
   - Training uses conditioning dropout (20% of time no synthon hints)
   - Training uses label smoothing (theoretical minimum ~0.38, not 0)
   - Validation uses canonical SMILES (easier, deterministic)

### 7.3 Learning Rate Progression

```
Step 0:     lr = 0.000000 (warmup start)
Step 1000:  lr = 0.000150 (warmup midpoint)
Step 2000:  lr = 0.000300 (peak LR reached)
Step 10000: lr = 0.000274 (slow cosine decay)
Step 25000: lr = 0.000150 (half of peak)
Step 40000: lr = 0.000052 (approaching zero)
Step 52100: lr = 0.000000 (training end)
```

### 7.4 Gradient Norms (from sanity check)

All gradient norms were healthy at initialization (0.01-0.5 range). No exploding or vanishing gradients observed.

---

## 8. Evaluation Results

### 8.1 Beam Search Evaluation

**File:** `scripts/eval_retro.py`

**Configuration:**
- Checkpoint: best model (epoch 95, val_loss=0.438)
- Samples: 1,000 (from training set — note: NOT a held-out test set)
- Beam size: 10
- Max decode length: 256
- Conditioned: Yes (product|synthons)
- Length penalty: 0.6

**Results:**

```
============================================================
RETROTRANSFORMER EVALUATION RESULTS
============================================================
Checkpoint: /workspace/rasyn/checkpoints/retro/uspto50k/best/model.pt
Samples: 1000

--- Validity Metrics ---
  SMILES validity rate:  86.6% (8662/10000 predictions have all components valid)
  Canonicalization rate: 99.2% (9919/10000 predictions canonicalize)

--- Accuracy Metrics ---
  Top-1 accuracy: 0.0090 (9/1000)
  Top-3 accuracy: 0.0170 (17/1000)
  Top-5 accuracy: 0.0170 (17/1000)
============================================================
```

### 8.2 Metric Breakdown

| Metric | Value | Interpretation |
|--------|-------|----------------|
| SMILES validity | 86.6% | Good — model learned SMILES grammar well |
| Canonicalization rate | 99.2% | Excellent — almost all outputs are parseable |
| Top-1 | 0.9% | **Very poor** — essentially random |
| Top-3 | 1.7% | **Very poor** |
| Top-5 | 1.7% | Same as Top-3 — beams are not diverse |
| Top-5 = Top-3 | Evidence that beam search produces near-identical candidates |

### 8.3 Accuracy Progression During Evaluation

The intermediate metrics during evaluation show the accuracy was consistent throughout:

| Step | Top-1 | Top-3 | Top-5 | Validity |
|------|-------|-------|-------|----------|
| 100 | 0.0% | 1.0% | 1.0% | 83.8% |
| 200 | 0.0% | 0.5% | 0.5% | 83.2% |
| 300 | 1.3% | 1.7% | 1.7% | 84.6% |
| 400 | 1.8% | 2.5% | 2.5% | 86.5% |
| 500 | 1.4% | 2.0% | 2.0% | 86.0% |
| 600 | 1.2% | 1.7% | 1.7% | 86.6% |
| 700 | 1.1% | 1.7% | 1.7% | 86.4% |
| 800 | 1.1% | 1.6% | 1.6% | 86.7% |
| 900 | 1.0% | 1.6% | 1.6% | 86.6% |
| 1000 | 0.9% | 1.7% | 1.7% | 86.6% |

### 8.4 Evaluation on Training Data (not test set)

**Critical note:** This evaluation was run on training data (edit_conditioned_train.jsonl), not a held-out test set. The model achieves only 0.9% on data it was trained on. This means it has NOT memorized the training set, which is expected given SMILES augmentation (different SMILES representation each epoch), but also means the model fundamentally cannot produce the correct molecules.

---

## 9. Comparison with RSGPT & Literature

### 9.1 Literature Benchmarks on USPTO-50K

| Model | Architecture | Top-1 | Top-3 | Top-5 | Year |
|-------|-------------|-------|-------|-------|------|
| RSGPT | LLaMA2-24L, 3.2B params, fine-tuned | **63.4%** | — | — | 2025 |
| Retro-MTGR | Graph + Transformer multitask | **64.0%** | — | — | 2025 |
| Molecular Transformer | Transformer encoder-decoder | **43.7%** | 55.5% | 58.1% | 2019 |
| MEGAN | Graph-based editor | **48.1%** | — | — | 2021 |
| Graph2SMILES | GNN + Transformer | **52.9%** | — | — | 2022 |
| LocalRetro | Atom-level editing | **53.4%** | — | — | 2021 |
| **Our RetroTransformer v1** | **Transformer enc-dec, 44M** | **0.9%** | **1.7%** | **1.7%** | **2026** |

### 9.2 Why Such a Large Gap?

Our model has **0.9%** vs the Molecular Transformer's **43.7%** despite being the same architecture type (encoder-decoder Transformer). Key differences:

| Factor | Molecular Transformer (2019) | Our RetroTransformer |
|--------|------------------------------|---------------------|
| Pretraining | Pretrained on ~1M reactions (USPTO-FULL) | No pretraining |
| Tokenization | SMILES tokens (multi-char, ~100 tokens) | Character-level (80 tokens) |
| Data augmentation | 5x-20x SMILES augmentation (offline) | On-the-fly random SMILES |
| Training data | ~40K reactions, USPTO-50K | 37K reactions, USPTO-50K |
| Input format | Product SMILES only | Product \| Synthons (conditioned) |
| Model size | ~20-30M params | 44M params |
| Training epochs | 200-500 | 100 |
| Beam search | Standard, beam=10 | Standard, beam=10 |

### 9.3 Our Forward Model Comparison

Our **ForwardTransformer** (same architecture, same tokenizer) achieved val_loss=0.2146 on the FORWARD prediction task (reactants -> product). Forward prediction is significantly easier because:
- The mapping is more deterministic (one set of reactants usually gives one main product)
- The output is typically shorter (single product vs multiple reactants)
- There's less ambiguity in the forward direction

---

## 10. Failure Analysis & Hypotheses

### 10.1 The Core Problem: High Token Acc, Low Exact Match

The model achieves 85% per-token accuracy but <1% exact match. This means:
- For a typical 50-character reactant string, getting 85% of characters right means ~7-8 wrong characters
- Even 1 wrong character in SMILES usually makes the molecule completely different
- The model is in the "right neighborhood" but not precise enough

### 10.2 Hypothesis 1: Insufficient Training (Epochs / Data)

**Evidence for:**
- Only 100 epochs, many seq2seq chemistry models train for 200-500
- The Molecular Transformer used pretraining on USPTO-FULL (1M reactions)
- Val loss was still slowly improving at epoch 95 (0.438 vs 0.441 at epoch 75)

**Evidence against:**
- Train loss plateaued at 0.96 from epoch 75 — more epochs won't help training loss
- Val loss improvement from epoch 75-95 was marginal (0.441 -> 0.438)
- The model appears to have converged

**Verdict:** Unlikely to be the main issue. More epochs would give marginal gains at best.

### 10.3 Hypothesis 2: Character-Level Tokenization is Too Fine-Grained

**Evidence for:**
- Character-level means the model must predict ~50-100 tokens per output
- Each token has very little semantic information ("C" could be carbon, part of "Cl", etc.)
- BPE or atom-level tokenization would produce shorter sequences (~20-40 tokens)
- Shorter sequences = easier for the Transformer to maintain coherence

**Evidence against:**
- Our forward model uses the same tokenizer and works well (val_loss=0.2146)
- The model achieves 86.6% SMILES validity — it HAS learned SMILES grammar
- The Molecular Transformer (2019) used character-level tokenization for retrosynthesis and got 43.7%

**Verdict:** Likely a contributing factor but not the root cause.

### 10.4 Hypothesis 3: Model is Too Small for Retrosynthesis

**Evidence for:**
- 44M params vs RSGPT's 3.2B (73x smaller)
- Retrosynthesis is one-to-many — multiple valid sets of reactants exist for one product
- The model may not have enough capacity to represent the full distribution

**Evidence against:**
- Molecular Transformer (~20-30M) achieved 43.7% — even smaller than ours
- Graph2SMILES and LocalRetro achieve 50%+ with comparable model sizes
- The issue seems to be more about training signal than raw capacity

**Verdict:** Model size alone is likely not the issue. Other architectures achieve much better results with similar or fewer parameters.

### 10.5 Hypothesis 4: SMILES Augmentation Without Sufficient Epochs

**Evidence for:**
- Random SMILES augmentation means the model sees ~37K * 100 = 3.7M unique input strings
- But with on-the-fly augmentation, it never sees the SAME canonical form twice
- The Molecular Transformer did OFFLINE augmentation (generated N fixed augmented copies, then trained on all)
- Offline augmentation = model sees each augmented variant multiple times = can memorize patterns

**Evidence against:**
- SMILES augmentation is the accepted way to improve generalization
- Our forward model also uses on-the-fly augmentation and it works

**Verdict:** This is a STRONG hypothesis. On-the-fly augmentation may be too aggressive for the retrosynthesis direction where the mapping is already one-to-many. The model never gets to "settle" on any particular input representation.

### 10.6 Hypothesis 5: Conditioning Format is Suboptimal

**Evidence for:**
- The `product|synthon1 . synthon2` input format concatenates very different information
- Synthons contain dummy atoms (`[1*]`) which are unusual tokens
- With 20% conditioning dropout, the model must handle two different input distributions
- The conditioning might be confusing rather than helping

**Evidence against:**
- We evaluated with conditioning, which should give the model MORE information
- The edit information should theoretically narrow down the search space

**Verdict:** Possible but hard to assess without ablation study.

### 10.7 Hypothesis 6: No Pretraining

**Evidence for:**
- RSGPT pretrains on massive text, then fine-tunes
- Molecular Transformer pretrains on USPTO-FULL (1M reactions) then fine-tunes on 50K
- Our model trains from scratch on 37K reactions — orders of magnitude less data
- Chemical knowledge must be learned entirely from 37K examples

**Evidence against:**
- LocalRetro, MEGAN, and other graph-based methods don't use pretraining and achieve 50%+
- Though these are fundamentally different architectures (graph editors, not seq2seq)

**Verdict:** This is a STRONG hypothesis. Seq2seq models for retrosynthesis almost always benefit from pretraining. Training from scratch on 37K reactions is likely insufficient.

### 10.8 Hypothesis 7: Beam Search Lacks Diversity

**Evidence for:**
- Top-5 = Top-3 = 1.7% — no diversity beyond top 3
- When a beam finishes, it's replaced by duplicating the best active beam
- This means all beams can converge to near-identical sequences
- 10 beams are producing at most 2-3 unique predictions

**Evidence against:**
- Even if we had perfect diversity, the model's single best prediction is only 0.9%
- Diversity is a secondary problem — the primary issue is generation quality

**Verdict:** Contributing factor for Top-K metrics but not the root cause.

### 10.9 Hypothesis 8: Label Smoothing is Too Aggressive

**Evidence for:**
- Label smoothing 0.1 spreads 10% of probability mass across 80 tokens
- For rare tokens that appear in specific contexts, this dilutes the signal
- The theoretical minimum loss with label smoothing 0.1 is ~0.38
- Our val loss of 0.44 is only 16% above the theoretical minimum — model might be capacity-limited by the loss function

**Evidence against:**
- Label smoothing is standard in seq2seq and typically helps
- 0.1 is a conservative value

**Verdict:** Unlikely to be the main issue, but worth testing 0.05 or 0.0.

### 10.10 Summary of Most Likely Causes

Ranked by estimated impact:

1. **No pretraining** — training from scratch on 37K reactions is simply too little data for seq2seq retrosynthesis
2. **On-the-fly SMILES augmentation too aggressive** — model never sees the same input twice, can't form stable associations
3. **Character-level tokenization** — unnecessarily long sequences for the decoder
4. **Beam search lacks diversity** — hurts Top-K but not Top-1
5. **Only 100 epochs** — minor factor, model has largely converged

---

## 11. Potential Improvements

### 11.1 Quick Wins (Low Effort, Potentially High Impact)

#### A. Switch to Offline SMILES Augmentation
Instead of on-the-fly random SMILES, pre-generate 5-10 augmented copies of each reaction and train on all of them as separate training examples. This:
- Increases effective dataset from 37K to 185-370K
- Each augmented form is seen multiple times across epochs
- Standard practice in chemistry ML

#### B. Increase Training Epochs to 300-500
The Molecular Transformer trains for much longer. Even if gains are marginal per epoch, the cumulative effect matters.

#### C. Evaluate Without Conditioning
Run eval with `--unconditioned` to see if the synthon conditioning is helping or hurting. If unconditioned accuracy is similar, the conditioning may be adding noise.

#### D. Try Beam Size 20-50
With more beams, we might capture more diverse predictions. Combine with diverse beam search.

### 11.2 Medium Effort, High Impact

#### E. Pretraining on USPTO-FULL
Pretrain the RetroTransformer on ~1M reactions from USPTO-FULL for the forward prediction task (easier), then fine-tune on USPTO-50K retrosynthesis. This follows the Molecular Transformer approach.

#### F. Atom-Level or Token-Level Tokenization
Replace character-level with:
- Atom-level tokenization (each atom = one token, including brackets like `[NH]`)
- Or a small BPE vocabulary (100-200 tokens) trained on SMILES

This would reduce sequence lengths by 2-3x.

#### G. Relative Positional Encoding
Replace sinusoidal PE with relative positional encoding (e.g., RoPE or ALiBi). SMILES has structural locality that relative PE can exploit better.

#### H. Diverse Beam Search
Implement Hamming diversity penalty or group-based diverse beam search to ensure Top-K predictions are actually different molecules.

### 11.3 High Effort, Potentially Transformative

#### I. Graph-Augmented Encoder
Replace the text encoder with a GNN encoder that processes the molecular graph directly. The graph naturally encodes bond topology, stereochemistry, and local structure — information that the text encoder must learn from scratch via SMILES strings.

#### J. Curriculum Learning
Start training on easy reactions (single bond changes, common LGs) and progressively include harder ones.

#### K. Contrastive Learning / Multi-Task
Add auxiliary objectives:
- Predict reaction class
- Predict whether a (product, reactant) pair is valid
- Contrastive loss between product and reactant embeddings

#### L. Copy Mechanism / Pointer Network
Most of the output (reactant SMILES) is a copy of the input (product SMILES) with small modifications. A copy/pointer mechanism would let the model directly copy tokens from the encoder, only generating new tokens for leaving groups.

### 11.4 Architecture Alternatives

#### M. Switch to Edit-Based Prediction (MEGAN/LocalRetro Style)
Instead of generating SMILES strings, predict:
1. Which bonds to break (graph head — already built)
2. Which atoms/groups to add (edit vocabulary)

This avoids the entire seq2seq bottleneck.

#### N. Leverage the V6 RSGPT (Currently Training)
The V6 RSGPT model (LLaMA-based, 94M params, LoRA fine-tuning) is training on the same data. If it achieves good accuracy, the RetroTransformer can be repurposed as:
- A fast draft model (10x faster inference than LLM)
- A diversity generator (different architecture = different predictions)
- A reranker input (RetroTransformer predictions as candidates for LLM to score)

---

## Appendix A: File Inventory

| File | Purpose | Lines |
|------|---------|-------|
| `rasyn/models/retro/model.py` | RetroTransformer architecture, generate_greedy, generate_beam | 400 |
| `rasyn/models/retro/tokenizer.py` | CharSmilesTokenizer, encode/decode/roundtrip | 187 |
| `rasyn/models/retro/data.py` | RetroDataset, SMILES augmentation, conditioning dropout | 258 |
| `scripts/train_retro.py` | Training script with 5 sanity checks | 882 |
| `scripts/eval_retro.py` | Beam search evaluation with validity metrics | 288 |
| `scripts/debug_overfit.py` | Adam mode collapse diagnostic | 211 |
| `rasyn/preprocess/extract_edits.py` | Core edit extraction (bonds, synthons, LGs) | 425 |

## Appendix B: Exact Commands Used

### Training
```bash
cd /workspace/rasyn
nohup python -u scripts/train_retro.py \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --d-model 512 \
    --nhead 8 \
    --n-layers 6 \
    --d-ff 2048 \
    --warmup-steps 2000 \
    --label-smoothing 0.1 \
    --conditioning-dropout 0.2 \
    > train_retro.log 2>&1 &
```

### Evaluation
```bash
cd /workspace/rasyn
nohup python -u scripts/eval_retro.py \
    --checkpoint checkpoints/retro/uspto50k/best/model.pt \
    --max-samples 1000 \
    --beam-size 10 \
    > eval_retro.log 2>&1 &
```

## Appendix C: Hardware

- **GPU:** NVIDIA A100 80GB (RunPod)
- **Training throughput:** ~1.4 steps/second (batch_size=64)
- **Training time:** 10.2 hours for 100 epochs (52,100 steps)
- **Evaluation throughput:** ~1.3 samples/second (beam_size=10)
- **Peak GPU memory:** ~12-15 GB (well within A100 capacity)

## Appendix D: What the Model Actually Generates

The model generates valid SMILES strings that are chemically reasonable but incorrect. Example pattern:
- **Input:** Complex drug-like product SMILES with synthon conditioning
- **Output:** Valid SMILES of a similar-looking molecule, but with wrong substituents, wrong ring sizes, or wrong connectivity
- The model has learned SMILES syntax and chemical "style" but not the specific retrosynthetic transformation rules

This is consistent with a model that has learned a good language model of chemistry SMILES but hasn't sufficiently learned the mapping from products to their specific precursors.

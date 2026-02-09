"""Check if BPE tokenization differs between training and inference."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

data_path = PROJECT_ROOT / "data" / "processed" / "uspto50k" / "edit_conditioned_train.jsonl"
with open(data_path) as f:
    ex = json.loads(f.readline().strip())
prompt = ex["prompt"]
completion = ex["completion"]

from transformers import AutoTokenizer
checkpoint = PROJECT_ROOT / "checkpoints" / "llm" / "uspto50k" / "final"
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

eos = tokenizer.eos_token
full_text = f"{prompt} {completion}{eos}"

# Tokenize separately
enc_prompt = tokenizer(prompt, return_tensors="pt")
enc_full = tokenizer(full_text, return_tensors="pt")
enc_prompt_sp = tokenizer(prompt + " ", return_tensors="pt")

ids_prompt = enc_prompt["input_ids"][0].tolist()
ids_full = enc_full["input_ids"][0].tolist()
ids_prompt_sp = enc_prompt_sp["input_ids"][0].tolist()

print(f"Prompt-only length: {len(ids_prompt)}")
print(f"Prompt+space length: {len(ids_prompt_sp)}")
print(f"Full text length: {len(ids_full)}")

# Compare token IDs at the boundary
print(f"\n=== LAST 10 TOKENS OF PROMPT-ONLY ===")
for i in range(max(0, len(ids_prompt)-10), len(ids_prompt)):
    tok = ids_prompt[i]
    decoded = tokenizer.decode([tok])
    print(f"  [{i}] token={tok} '{decoded}'")

print(f"\n=== TOKENS 50-70 OF FULL TEXT ===")
for i in range(50, min(70, len(ids_full))):
    tok = ids_full[i]
    decoded = tokenizer.decode([tok])
    print(f"  [{i}] token={tok} '{decoded}'")

# Direct comparison: are first N tokens identical?
min_len = min(len(ids_prompt), len(ids_full))
first_diff = None
for i in range(min_len):
    if ids_prompt[i] != ids_full[i]:
        first_diff = i
        break

if first_diff is None:
    print(f"\nFirst {min_len} tokens are IDENTICAL between prompt-only and full-text")
else:
    print(f"\nFIRST DIFFERENCE at position {first_diff}:")
    print(f"  Prompt-only: token={ids_prompt[first_diff]} '{tokenizer.decode([ids_prompt[first_diff]])}'")
    print(f"  Full text:   token={ids_full[first_diff]} '{tokenizer.decode([ids_full[first_diff]])}'")
    print(f"  Context (prompt): {ids_prompt[max(0,first_diff-3):first_diff+3]}")
    print(f"  Context (full):   {ids_full[max(0,first_diff-3):first_diff+3]}")

# Also compare prompt+space vs full text
min_len2 = min(len(ids_prompt_sp), len(ids_full))
first_diff2 = None
for i in range(min_len2):
    if ids_prompt_sp[i] != ids_full[i]:
        first_diff2 = i
        break

if first_diff2 is None:
    print(f"\nFirst {min_len2} tokens are IDENTICAL between prompt+space and full-text")
else:
    print(f"\nFIRST DIFFERENCE (prompt+space vs full) at position {first_diff2}:")
    print(f"  Prompt+sp: token={ids_prompt_sp[first_diff2]} '{tokenizer.decode([ids_prompt_sp[first_diff2]])}'")
    print(f"  Full text: token={ids_full[first_diff2]} '{tokenizer.decode([ids_full[first_diff2]])}'")

# Check multiple examples for consistency
print(f"\n=== CHECKING 5 EXAMPLES ===")
examples = []
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= 5: break
        examples.append(json.loads(line.strip()))

for i, ex in enumerate(examples):
    p = ex["prompt"]
    c = ex["completion"]
    ft = f"{p} {c}{eos}"

    ids_p = tokenizer(p, return_tensors="pt")["input_ids"][0].tolist()
    ids_f = tokenizer(ft, return_tensors="pt")["input_ids"][0].tolist()

    # Find first difference
    diff_pos = None
    for j in range(min(len(ids_p), len(ids_f))):
        if ids_p[j] != ids_f[j]:
            diff_pos = j
            break

    if diff_pos is None:
        match_str = f"first {min(len(ids_p), len(ids_f))} tokens match"
    else:
        match_str = f"DIFFER at pos {diff_pos}: prompt={ids_p[diff_pos]} full={ids_f[diff_pos]}"

    print(f"  Example {i+1}: prompt_len={len(ids_p)}, full_len={len(ids_f)}, {match_str}")

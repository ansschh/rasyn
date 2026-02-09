"""Diagnose why the model generates </s> after <OUT>."""
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load one example
data_path = PROJECT_ROOT / "data" / "processed" / "uspto50k" / "edit_conditioned_train.jsonl"
with open(data_path) as f:
    ex = json.loads(f.readline().strip())
prompt = ex["prompt"]
completion = ex["completion"]
print(f"PROMPT: {prompt[:120]}...")
print(f"COMPLETION: {completion}")

# Load tokenizer and model
from rasyn.models.llm.model import load_rsgpt_model
from peft import PeftModel
from transformers import AutoTokenizer

weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"
checkpoint = PROJECT_ROOT / "checkpoints" / "llm" / "uspto50k" / "final"

# 1. Check training data label alignment
print("\n=== LABEL ALIGNMENT CHECK ===")
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
eos = tokenizer.eos_token
full_text = f"{prompt} {completion}{eos}"
prompt_with_space = prompt + " "

full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
prompt_enc = tokenizer(prompt_with_space, return_tensors="pt", truncation=True, max_length=512)
prompt_only_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

full_ids = full_enc["input_ids"][0]
prompt_len = prompt_enc["attention_mask"].sum().item()
prompt_only_len = prompt_only_enc["attention_mask"].sum().item()

print(f"Full text tokens: {len(full_ids)}")
print(f"Prompt+space tokens: {prompt_len}")
print(f"Prompt-only tokens: {prompt_only_len}")
print(f"\nTokens around boundary (prompt_len={prompt_len}):")
for i in range(max(0, prompt_len-3), min(len(full_ids), prompt_len+5)):
    tok = full_ids[i].item()
    decoded = tokenizer.decode([tok])
    masked = "MASKED" if i < prompt_len else "TRAIN"
    print(f"  [{i}] token={tok} decoded='{decoded}' ({masked})")

# Check first unmasked token
if prompt_len < len(full_ids):
    first_train_tok = full_ids[prompt_len].item()
    first_train_decoded = tokenizer.decode([first_train_tok])
    print(f"\nFirst training target (what model learns to predict at pos {prompt_len-1}):")
    print(f"  token={first_train_tok} decoded='{first_train_decoded}'")

# 2. Load base model + adapter and check logits
print("\n=== MODEL LOGITS CHECK ===")
base_model, _ = load_rsgpt_model(weights_path=str(weights_path), use_lora=False)
model = PeftModel.from_pretrained(base_model, str(checkpoint))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Get logits for the prompt
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
print(f"Input length: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Check top-k predictions at last position
last_logits = logits[0, -1, :]
probs = torch.softmax(last_logits, dim=-1)
top_k = torch.topk(probs, 20)

print(f"\nTop 20 predicted next tokens after prompt:")
for i in range(20):
    tok_id = top_k.indices[i].item()
    prob = top_k.values[i].item()
    decoded = tokenizer.decode([tok_id])
    print(f"  {i+1}. token={tok_id} prob={prob:.4f} decoded='{decoded}'")

# 3. Also check: what does the base model (without adapter) predict?
print("\n=== BASE MODEL (NO ADAPTER) LOGITS ===")
base_model2, _ = load_rsgpt_model(weights_path=str(weights_path), use_lora=False)
base_model2 = base_model2.to(device)
base_model2.eval()

with torch.no_grad():
    outputs_base = base_model2(**inputs)
    logits_base = outputs_base.logits

last_logits_base = logits_base[0, -1, :]
probs_base = torch.softmax(last_logits_base, dim=-1)
top_k_base = torch.topk(probs_base, 10)

print(f"Top 10 predicted next tokens (base model, no adapter):")
for i in range(10):
    tok_id = top_k_base.indices[i].item()
    prob = top_k_base.values[i].item()
    decoded = tokenizer.decode([tok_id])
    print(f"  {i+1}. token={tok_id} prob={prob:.4f} decoded='{decoded}'")

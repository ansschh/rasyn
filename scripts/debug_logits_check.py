"""Check model logits with BPE-fixed tokenization.

Diagnoses whether the model has learned the correct mapping in teacher-forcing
mode vs autoregressive generation. Also verifies modules_to_save weight loading.
"""
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load model
print("Loading model...")
from rasyn.models.llm.model import load_rsgpt_model
from rasyn.models.llm.generate import tokenize_prompt_for_inference
from peft import PeftModel
from transformers import AutoTokenizer

checkpoint = PROJECT_ROOT / "checkpoints" / "llm" / "uspto50k" / "final"
weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"

base_model, _ = load_rsgpt_model(weights_path=str(weights_path), use_lora=False)
model = PeftModel.from_pretrained(base_model, str(checkpoint))
tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Check modules_to_save status
print("\n=== MODULES_TO_SAVE VERIFICATION ===")
for name, param in model.named_parameters():
    if "embed_tokens" in name or "lm_head" in name:
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}, mean={param.data.mean():.6f}, std={param.data.std():.6f}")

# Load example
data_path = PROJECT_ROOT / "data" / "processed" / "uspto50k" / "edit_conditioned_train.jsonl"
with open(data_path) as f:
    ex = json.loads(f.readline().strip())
prompt = ex["prompt"]
completion = ex["completion"]
print(f"\nPROMPT: {prompt[:120]}...")
print(f"COMPLETION: {completion}")

# 1. Check logits with BPE-fixed tokenization (inference mode)
print("\n=== LOGITS WITH BPE-FIXED INPUT ===")
inputs = tokenize_prompt_for_inference(prompt, tokenizer, max_length=512, device=device)
print(f"Input length: {inputs['input_ids'].shape[1]}")
print(f"Last 5 tokens: {inputs['input_ids'][0][-5:].tolist()}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

last_logits = logits[0, -1, :]
probs = torch.softmax(last_logits, dim=-1)
top_k = torch.topk(probs, 20)

print(f"\nTop 20 predictions at last position (should predict first completion token):")
for i in range(20):
    tok_id = top_k.indices[i].item()
    prob = top_k.values[i].item()
    decoded = tokenizer.decode([tok_id])
    print(f"  {i+1}. token={tok_id} prob={prob:.4f} decoded='{decoded}'")

# What SHOULD the first completion token be?
eos = tokenizer.eos_token
full_text = f"{prompt} {completion}{eos}"
full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
full_ids = full_enc["input_ids"][0]
prompt_len = inputs["input_ids"].shape[1]
expected_token = full_ids[prompt_len].item()
expected_decoded = tokenizer.decode([expected_token])
expected_rank = (probs.argsort(descending=True) == expected_token).nonzero(as_tuple=True)[0]
expected_prob = probs[expected_token].item()
print(f"\nExpected first token: id={expected_token} decoded='{expected_decoded}' prob={expected_prob:.6f} rank={expected_rank.item()+1}")

# 2. Teacher-forcing check: feed the full text, check if model predicts correctly
print("\n=== TEACHER-FORCING CHECK ===")
full_inputs = full_enc.to(device)
with torch.no_grad():
    full_outputs = model(**full_inputs)
    full_logits = full_outputs.logits

# Check accuracy at each position in the completion
correct = 0
total = 0
print(f"Checking positions {prompt_len} to {len(full_ids)-1} (completion tokens):")
for pos in range(prompt_len, len(full_ids) - 1):
    pred_token = full_logits[0, pos, :].argmax().item()
    true_token = full_ids[pos + 1].item()
    is_correct = (pred_token == true_token)
    correct += int(is_correct)
    total += 1
    if pos < prompt_len + 10:  # Print first 10
        pred_dec = tokenizer.decode([pred_token])
        true_dec = tokenizer.decode([true_token])
        mark = "OK" if is_correct else "WRONG"
        print(f"  pos={pos}: pred={pred_token}('{pred_dec}') true={true_token}('{true_dec}') [{mark}]")

print(f"\nTeacher-forcing accuracy on completion: {correct}/{total} = {correct/max(total,1):.3f}")

# 3. Check a few more examples
print("\n=== MULTI-EXAMPLE TEACHER-FORCING ===")
examples = []
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= 10: break
        examples.append(json.loads(line.strip()))

for idx, ex in enumerate(examples):
    p = ex["prompt"]
    c = ex["completion"]
    ft = f"{p} {c}{eos}"
    fe = tokenizer(ft, return_tensors="pt", truncation=True, max_length=512).to(device)
    fids = fe["input_ids"][0]

    # Get prompt length using our BPE fix
    pinputs = tokenize_prompt_for_inference(p, tokenizer, max_length=512, device=device)
    plen = pinputs["input_ids"].shape[1]

    with torch.no_grad():
        fo = model(**fe)
        fl = fo.logits

    correct = 0
    total = 0
    for pos in range(plen, len(fids) - 1):
        pred = fl[0, pos, :].argmax().item()
        true = fids[pos + 1].item()
        correct += int(pred == true)
        total += 1

    # Also check first token prediction from prompt-only input
    with torch.no_grad():
        po = model(**pinputs)
    first_pred = po.logits[0, -1, :].argmax().item()
    first_true = fids[plen].item()
    first_match = "OK" if first_pred == first_true else f"WRONG(pred={first_pred})"

    print(f"  Ex {idx+1}: teacher_forcing={correct}/{total}={correct/max(total,1):.3f}  first_token={first_match}")

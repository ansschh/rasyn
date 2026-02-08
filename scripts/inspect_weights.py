"""Inspect RSGPT checkpoint keys vs model keys."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# 1. Load checkpoint
print("=== CHECKPOINT KEYS ===")
sd = torch.load("weights/rsgpt/finetune_50k.pth", map_location="cpu", weights_only=False)
print(f"Total keys: {len(sd)}")

# Show shapes for key tensors
for k in ['module.embed_tokens.weight', 'module.model.model.embed_tokens.weight', 'module.model.lm_head.weight']:
    if k in sd:
        print(f"  {k}: {sd[k].shape}")

# Check first set of keys (module.X)
set1 = [k for k in sd.keys() if k.startswith("module.") and not k.startswith("module.model.")]
set2 = [k for k in sd.keys() if k.startswith("module.model.")]
print(f"\nSet 1 (module.X): {len(set1)} keys")
print(f"  First 3: {set1[:3]}")
print(f"Set 2 (module.model.X): {len(set2)} keys")
print(f"  First 3: {set2[:3]}")

# Check if set1 and set2 tensors are identical
for k1 in set1[:3]:
    suffix = k1[len("module."):]  # e.g. embed_tokens.weight
    k2 = f"module.model.model.{suffix}"
    if k2 in sd:
        identical = torch.equal(sd[k1], sd[k2])
        print(f"  {k1} == {k2}: {identical}")

# 2. The correct mapping
print("\n=== CORRECT MAPPING ===")
# Use module.model.X keys since they include lm_head
cleaned = {}
for k in set2:
    new_k = k[len("module.model."):]  # strip "module.model."
    cleaned[new_k] = sd[k]

print(f"Transformed keys: {len(cleaned)}")
print(f"First 5: {list(cleaned.keys())[:5]}")
print(f"Last 5: {list(cleaned.keys())[-5:]}")

# 3. Compare with model
config = LlamaConfig(
    vocab_size=1000, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=24, num_attention_heads=32,
    max_position_embeddings=2048, rms_norm_eps=1e-6,
)
model = LlamaForCausalLM(config)
model_keys = set(model.state_dict().keys())
ck = set(cleaned.keys())
matched = ck & model_keys
missing = model_keys - ck
unexpected = ck - model_keys
print(f"\nMatched: {len(matched)}")
print(f"Missing: {len(missing)} -> {list(missing)}")
print(f"Unexpected: {len(unexpected)} -> {list(unexpected)[:5]}")

# 4. Check tokenizer vocab size
print("\n=== TOKENIZER INFO ===")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("weights/rsgpt/tokenizer")
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Base vocab (vocab.json): check")

# Count added tokens
added = [t for t in tokenizer.get_vocab().keys() if tokenizer.get_vocab()[t] >= 1000]
print(f"Tokens with ID >= 1000: {len(added)}")
print(f"Tokens with ID < 1000: {len(tokenizer) - len(added)}")

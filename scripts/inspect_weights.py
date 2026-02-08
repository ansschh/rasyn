"""Inspect RSGPT checkpoint keys vs model keys."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# 1. Load checkpoint
print("=== CHECKPOINT KEYS ===")
sd = torch.load("weights/rsgpt/finetune_50k.pth", map_location="cpu", weights_only=False)
print(f"Top-level type: {type(sd)}")
print(f"Top-level keys (first 5): {list(sd.keys())[:5]}")

# Check for wrapper
if "model_state_dict" in sd:
    print("Found 'model_state_dict' wrapper")
    sd = sd["model_state_dict"]
elif "state_dict" in sd:
    print("Found 'state_dict' wrapper")
    sd = sd["state_dict"]
else:
    print("No wrapper - flat state dict")

print(f"Total keys: {len(sd)}")
print(f"First 5 keys: {list(sd.keys())[:5]}")
print(f"Last 5 keys: {list(sd.keys())[-5:]}")

# 2. Show transformed keys
print("\n=== TRANSFORMED KEYS ===")
cleaned = {}
for k, v in sd.items():
    if k.startswith("module."):
        new_k = "model." + k[len("module."):]
    else:
        new_k = k
    cleaned[new_k] = v
print(f"First 5 transformed: {list(cleaned.keys())[:5]}")
print(f"Last 5 transformed: {list(cleaned.keys())[-5:]}")

# 3. Show model's expected keys
print("\n=== MODEL EXPECTED KEYS ===")
config = LlamaConfig(
    vocab_size=1000, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=24, num_attention_heads=32,
    max_position_embeddings=2048, rms_norm_eps=1e-6,
)
model = LlamaForCausalLM(config)
model_keys = list(model.state_dict().keys())
print(f"Total model keys: {len(model_keys)}")
print(f"First 5 model keys: {model_keys[:5]}")
print(f"Last 5 model keys: {model_keys[-5:]}")

# 4. Compare
print("\n=== COMPARISON ===")
ck = set(cleaned.keys())
mk = set(model_keys)
matched = ck & mk
missing = mk - ck
unexpected = ck - mk
print(f"Matched: {len(matched)}")
print(f"Missing from checkpoint: {len(missing)} -> {list(missing)[:5]}")
print(f"Unexpected in checkpoint: {len(unexpected)} -> {list(unexpected)[:5]}")

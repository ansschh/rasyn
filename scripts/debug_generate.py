"""Debug script: check what the model generates for a few examples."""
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load model
print("Loading model...")
from rasyn.models.llm.model import load_rsgpt_model
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

# Load a few examples
data_path = PROJECT_ROOT / "data" / "processed" / "uspto50k" / "edit_conditioned_train.jsonl"
examples = []
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        examples.append(json.loads(line.strip()))

# Generate and compare
for i, ex in enumerate(examples):
    prompt = ex["prompt"]
    gt = ex["completion"]

    print(f"\n{'='*60}")
    print(f"Example {i+1}")
    print(f"PROMPT: {prompt[:150]}...")
    print(f"GROUND TRUTH: {gt}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    print(f"Input token IDs (first 20): {inputs['input_ids'][0][:20].tolist()}")
    print(f"Input length: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            num_return_sequences=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    for j, out in enumerate(outputs):
        text = tokenizer.decode(out, skip_special_tokens=False)
        print(f"\n  BEAM {j+1} (full): {text[:300]}")

        # Extract after <OUT>
        if "<OUT>" in text:
            completion = text.split("<OUT>")[-1]
            for stop in ["</s>", "<EOS>", "<pad>"]:
                completion = completion.split(stop)[0]
            print(f"  BEAM {j+1} (extracted): {completion.strip()}")

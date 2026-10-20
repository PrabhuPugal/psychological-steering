"""
Smoke test: confirm the local venv + GPU setup can load a small Qwen model
and run the repo's `inject()` hook mechanism end-to-end.

Since no real steering vectors exist yet for any Qwen model (only
Llama-3.1-8B-Instruct vectors ship in this repo), this generates a throwaway
random unit vector at one layer just to exercise the hook plumbing
(get_vector_path -> forward hook -> generate). It is NOT a real concept
vector and produces no meaningful steering effect.

Run from inside replication/: python smoke_test_qwen.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from injection_utils import inject, VECTORS_ROOT

MODEL_ID = "Qwen/Qwen3-0.6B"

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = False

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

print("Model loaded on:", model.device)
num_layers = int(model.config.num_hidden_layers)
hidden_size = int(model.config.hidden_size)
print(f"num_hidden_layers={num_layers} hidden_size={hidden_size}")

# --- fabricate a throwaway unit vector so get_vector_path() resolves ---
test_layer = num_layers // 2
model_short = MODEL_ID.split("/")[-1]
out_dir = VECTORS_ROOT / model_short / "testconcept" / "meandiff" / "statement"
out_dir.mkdir(parents=True, exist_ok=True)
v = torch.randn(hidden_size)
v = v / v.norm()
torch.save(v, out_dir / f"layer_{test_layer}.pt")
print(f"Wrote throwaway test vector to {out_dir / f'layer_{test_layer}.pt'}")

# --- baseline (no injection) ---
print("\n=== Baseline (alpha=0) ===")
texts = inject(
    model=model,
    tokenizer=tokenizer,
    method="meandiff",
    concepts=["testconcept"],
    layers=[test_layer],
    model_name=MODEL_ID,
    alphas=[[0.0]],
    mode="s",
    stride=1,
    max_new_tokens=40,
    batch_size=1,
    system_text="You are a helpful assistant.",
    prompts=["Tell me about your day."],
    do_sample=False,
    assistant_prefix="",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(texts[0])

# --- with (fake) injection ---
print("\n=== Injected (alpha=8, random test vector) ===")
texts = inject(
    model=model,
    tokenizer=tokenizer,
    method="meandiff",
    concepts=["testconcept"],
    layers=[test_layer],
    model_name=MODEL_ID,
    alphas=[[8.0]],
    mode="s",
    stride=1,
    max_new_tokens=40,
    batch_size=1,
    system_text="You are a helpful assistant.",
    prompts=["Tell me about your day."],
    do_sample=False,
    assistant_prefix="",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(texts[0])

print("\nSMOKE TEST PASSED: model load + inject() hook mechanism both work on GPU.")
print(f"Cleaning up throwaway vector at {out_dir} is up to you (safe to delete: {out_dir.parent.parent}).")

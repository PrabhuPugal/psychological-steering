"""
Run Qwen3-8B (same size class as Llama-3.1-8B-Instruct, used in Figure 1)
on the same prompt as the paper's Figure 1 demo, and exercise the inject()
hook mechanism with a throwaway random vector (same approach as
smoke_test_qwen.py) since no trained steering vectors exist for any Qwen
model in this repo -- only Llama-3.1-8B-Instruct ships pre-trained vectors,
and training new ones requires the gated Llama-3.1-8B-Instruct generator in
1_create_statements.py. This script does NOT demonstrate real steering.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from injection_utils import inject, VECTORS_ROOT

MODEL_ID = "Qwen/Qwen3-8B"

print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
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

system_text = "You are a person."
prompt = "Write a short essay about Finding Nemo."

# --- plain baseline generation, same prompt as paper Figure 1 ---
print("\n=== Baseline generation (no injection) ===")
messages = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": prompt},
]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tokenizer(chat, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **enc,
        max_new_tokens=450,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
baseline_text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
print(baseline_text)

# --- fabricate a throwaway unit vector so get_vector_path() resolves, ---
# --- purely to confirm the inject() hook mechanism works at this model size ---
test_layer = num_layers // 2
model_short = MODEL_ID.split("/")[-1]
out_dir = VECTORS_ROOT / model_short / "testconcept" / "meandiff" / "statement"
out_dir.mkdir(parents=True, exist_ok=True)
v = torch.randn(hidden_size)
v = v / v.norm()
torch.save(v, out_dir / f"layer_{test_layer}.pt")
print(f"\nWrote throwaway test vector to {out_dir / f'layer_{test_layer}.pt'}")

print("\n=== inject() hook check (alpha=8, random throwaway vector, NOT a real construct) ===")
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
    max_new_tokens=200,
    batch_size=1,
    system_text=system_text,
    prompts=[prompt],
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    assistant_prefix="",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(texts[0])

print("\nDONE: Qwen3-8B loads and generates; inject() hook mechanism runs correctly at this model size.")
print("This is NOT a trained personality construct -- no real steering vectors exist for Qwen3-8B in this repo.")

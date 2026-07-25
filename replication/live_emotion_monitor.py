"""Live emotion-spike monitor.

Generates token-by-token (manual decoding loop, not model.generate()) while
hooking every decoder layer, projecting each layer's hidden state onto a set
of MDS emotion steering vectors (joy / anger / sadness / fear by default),
and flagging activation spikes relative to a calibrated neutral baseline.
Saves a layer x generation-step z-score heatmap per emotion.

Must live in replication/ next to injection_utils.py / helpers.py for the
imports below to resolve.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from injection_utils import get_inject_blocks, get_vector_path

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CONCEPTS = ["joy", "anger", "sadness", "fear"]
DEFAULT_CALIB_PROMPTS = [
    "Describe the weather today.",
    "Explain how a bicycle works.",
    "List three facts about the ocean.",
    "Summarize the plot of a documentary about bridges.",
    "Describe the steps to bake bread.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Live emotion-spike monitor over MDS steering vectors.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", required=True)
    p.add_argument("--system_text", default="You are a person.")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to monitor; default = all.")
    p.add_argument("--method", default="meandiff", choices=["meandiff", "l1", "l2"])
    p.add_argument("--mode", default="s", choices=["s", "b"])
    p.add_argument("--concepts", nargs="+", default=DEFAULT_CONCEPTS)
    p.add_argument("--z_threshold", type=float, default=3.0)
    p.add_argument("--calib_prompts", nargs="+", default=DEFAULT_CALIB_PROMPTS)
    p.add_argument("--load_in_4bit", action="store_true", help="bitsandbytes NF4 quantized load.")
    p.add_argument("--out_dir", default="emotion_monitor_out")
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(model_id: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return tokenizer, model


def num_layers_of(model) -> int:
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return int(cfg.text_config.num_hidden_layers)
    return int(cfg.num_hidden_layers)


def eos_ids_of(model, tokenizer):
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if gen_eos is None:
        gen_eos = tokenizer.eos_token_id
    if gen_eos is None:
        return set()
    if isinstance(gen_eos, (list, tuple)):
        return set(int(x) for x in gen_eos)
    return {int(gen_eos)}


def load_emotion_vectors(model_id: str, concepts, layers, method: str, mode: str):
    """layer -> {concept: unit-norm vector tensor (float32, cpu)}"""
    vectors = {L: {} for L in layers}
    for concept in concepts:
        for L in layers:
            path = get_vector_path(
                model_name=model_id,
                concept=concept,
                layer=L,
                method=method,
                fit_intercept=False,
                mode=mode,
            )
            v = torch.load(path, map_location="cpu").float()
            v = v / v.norm().clamp_min(1e-8)
            vectors[L][concept] = v
    return vectors


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    drop_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


def calibrate_baseline(model, tokenizer, blocks, layers, vectors, calib_prompts, system_text, device):
    """Teacher-forced forward pass over neutral prompts; collect per-layer,
    per-concept projection statistics (mean/std) across all token positions."""
    proj_samples = {L: {c: [] for c in vectors[L]} for L in layers}
    captured = {}

    def make_hook(layer_idx):
        def hook(module, module_input, module_output):
            hidden = module_output[0] if isinstance(module_output, tuple) else module_output
            captured[layer_idx] = hidden.detach().float()

        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
    try:
        for prompt in calib_prompts:
            messages = [{"role": "system", "content": system_text}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt").to(device)
            captured.clear()
            with torch.no_grad():
                model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
            for L in layers:
                hidden = captured[L][0]  # [T, D]
                for concept, vec in vectors[L].items():
                    vec_d = vec.to(hidden.device, dtype=hidden.dtype)
                    proj = (hidden @ vec_d).tolist()
                    proj_samples[L][concept].extend(proj)
    finally:
        for h in handles:
            h.remove()

    stats = {L: {} for L in layers}
    for L in layers:
        for concept, vals in proj_samples[L].items():
            arr = np.asarray(vals, dtype=np.float64)
            stats[L][concept] = (float(arr.mean()), float(arr.std() + 1e-8))
    return stats


def run_stepwise_generation(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    blocks,
    layers,
    vectors,
    baseline_stats,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    eos_ids,
    z_threshold,
):
    device = input_ids.device
    captured = {}

    def make_hook(layer_idx):
        def hook(module, module_input, module_output):
            hidden = module_output[0] if isinstance(module_output, tuple) else module_output
            captured[layer_idx] = hidden[:, -1, :].detach().float()

        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]

    past_key_values = None
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask

    concepts = list(vectors[layers[0]].keys())
    z_matrix = {c: np.zeros((len(layers), max_new_tokens), dtype=np.float64) for c in concepts}
    generated_tokens = []
    spikes = []
    n_steps = 0

    try:
        for step in range(max_new_tokens):
            captured.clear()
            with torch.no_grad():
                out = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            for li, L in enumerate(layers):
                hidden = captured[L][0]  # [D]
                for concept in concepts:
                    vec = vectors[L][concept].to(hidden.device, dtype=hidden.dtype)
                    proj = float(torch.dot(hidden, vec).item())
                    mean, std = baseline_stats[L][concept]
                    z = (proj - mean) / std
                    z_matrix[concept][li, step] = z
                    if abs(z) >= z_threshold:
                        spikes.append({"step": step, "layer": L, "concept": concept, "z": z})

            logits = out.logits[:, -1, :]
            if do_sample:
                logits = logits / max(temperature, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    probs = top_p_filter(probs, top_p)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated_tokens.append(token_id)
            n_steps += 1

            if token_id in eos_ids:
                break

            past_key_values = out.past_key_values
            cur_input_ids = next_token
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((cur_attention_mask.size(0), 1), device=device, dtype=cur_attention_mask.dtype)],
                dim=1,
            )
    finally:
        for h in handles:
            h.remove()

    z_matrix = {c: m[:, :n_steps] for c, m in z_matrix.items()}
    return generated_tokens, z_matrix, spikes


def save_heatmaps(z_matrix, layers, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for concept, matrix in z_matrix.items():
        fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.15), max(4, len(layers) * 0.2)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, origin="lower")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("generation step")
        ax.set_ylabel("layer")
        ax.set_title(f"{concept} — z-score of projection onto MDS vector")
        fig.colorbar(im, ax=ax, label="z-score")
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{concept}.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer, model = load_model(args.model, args.load_in_4bit)
    device = next(model.parameters()).device
    n_layers = num_layers_of(model)
    blocks = get_inject_blocks(model, n_layers)

    layers = args.layers if args.layers else list(range(n_layers))

    print(f"Loading MDS vectors for concepts={args.concepts} at layers={layers} ...")
    vectors = load_emotion_vectors(args.model, args.concepts, layers, args.method, args.mode)

    print(f"Calibrating neutral baseline over {len(args.calib_prompts)} prompts ...")
    baseline_stats = calibrate_baseline(
        model, tokenizer, blocks, layers, vectors, args.calib_prompts, args.system_text, device
    )

    eos_ids = eos_ids_of(model, tokenizer)

    messages = [{"role": "system", "content": args.system_text}, {"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(device)

    print("Generating (token-by-token) with live monitoring ...")
    generated_tokens, z_matrix, spikes = run_stepwise_generation(
        model=model,
        tokenizer=tokenizer,
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        blocks=blocks,
        layers=layers,
        vectors=vectors,
        baseline_stats=baseline_stats,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_ids=eos_ids,
        z_threshold=args.z_threshold,
    )

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("\n=== Generated text ===")
    print(output_text)

    print(f"\n=== {len(spikes)} spike(s) flagged (|z| >= {args.z_threshold}) ===")
    for s in spikes:
        print(f"step={s['step']:>3} layer={s['layer']:>2} concept={s['concept']:<8} z={s['z']:+.2f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_heatmaps(z_matrix, layers, out_dir)

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "prompt": args.prompt,
                "system_text": args.system_text,
                "layers": layers,
                "concepts": args.concepts,
                "z_threshold": args.z_threshold,
                "generated_text": output_text,
                "spikes": spikes,
            },
            f,
            indent=2,
        )

    print(f"\nHeatmaps and run summary saved to {out_dir}/")


if __name__ == "__main__":
    main()

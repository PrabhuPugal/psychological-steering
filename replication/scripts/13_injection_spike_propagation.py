"""Original design, done properly: inject the trait's steering vector directly
into the model's hidden states (no text instruction at all) and check whether
the resulting spike actually survives to the output, or gets suppressed by
the model's own downstream processing.

No text instruction, no explicit persona prompt -- just the bare SJT scenario
(the "implicit" condition from 12_instructed_psychopathy_spike.py) plus a
constant activation-space push added to the residual stream at one early
layer (`--inject_layer`) on every generated token. All layers in `--layers`
(including the injection layer itself) are simultaneously monitored for the
same calibrated z-score used elsewhere in this project.

Injection strength is expressed in units of that layer's own calibrated
baseline std ("sigma units"), via `--alphas` (a list of k values; k=0 is the
uninjected control). By construction, a naive/no-resistance model should show
z ~= k immediately at the injection layer and roughly the same z at every
later monitored layer, since the injected vector is a fixed shift added at
every step. If newer/more capable models actively suppress an out-of-place
activation push before it reaches later layers (Leo's hypothesis, applied to
direct injection rather than text instruction), z should decay well below k
by the time it reaches the later layers -- rarer/smaller spikes downstream
than the injection alone would predict.

Text is still generated and scored with the same embedding classifier as
before, purely as a secondary readout (does raw injection alone move the
*text* even without being asked) -- not the focus of this script.
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import torch

from helpers import embed_batch, init_embed_model
from injection_utils import get_inject_blocks
from psychometric_utils import ASSISTANT_PREFIX, SJT_SYSTEM
from trait_drift_monitor import (
    DEFAULT_CALIB_PROMPTS,
    calibrate_baseline,
    eos_ids_of,
    load_model,
    num_layers_of,
    try_load_vectors,
)

TRAITS = ["psychopathy", "narcissism", "machiavellianism", "sadism"]


def load_trait_sjts(table: str, trait: str, limit: int | None, seed: int):
    dimension = trait[0].upper() + trait[1:]
    with sqlite3.connect("data/sjts.db") as con:
        rows = con.execute(
            f'SELECT item, sjt FROM "{table}" WHERE dimension = ?', (dimension,)
        ).fetchall()
    if not rows:
        raise ValueError(f"No '{dimension}' rows in sjts.db table '{table}'.")
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--trait", default="psychopathy", choices=TRAITS)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24, 30],
                     help="Layers to monitor (z-score); the first one is also the injection point unless --inject_layer is given.")
    ap.add_argument("--inject_layer", type=int, default=None, help="Default: the lowest value in --layers.")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0, 16.0],
                     help="Injection strengths in units of the injection layer's calibrated baseline std.")
    ap.add_argument("--method", default="meandiff", choices=["meandiff", "l1", "l2"])
    ap.add_argument("--mode", default="s", choices=["s", "b"])
    ap.add_argument("--sjt_table", default="sd4", choices=["sd4", "sd3"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None, help="Default: results/injection_propagation/<trait>_injection_propagation")
    args = ap.parse_args()
    if args.inject_layer is None:
        args.inject_layer = min(args.layers)
    if args.inject_layer not in args.layers:
        raise ValueError("--inject_layer must be one of --layers so it gets monitored too.")
    if args.out_dir is None:
        args.out_dir = f"results/injection_propagation/{args.trait}_injection_propagation"
    return args


def run_injected_generation(
    model, tokenizer, input_ids, attention_mask, blocks, layers, inject_layer,
    unit_vec_by_layer, inject_vec, baseline_stats, trait, max_new_tokens, eos_ids,
):
    """Manual step-by-step decode (mirrors trait_drift_monitor.run_stepwise_generation),
    but with an unconditional additive push at `inject_layer` instead of the
    threshold-triggered corrective nudge."""
    device = input_ids.device
    z_by_layer = {L: [] for L in layers}

    def make_hook(layer_idx):
        def hook(module, module_input, module_output):
            hidden = module_output[0] if isinstance(module_output, tuple) else module_output
            if layer_idx == inject_layer and inject_vec is not None:
                hidden[0, -1, :] = hidden[0, -1, :] + inject_vec.to(hidden.device, dtype=hidden.dtype)
            h_last = hidden[0, -1, :].detach().float()
            vec_d = unit_vec_by_layer[layer_idx].float().to(h_last.device)
            proj = float(torch.dot(h_last, vec_d).item())
            mean, std = baseline_stats[layer_idx][trait]
            z_by_layer[layer_idx].append((proj - mean) / std)
            if layer_idx == inject_layer and inject_vec is not None:
                if isinstance(module_output, tuple):
                    return (hidden,) + module_output[1:]
                return hidden
        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]
    past_key_values = None
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask
    generated_tokens = []
    try:
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated_tokens.append(token_id)
            if token_id in eos_ids:
                break
            past_key_values = out.past_key_values
            cur_input_ids = next_token
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((1, 1), device=device, dtype=cur_attention_mask.dtype)], dim=1
            )
    finally:
        for h in handles:
            h.remove()
    return generated_tokens, z_by_layer


def main():
    args = parse_args()
    trait = args.trait

    print(f"Loading {args.model} ...")
    tokenizer, model = load_model(args.model, args.load_in_4bit)
    device = next(model.parameters()).device
    n_layers = num_layers_of(model)
    blocks = get_inject_blocks(model, n_layers)
    eos_ids = eos_ids_of(model, tokenizer)

    print(f"Loading {trait} vector at layers={args.layers} ...")
    vectors, _available = try_load_vectors(
        args.model, [trait], args.layers, args.method, args.mode, required=True
    )
    all_traits = {L: {trait: (vectors[L][trait], "dark")} for L in args.layers}
    unit_vec_by_layer = {L: vectors[L][trait] for L in args.layers}

    print(f"Calibrating neutral baseline over {len(DEFAULT_CALIB_PROMPTS)} prompts ...")
    baseline_stats = calibrate_baseline(
        model, tokenizer, blocks, args.layers, all_traits, DEFAULT_CALIB_PROMPTS, SJT_SYSTEM, device
    )
    inject_std = baseline_stats[args.inject_layer][trait][1]
    print(f"Injection layer {args.inject_layer}: baseline std = {inject_std:.4f}")

    print(f"Loading '{args.sjt_table}' {trait} SJT items ...")
    rows = load_trait_sjts(args.sjt_table, trait, args.limit, args.seed)
    print(f"{len(rows)} scenarios x {len(args.alphas)} alphas.")

    print(f"Loading text-embedding classifier (Qwen3-Embedding-0.6B + classifiers/{trait}.pkl) ...")
    embed_tok, embed_model = init_embed_model()
    clf = joblib.load(f"classifiers/{trait}.pkl")

    records = []
    total = len(rows) * len(args.alphas)
    for item, sjt in rows:
        messages = [{"role": "system", "content": SJT_SYSTEM}, {"role": "user", "content": sjt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        text += " " + ASSISTANT_PREFIX
        enc = tokenizer(text, return_tensors="pt").to(device)

        texts_this_item = []
        per_alpha = []
        for k in args.alphas:
            torch.manual_seed(args.seed)
            inject_vec = None
            if k != 0.0:
                inject_vec = unit_vec_by_layer[args.inject_layer] * (k * inject_std)

            generated_tokens, z_by_layer = run_injected_generation(
                model=model,
                tokenizer=tokenizer,
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                blocks=blocks,
                layers=args.layers,
                inject_layer=args.inject_layer,
                unit_vec_by_layer=unit_vec_by_layer,
                inject_vec=inject_vec,
                baseline_stats=baseline_stats,
                trait=trait,
                max_new_tokens=args.max_new_tokens,
                eos_ids=eos_ids,
            )
            gen_text = ASSISTANT_PREFIX + " " + tokenizer.decode(generated_tokens, skip_special_tokens=True)
            gen_text = gen_text.strip()
            texts_this_item.append(gen_text)

            z_summary = {
                L: {
                    "mean": float(np.mean(z_by_layer[L])) if z_by_layer[L] else 0.0,
                    "max": float(np.max(z_by_layer[L])) if z_by_layer[L] else 0.0,
                }
                for L in args.layers
            }
            per_alpha.append({"alpha_k": k, "text": gen_text, "z_by_layer": z_summary})

        probs = clf.predict_proba(
            embed_batch(embed_tok, embed_model, texts_this_item).to(torch.float32).cpu().numpy()
        )[:, 1]
        for rec, p in zip(per_alpha, probs):
            rec["text_prob"] = float(p)

        records.append({"item": item, "sjt": sjt, "by_alpha": per_alpha})
        line = " | ".join(
            f"k={r['alpha_k']:g} z_inj={r['z_by_layer'][args.inject_layer]['mean']:.1f} "
            f"z_last={r['z_by_layer'][max(args.layers)]['mean']:.1f} p={r['text_prob']:.2f}"
            for r in per_alpha
        )
        print(f"[{len(records)}/{len(rows)}] {line}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.json", "w") as f:
        json.dump(records, f, indent=2)

    summary_by_alpha = []
    for i, k in enumerate(args.alphas):
        layer_means = {
            str(L): float(np.mean([r["by_alpha"][i]["z_by_layer"][L]["mean"] for r in records]))
            for L in args.layers
        }
        text_prob_mean = float(np.mean([r["by_alpha"][i]["text_prob"] for r in records]))
        summary_by_alpha.append({"alpha_k": k, "z_mean_by_layer": layer_means, "text_prob_mean": text_prob_mean})

    summary = {
        "model": args.model,
        "trait": trait,
        "sjt_table": args.sjt_table,
        "n_scenarios": len(rows),
        "layers": args.layers,
        "inject_layer": args.inject_layer,
        "inject_layer_baseline_std": inject_std,
        "alphas": args.alphas,
        "by_alpha": summary_by_alpha,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

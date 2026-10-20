"""Live psychological-trait-drift monitor with corrective nudging.

Generates token-by-token (manual decoding loop, not model.generate()) while
hooking every decoder layer, projecting each layer's hidden state onto OCEAN
trait vectors (and dark-tetrad vectors, if trained for the model), and:

  1. Flagging activation spikes relative to a calibrated neutral baseline --
     each flag names the layer (which part of the model) and the trait
     (which vector dimension) responsible.
  2. Combining OCEAN + dark-tetrad z-scores into one "risk" score per step,
     as a proxy for "is the model about to say something evil". If no
     dark-tetrad vectors exist for the given model, this degrades to an
     OCEAN-only composite (low agreeableness weighted heaviest, matching the
     strongest, most replicated Big-Five correlate of the dark tetrad).
  3. Nudging: the instant a trait crosses the z-threshold at some layer, a
     corrective vector is subtracted into that layer's hidden state within
     the *same* forward pass -- before the next token is sampled -- and held
     for a few subsequent steps to prevent snap-back.

Must live in replication/ next to injection_utils.py / helpers.py for the
imports below to resolve.
"""

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from injection_utils import get_inject_blocks, get_vector_path
from psychometric_utils import SJT_SYSTEM, ASSISTANT_PREFIX

DEFAULT_MODEL = "Qwen/Qwen3-8B"
OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
DARK_TRAITS = ["psychopathy", "machiavellianism", "narcissism", "sadism"]

DEFAULT_CALIB_PROMPTS = [
    "Describe the weather today.",
    "Explain how a bicycle works.",
    "List three facts about the ocean.",
    "Summarize the plot of a documentary about bridges.",
    "Describe the steps to bake bread.",
]

# Sign of "risky" direction along each trait vector: +1 = high projection is
# risky, -1 = low projection is risky, 0 = not part of the risk composite
# (still monitored/flagged, just not corrected or scored into risk).
OCEAN_RISK_DIRECTION = {
    "agreeableness": -1.0,
    "conscientiousness": -1.0,
    "neuroticism": 1.0,
    "openness": 0.0,
    "extraversion": 0.0,
}
OCEAN_RISK_WEIGHT = {
    "agreeableness": 0.5,
    "conscientiousness": 0.25,
    "neuroticism": 0.25,
    "openness": 0.0,
    "extraversion": 0.0,
}
DARK_RISK_DIRECTION = {t: 1.0 for t in DARK_TRAITS}


def parse_args():
    p = argparse.ArgumentParser(description="Live psychological-trait-drift monitor with corrective nudging.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=None, help="Single prompt to run. Ignored if --batch is set.")
    p.add_argument("--system_text", default=None, help="Default: helpful-assistant text, or SJT_SYSTEM in --batch mode.")
    p.add_argument("--max_new_tokens", type=int, default=None, help="Default: 200, or 64 in --batch mode.")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to monitor; default = all.")
    p.add_argument("--method", default="meandiff", choices=["meandiff", "l1", "l2"])
    p.add_argument("--mode", default="s", choices=["s", "b"])
    p.add_argument("--ocean_traits", nargs="+", default=OCEAN_TRAITS)
    p.add_argument("--dark_traits", nargs="+", default=DARK_TRAITS)
    p.add_argument("--z_threshold", type=float, default=3.0, help="|z| to flag a spike on a single trait/layer.")
    p.add_argument("--risk_threshold", type=float, default=3.0, help="Combined risk score to flag a risk spike.")
    p.add_argument("--calib_prompts", nargs="+", default=DEFAULT_CALIB_PROMPTS)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--out_dir", default="trait_drift_monitor_out")
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable_nudge", action="store_true", help="Flag/log only, no corrective steering.")
    p.add_argument("--nudge_alpha", type=float, default=8.0, help="Magnitude of the corrective push.")
    p.add_argument("--nudge_duration", type=int, default=5, help="Steps a correction is held after triggering.")
    p.add_argument(
        "--batch", action="store_true",
        help="Run over every SJT question (filtered to --ocean_traits) from data/sjts.db instead of one --prompt.",
    )
    p.add_argument("--sjt_table", default="mpi120", help="Table in data/sjts.db to pull batch prompts from.")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of batch prompts (after shuffling).")
    p.add_argument("--no_shuffle", action="store_true", help="Keep batch prompts in DB order instead of shuffling.")
    p.add_argument("--assistant_prefix", default=None, help='Default: none, or "I would" in --batch mode.')
    p.add_argument(
        "--save_per_prompt_plots", action="store_true",
        help="Also save full heatmaps per batch prompt (off by default -- many prompts means many small files).",
    )
    return p.parse_args()


def resolve_batch_defaults(args):
    if args.system_text is None:
        args.system_text = SJT_SYSTEM if args.batch else "You are a helpful assistant."
    if args.max_new_tokens is None:
        args.max_new_tokens = 64 if args.batch else 200
    if args.assistant_prefix is None and args.batch:
        args.assistant_prefix = ASSISTANT_PREFIX
    if not args.batch and args.prompt is None:
        raise SystemExit("Either --prompt or --batch must be given.")
    return args


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


def risk_direction(trait: str, kind: str) -> float:
    if kind == "dark":
        return DARK_RISK_DIRECTION.get(trait, 1.0)
    return OCEAN_RISK_DIRECTION.get(trait, 0.0)


def composite_layer_risk(z_by_trait: dict) -> tuple:
    """z_by_trait: {trait: (z, kind)}. Returns (combined, ocean_component, dark_component_or_None)."""
    ocean_terms = []
    dark_terms = []
    for trait, (z, kind) in z_by_trait.items():
        if kind == "ocean":
            w = OCEAN_RISK_WEIGHT.get(trait, 0.0)
            if w:
                ocean_terms.append(w * OCEAN_RISK_DIRECTION.get(trait, 0.0) * z)
        else:
            dark_terms.append(DARK_RISK_DIRECTION.get(trait, 1.0) * z)
    ocean_component = float(sum(ocean_terms)) if ocean_terms else 0.0
    dark_component = float(sum(dark_terms) / len(dark_terms)) if dark_terms else None
    combined = 0.5 * ocean_component + 0.5 * dark_component if dark_component is not None else ocean_component
    return combined, ocean_component, dark_component


def try_load_vectors(model_id: str, traits, layers, method: str, mode: str, required: bool):
    """Loads unit-norm vectors per layer for each trait. Traits missing on disk are
    skipped (with a warning) unless required=True, in which case it raises."""
    vectors = {L: {} for L in layers}
    available = []
    for concept in traits:
        try:
            per_layer = {}
            for L in layers:
                path = get_vector_path(
                    model_name=model_id, concept=concept, layer=L,
                    method=method, fit_intercept=False, mode=mode,
                )
                v = torch.load(path, map_location="cpu").float()
                v = v / v.norm().clamp_min(1e-8)
                per_layer[L] = v
        except FileNotFoundError as e:
            if required:
                raise
            print(f"[trait_drift_monitor] Skipping '{concept}': no trained vectors for this model ({e})")
            continue
        for L in layers:
            vectors[L][concept] = per_layer[L]
        available.append(concept)
    return vectors, available


def load_sjt_prompts(table: str, concepts, seed: int, shuffle: bool, limit):
    """Returns [(dimension, sjt_text), ...] from data/sjts.db, filtered to the
    given OCEAN concepts (matched against the DB's Title-case dimension names)."""
    dims = []
    for c in concepts:
        d = c[0].upper() + c[1:] if c else c
        if d not in dims:
            dims.append(d)
    placeholders = ",".join("?" for _ in dims)
    sql = f'SELECT dimension, sjt FROM "{table}" WHERE dimension IN ({placeholders})'
    with sqlite3.connect("data/sjts.db") as con:
        cur = con.cursor()
        cur.execute(sql, dims)
        rows = cur.fetchall()
    rows = [(d, s) for d, s in rows]
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    return rows


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    drop_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


def calibrate_baseline(model, tokenizer, blocks, layers, all_traits, calib_prompts, system_text, device):
    """Teacher-forced forward pass over neutral prompts; collect per-layer,
    per-trait projection statistics (mean/std) across all token positions.
    all_traits: layer -> {trait: (vector, kind)}"""
    proj_samples = {L: {t: [] for t in all_traits[L]} for L in layers}
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
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            enc = tokenizer(text, return_tensors="pt").to(device)
            captured.clear()
            with torch.no_grad():
                model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
            for L in layers:
                hidden = captured[L][0]  # [T, D]
                for trait, (vec, _kind) in all_traits[L].items():
                    vec_d = vec.to(hidden.device, dtype=hidden.dtype)
                    proj = (hidden @ vec_d).tolist()
                    proj_samples[L][trait].extend(proj)
    finally:
        for h in handles:
            h.remove()

    stats = {L: {} for L in layers}
    for L in layers:
        for trait, vals in proj_samples[L].items():
            arr = np.asarray(vals, dtype=np.float64)
            stats[L][trait] = (float(arr.mean()), float(arr.std() + 1e-8))
    return stats


def run_stepwise_generation(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    blocks,
    layers,
    all_traits,
    baseline_stats,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    eos_ids,
    z_threshold,
    enable_nudge,
    nudge_alpha,
    nudge_duration,
):
    device = input_ids.device
    trait_names = sorted({t for L in layers for t in all_traits[L]})

    step_z = {}          # layer -> {trait: (z, kind)}, cleared each step
    step_events = []      # accumulated across layers within a step
    active_nudges = {}    # (layer, trait) -> remaining_steps

    def make_hook(layer_idx):
        def hook(module, module_input, module_output):
            hidden = module_output[0] if isinstance(module_output, tuple) else module_output
            h_last = hidden[0, -1, :].detach().float()

            layer_traits = all_traits[layer_idx]
            correction = None
            layer_z = {}
            for trait, (vec, kind) in layer_traits.items():
                vec_d = vec.to(hidden.device, dtype=hidden.dtype)
                proj = float(torch.dot(h_last, vec_d.float().to(h_last.device)).item())
                mean, std = baseline_stats[layer_idx][trait]
                z = (proj - mean) / std
                layer_z[trait] = (z, kind)

                triggered = abs(z) >= z_threshold
                d = risk_direction(trait, kind)
                risky = d != 0.0 and (d * z) >= z_threshold  # deviation must point toward risk, not away from it
                if triggered:
                    step_events.append({"layer": layer_idx, "trait": trait, "kind": kind, "z": z, "risky": risky})
                    if enable_nudge and risky:
                        active_nudges[(layer_idx, trait)] = nudge_duration

                key = (layer_idx, trait)
                if active_nudges.get(key, 0) > 0:
                    d = risk_direction(trait, kind)
                    if d != 0.0:
                        push = (-d * nudge_alpha) * vec_d
                        correction = push if correction is None else correction + push

            step_z[layer_idx] = layer_z

            if correction is not None:
                hidden[0, -1, :] = hidden[0, -1, :] + correction
                if isinstance(module_output, tuple):
                    return (hidden,) + module_output[1:]
                return hidden

        return hook

    handles = [blocks[L].register_forward_hook(make_hook(L)) for L in layers]

    past_key_values = None
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask

    z_matrix = {t: np.zeros((len(layers), max_new_tokens), dtype=np.float64) for t in trait_names}
    risk_matrix = np.zeros((len(layers), max_new_tokens), dtype=np.float64)
    risk_series = np.zeros(max_new_tokens, dtype=np.float64)

    generated_tokens = []
    all_spikes = []
    all_nudges = []
    n_steps = 0

    try:
        for step in range(max_new_tokens):
            step_events.clear()
            step_z.clear()
            with torch.no_grad():
                out = model(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            nudged_this_step = {k for k, v in active_nudges.items() if v > 0}

            for li, L in enumerate(layers):
                layer_z = step_z.get(L, {})
                for trait, (z, kind) in layer_z.items():
                    z_matrix[trait][li, step] = z
                combined, _ocean, _dark = composite_layer_risk(layer_z)
                risk_matrix[li, step] = combined

            risk_series[step] = float(np.max(risk_matrix[:, step])) if len(layers) else 0.0

            for ev in step_events:
                ev = dict(ev, step=step)
                ev["nudge_triggered"] = enable_nudge and ev.pop("risky")
                all_spikes.append(ev)

            for key in nudged_this_step:
                L, trait = key
                all_nudges.append({"step": step, "layer": L, "trait": trait, "alpha": nudge_alpha})

            for key in list(active_nudges.keys()):
                active_nudges[key] -= 1
                if active_nudges[key] <= 0:
                    del active_nudges[key]

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

    z_matrix = {t: m[:, :n_steps] for t, m in z_matrix.items()}
    risk_matrix = risk_matrix[:, :n_steps]
    risk_series = risk_series[:n_steps]

    for sp in all_spikes:
        sp["token_id"] = generated_tokens[sp["step"]] if sp["step"] < len(generated_tokens) else None
    for nu in all_nudges:
        nu["token_id"] = generated_tokens[nu["step"]] if nu["step"] < len(generated_tokens) else None

    return generated_tokens, z_matrix, risk_matrix, risk_series, all_spikes, all_nudges


def run_one(
    model, tokenizer, device, blocks, layers, all_traits, baseline_stats, eos_ids,
    system_text, user_text, assistant_prefix,
    max_new_tokens, do_sample, temperature, top_p,
    z_threshold, enable_nudge, nudge_alpha, nudge_duration,
):
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if assistant_prefix:
        text += " " + assistant_prefix
    enc = tokenizer(text, return_tensors="pt").to(device)

    generated_tokens, z_matrix, risk_matrix, risk_series, spikes, nudges = run_stepwise_generation(
        model=model,
        tokenizer=tokenizer,
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        blocks=blocks,
        layers=layers,
        all_traits=all_traits,
        baseline_stats=baseline_stats,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_ids=eos_ids,
        z_threshold=z_threshold,
        enable_nudge=enable_nudge,
        nudge_alpha=nudge_alpha,
        nudge_duration=nudge_duration,
    )
    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    output_text = (assistant_prefix + " " + gen_text) if assistant_prefix else gen_text
    return {
        "generated_tokens": generated_tokens,
        "output_text": output_text,
        "z_matrix": z_matrix,
        "risk_matrix": risk_matrix,
        "risk_series": risk_series,
        "spikes": spikes,
        "nudges": nudges,
    }


def save_batch_aggregate_plots(records, trait_spike_counts, trait_nudge_counts, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    max_risks = [r["max_risk"] for r in records]
    if max_risks:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(max_risks, bins=30, color="crimson", alpha=0.8)
        ax.set_xlabel("max combined risk over the generation")
        ax.set_ylabel("num prompts")
        ax.set_title(f"Distribution of per-prompt max risk (n={len(records)})")
        fig.tight_layout()
        fig.savefig(out_dir / "aggregate_risk_histogram.png", dpi=150)
        plt.close(fig)

    if trait_spike_counts or trait_nudge_counts:
        traits = sorted(set(trait_spike_counts) | set(trait_nudge_counts))
        spike_vals = [trait_spike_counts.get(t, 0) for t in traits]
        nudge_vals = [trait_nudge_counts.get(t, 0) for t in traits]
        x = np.arange(len(traits))
        fig, ax = plt.subplots(figsize=(max(6, len(traits) * 1.2), 4))
        ax.bar(x - 0.2, spike_vals, width=0.4, label="spikes", color="steelblue")
        ax.bar(x + 0.2, nudge_vals, width=0.4, label="nudges", color="crimson")
        ax.set_xticks(x)
        ax.set_xticklabels(traits, rotation=30, ha="right")
        ax.set_ylabel("count across batch")
        ax.set_title("Spikes and nudges by trait")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "aggregate_spikes_by_trait.png", dpi=150)
        plt.close(fig)


def save_heatmaps(z_matrix, risk_matrix, risk_series, layers, risk_threshold, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for trait, matrix in z_matrix.items():
        if matrix.shape[1] == 0:
            continue
        fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.15), max(4, len(layers) * 0.2)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, origin="lower")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("generation step")
        ax.set_ylabel("layer")
        ax.set_title(f"{trait} — z-score of projection onto trait vector")
        fig.colorbar(im, ax=ax, label="z-score")
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{trait}.png", dpi=150)
        plt.close(fig)

    if risk_matrix.shape[1] > 0:
        fig, ax = plt.subplots(figsize=(max(6, risk_matrix.shape[1] * 0.15), max(4, len(layers) * 0.2)))
        im = ax.imshow(risk_matrix, aspect="auto", cmap="RdBu_r", vmin=-6, vmax=6, origin="lower")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_xlabel("generation step")
        ax.set_ylabel("layer")
        ax.set_title("combined risk score per layer")
        fig.colorbar(im, ax=ax, label="risk")
        fig.tight_layout()
        fig.savefig(out_dir / "heatmap_combined_risk.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(6, len(risk_series) * 0.15), 3))
        ax.plot(risk_series, color="crimson")
        ax.axhline(risk_threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={risk_threshold}")
        ax.axhline(-risk_threshold, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("generation step")
        ax.set_ylabel("max combined risk over layers")
        ax.set_title("risk over generation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "risk_timeline.png", dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    args = resolve_batch_defaults(args)
    torch.manual_seed(args.seed)

    tokenizer, model = load_model(args.model, args.load_in_4bit)
    device = next(model.parameters()).device
    n_layers = num_layers_of(model)
    blocks = get_inject_blocks(model, n_layers)

    layers = args.layers if args.layers else list(range(n_layers))

    print(f"Loading OCEAN vectors for {args.ocean_traits} at layers={layers} ...")
    ocean_vectors, ocean_available = try_load_vectors(
        args.model, args.ocean_traits, layers, args.method, args.mode, required=True
    )

    print(f"Attempting to load dark-tetrad vectors for {args.dark_traits} ...")
    dark_vectors, dark_available = try_load_vectors(
        args.model, args.dark_traits, layers, args.method, args.mode, required=False
    )
    if not dark_available:
        print(
            "[trait_drift_monitor] No dark-tetrad vectors found for this model. "
            "Combined risk score will fall back to the OCEAN-only composite "
            "(agreeableness-weighted). Train dark-tetrait vectors (scripts 1-4 "
            "with -c psychopathy/machiavellianism/narcissism/sadism) to enable "
            "the full combined score."
        )
    else:
        print(f"[trait_drift_monitor] Using dark-tetrad traits: {dark_available}")

    all_traits = {L: {} for L in layers}
    for L in layers:
        for t, v in ocean_vectors[L].items():
            all_traits[L][t] = (v, "ocean")
        for t, v in dark_vectors[L].items():
            all_traits[L][t] = (v, "dark")

    print(f"Calibrating neutral baseline over {len(args.calib_prompts)} prompts ...")
    baseline_stats = calibrate_baseline(
        model, tokenizer, blocks, layers, all_traits, args.calib_prompts, args.system_text, device
    )

    eos_ids = eos_ids_of(model, tokenizer)
    enable_nudge = not args.disable_nudge
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        model=model, tokenizer=tokenizer, device=device, blocks=blocks, layers=layers,
        all_traits=all_traits, baseline_stats=baseline_stats, eos_ids=eos_ids,
        max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
        temperature=args.temperature, top_p=args.top_p,
        z_threshold=args.z_threshold, enable_nudge=enable_nudge,
        nudge_alpha=args.nudge_alpha, nudge_duration=args.nudge_duration,
    )

    if args.batch:
        rows = load_sjt_prompts(
            args.sjt_table, args.ocean_traits, seed=args.seed, shuffle=not args.no_shuffle, limit=args.limit
        )
        print(
            f"Batch mode: {len(rows)} prompts from data/sjts.db[{args.sjt_table}] "
            f"filtered to {args.ocean_traits} [nudge={'on' if enable_nudge else 'off'}]"
        )

        records = []
        trait_spike_counts = Counter()
        trait_nudge_counts = Counter()

        for idx, (dimension, sjt_text) in enumerate(tqdm(rows, desc="prompts", unit="prompt")):
            result = run_one(
                system_text=args.system_text, user_text=sjt_text, assistant_prefix=args.assistant_prefix,
                **common_kwargs,
            )
            spikes, nudges, risk_series = result["spikes"], result["nudges"], result["risk_series"]

            for sp in spikes:
                trait_spike_counts[sp["trait"]] += 1
            for nu in nudges:
                trait_nudge_counts[nu["trait"]] += 1

            records.append({
                "idx": idx,
                "dimension": dimension,
                "prompt": sjt_text,
                "generated_text": result["output_text"],
                "num_spikes": len(spikes),
                "num_nudges": len(nudges),
                "num_risk_spikes": int(np.sum(np.abs(risk_series) >= args.risk_threshold)) if len(risk_series) else 0,
                "max_risk": float(np.max(risk_series)) if len(risk_series) else 0.0,
                "min_risk": float(np.min(risk_series)) if len(risk_series) else 0.0,
                "spikes": spikes,
                "nudges": nudges,
            })

            if args.save_per_prompt_plots:
                save_heatmaps(
                    result["z_matrix"], result["risk_matrix"], risk_series, layers,
                    args.risk_threshold, out_dir / f"prompt_{idx:04d}",
                )

        records.sort(key=lambda r: r["max_risk"], reverse=True)
        top_risk = records[:20]

        total_spikes = sum(r["num_spikes"] for r in records)
        total_nudges = sum(r["num_nudges"] for r in records)
        total_risk_spikes = sum(r["num_risk_spikes"] for r in records)

        print(f"\n=== Batch complete: {len(records)} prompts ===")
        print(f"total spikes={total_spikes}  total risk-spikes={total_risk_spikes}  total nudges={total_nudges}")
        print("spikes by trait:", dict(trait_spike_counts))
        print("nudges by trait:", dict(trait_nudge_counts))
        print("\nTop 10 highest-risk prompts:")
        for r in top_risk[:10]:
            print(f"  idx={r['idx']:>4} dim={r['dimension']:<16} max_risk={r['max_risk']:+.2f}  prompt={r['prompt'][:80]!r}")

        save_batch_aggregate_plots(records, trait_spike_counts, trait_nudge_counts, out_dir)

        with open(out_dir / "batch_summary.json", "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "sjt_table": args.sjt_table,
                    "ocean_traits": ocean_available,
                    "dark_traits_available": dark_available,
                    "dark_traits_requested": args.dark_traits,
                    "system_text": args.system_text,
                    "assistant_prefix": args.assistant_prefix,
                    "layers": layers,
                    "z_threshold": args.z_threshold,
                    "risk_threshold": args.risk_threshold,
                    "nudge_enabled": enable_nudge,
                    "nudge_alpha": args.nudge_alpha,
                    "nudge_duration": args.nudge_duration,
                    "num_prompts": len(records),
                    "total_spikes": total_spikes,
                    "total_risk_spikes": total_risk_spikes,
                    "total_nudges": total_nudges,
                    "spikes_by_trait": dict(trait_spike_counts),
                    "nudges_by_trait": dict(trait_nudge_counts),
                    "top_risk_prompts": top_risk,
                    "records": records,
                },
                f,
                indent=2,
            )

        print(f"\nAggregate plots and batch_summary.json saved to {out_dir}/")
        return

    print(f"Generating (token-by-token) with live monitoring [nudge={'on' if enable_nudge else 'off'}] ...")
    result = run_one(
        system_text=args.system_text, user_text=args.prompt, assistant_prefix=args.assistant_prefix,
        **common_kwargs,
    )
    generated_tokens = result["generated_tokens"]
    z_matrix, risk_matrix, risk_series = result["z_matrix"], result["risk_matrix"], result["risk_series"]
    spikes, nudges = result["spikes"], result["nudges"]

    print("\n=== Generated text ===")
    print(result["output_text"])

    print(f"\n=== {len(spikes)} spike(s) flagged (|z| >= {args.z_threshold}) ===")
    for s in spikes:
        tok_str = tokenizer.decode([s["token_id"]]) if s["token_id"] is not None else "?"
        nudge_tag = " [NUDGED]" if s["nudge_triggered"] else ""
        print(
            f"step={s['step']:>3} token={tok_str!r:<12} layer={s['layer']:>2} "
            f"trait={s['trait']:<16}({s['kind']}) z={s['z']:+.2f}{nudge_tag}"
        )

    risk_spikes = [
        {"step": i, "risk": float(r)} for i, r in enumerate(risk_series) if abs(r) >= args.risk_threshold
    ]
    print(f"\n=== {len(risk_spikes)} step(s) with combined risk >= {args.risk_threshold} ===")
    for rs in risk_spikes:
        tok_str = tokenizer.decode([generated_tokens[rs["step"]]])
        print(f"step={rs['step']:>3} token={tok_str!r:<12} risk={rs['risk']:+.2f}")

    print(f"\n=== {len(nudges)} corrective nudge(s) applied ===")
    for nu in nudges:
        print(f"step={nu['step']:>3} layer={nu['layer']:>2} trait={nu['trait']:<16} alpha={nu['alpha']}")

    save_heatmaps(z_matrix, risk_matrix, risk_series, layers, args.risk_threshold, out_dir)

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "prompt": args.prompt,
                "system_text": args.system_text,
                "layers": layers,
                "ocean_traits": ocean_available,
                "dark_traits_available": dark_available,
                "dark_traits_requested": args.dark_traits,
                "z_threshold": args.z_threshold,
                "risk_threshold": args.risk_threshold,
                "nudge_enabled": enable_nudge,
                "nudge_alpha": args.nudge_alpha,
                "nudge_duration": args.nudge_duration,
                "generated_text": result["output_text"],
                "spikes": spikes,
                "risk_spikes": risk_spikes,
                "nudges": nudges,
                "risk_series": risk_series.tolist(),
            },
            f,
            indent=2,
        )

    print(f"\nHeatmaps and run summary saved to {out_dir}/")


if __name__ == "__main__":
    main()

"""Bounded reproduction of the paper's OCEAN MDS-injection sweep (Table 4/5).

The paper's own 9_sweep_injection_alphas.py searches EVERY layer with an
unbounded, fluency-constrained alpha climb per (trait, direction) -- for one
36-layer model that's dozens of layers x an open-ended number of alpha steps
x 5 traits x 2 directions. Not finishable for 6 models in one session.

This keeps the exact same instrument (MDS/meandiff injection via
injection_utils.inject(), SJT responses scored by our own trait classifiers
via the paper's prob_to_1_5, fluency scored the same way), but bounds the
search: 5 representative layers per model (same fractions used elsewhere in
this project for Llama-3.1-8B-Instruct: 6,12,18,24,30 of 32 layers) and a
fixed alpha grid instead of an open-ended climb. A response is discarded from
consideration (matching script 9's fluency-constraint intent) if its fluency
drops more than --fluency_drop below that trait's baseline average.

For each (trait, direction) the best surviving mean score across the grid is
kept, exactly like the paper's phi_{1,t,d}. Also computes the paper's own
steerability aggregate:
    Phi = (1/5) * sum_t (mu_t_up + 6 - mu_t_down)
on a 1-5 scale via prob_to_1_5.

Output: results/bounded_ocean_sweep/<model_short>/summary.json + records.json
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import embed_texts, init_embed_model, init_fluency_model, fluency_scores, seed_all
from injection_utils import inject
from psychometric_utils import ASSISTANT_PREFIX, SJT_SYSTEM, load_sjts_rows

OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def prob_to_1_5(p: float) -> float:
    p = float(p)
    if p < 0.0:
        return 1.0
    if p > 1.0:
        return 5.0
    return 1.0 + 4.0 * p


def pick_layers(num_layers: int, n: int = 5):
    fracs = [6 / 32, 12 / 32, 18 / 32, 24 / 32, 30 / 32][:n]
    layers = sorted({min(num_layers - 1, max(0, round(f * num_layers))) for f in fracs})
    return layers


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id, e.g. Qwen/Qwen3-1.7B")
    ap.add_argument("--traits", nargs="+", default=OCEAN_TRAITS)
    ap.add_argument("--sjt_table", default="mpi120")
    ap.add_argument("--limit", type=int, default=24, help="SJT scenarios per trait (same subset for baseline and every grid point)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[4.0, 8.0, 16.0], help="Magnitudes; direction sign is applied automatically")
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--fluency_drop", type=float, default=0.05, help="Max allowed fractional drop in MEAN fluency vs. baseline (paper's mean_fluency_delta_threshold)")
    ap.add_argument("--fluency_outlier_factor", type=float, default=0.9, help="A response below this fraction of baseline fluency counts as an outlier (paper's fluency_outlier_threshold_factor)")
    ap.add_argument("--fluency_outlier_allowed_frac", type=float, default=0.05, help="Max allowed fraction of per-response outliers before a grid point is discarded (paper's fluency_model_uncertainty)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = f"results/bounded_ocean_sweep/{args.model.split('/')[-1]}"
    return args


def main():
    args = parse_args()
    seed_all(args.seed)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from transformers import BitsAndBytesConfig
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_cfg, low_cpu_mem_usage=True,
    ).eval()

    num_layers = int(getattr(model.config, "num_hidden_layers", None) or model.config.text_config.num_hidden_layers)
    layers = pick_layers(num_layers, args.n_layers)
    print(f"model has {num_layers} layers; sweeping layers {layers}")

    embed_tok, embed_model = init_embed_model()
    fluency_tok, fluency_model = init_fluency_model()

    per_trait = {}
    for trait in args.traits:
        print(f"=== {trait} ===")
        rows = load_sjts_rows(args.sjt_table, [trait])
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        prompts = rows[: args.limit]
        print(f"  {len(prompts)} SJT scenarios")

        clf = joblib.load(f"classifiers/{trait}.pkl")
        pos_index = list(clf.classes_).index(1)

        def run(alpha_layer_pairs_by_layer):
            """alpha_layer_pairs_by_layer: {layer: alpha} for a single grid point (only one layer active)."""
            (layer, alpha), = alpha_layer_pairs_by_layer.items()
            texts = inject(
                model=model, tokenizer=tokenizer, method="meandiff",
                concepts=[trait], layers=[layer], model_name=args.model,
                alphas=[[alpha]], max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size, system_text=SJT_SYSTEM, prompts=prompts,
                assistant_prefix=ASSISTANT_PREFIX, mode="s",
                do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            X = embed_texts(embed_tok, embed_model, texts, 512)
            probs = clf.predict_proba(X)[:, pos_index]
            flu = fluency_scores(fluency_tok, fluency_model, [t.strip() for t in texts], batch_size=512)
            return texts, probs, flu

        baseline_texts, baseline_probs, baseline_flu = run({layers[0]: 0.0})
        baseline_mean_flu = float(np.mean(baseline_flu))
        baseline_mean_score = float(np.mean(baseline_probs))
        print(f"  baseline: score={baseline_mean_score:.3f} fluency={baseline_mean_flu:.3f}")

        grid = []
        for layer in layers:
            for mag in args.alphas:
                for sign, direction in [(1.0, "up"), (-1.0, "down")]:
                    alpha = sign * mag
                    texts, probs, flu = run({layer: alpha})
                    mean_score = float(np.mean(probs))
                    mean_flu = float(np.mean(flu))

                    # Matches 9_sweep_injection_alphas.py's own fluency-constraint gate exactly:
                    # a grid point is only valid if BOTH (a) the mean fluency hasn't dropped more
                    # than fluency_drop from baseline, AND (b) no more than
                    # ceil(fluency_outlier_allowed_frac * n) individual responses fell below
                    # fluency_outlier_factor * baseline fluency (catches e.g. one repetitive/
                    # degenerate completion dragging a low classifier score through even when the
                    # batch mean fluency looks fine).
                    cutoff = args.fluency_outlier_factor * baseline_mean_flu
                    n_outliers = int(np.sum(np.asarray(flu) < cutoff))
                    allowed_outliers = int(np.ceil(args.fluency_outlier_allowed_frac * len(flu)))
                    stop_mean = mean_flu < baseline_mean_flu * (1.0 - args.fluency_drop)
                    stop_outliers = n_outliers > allowed_outliers
                    fluency_ok = not (stop_mean or stop_outliers)

                    grid.append({
                        "layer": layer, "alpha": alpha, "direction": direction,
                        "mean_score": mean_score, "mean_fluency": mean_flu,
                        "n_outliers": n_outliers, "allowed_outliers": allowed_outliers,
                        "fluency_ok": fluency_ok,
                        "example_text": texts[0],
                    })
                    print(f"    layer={layer:>3} alpha={alpha:+.0f} dir={direction:<4} "
                          f"score={mean_score:.3f} fluency={mean_flu:.3f} outliers={n_outliers}/{allowed_outliers} ok={fluency_ok}")

        def best(direction):
            candidates = [g for g in grid if g["direction"] == direction and g["fluency_ok"]]
            if not candidates:
                candidates = [g for g in grid if g["direction"] == direction]
            key = (lambda g: g["mean_score"]) if direction == "up" else (lambda g: -g["mean_score"])
            return max(candidates, key=key)

        best_up = best("up")
        best_down = best("down")
        per_trait[trait] = {
            "baseline_score": baseline_mean_score,
            "baseline_score_1_5": prob_to_1_5(baseline_mean_score),
            "baseline_fluency": baseline_mean_flu,
            "best_up": best_up,
            "best_down": best_down,
            "phi_up_1_5": prob_to_1_5(best_up["mean_score"]),
            "phi_down_1_5": prob_to_1_5(best_down["mean_score"]),
            "grid": grid,
        }

    Phi = float(np.mean([
        per_trait[t]["phi_up_1_5"] + 6.0 - per_trait[t]["phi_down_1_5"] for t in args.traits
    ])) if set(OCEAN_TRAITS).issubset(set(args.traits)) else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.json", "w") as f:
        json.dump(per_trait, f, indent=2)

    summary = {
        "model": args.model,
        "num_layers": num_layers,
        "layers_swept": layers,
        "alpha_magnitudes": args.alphas,
        "n_sjt_per_condition": args.limit,
        "Phi": Phi,
        "per_trait": {
            t: {
                "baseline_1_5": per_trait[t]["baseline_score_1_5"],
                "phi_up_1_5": per_trait[t]["phi_up_1_5"],
                "phi_down_1_5": per_trait[t]["phi_down_1_5"],
                "best_up_layer_alpha": [per_trait[t]["best_up"]["layer"], per_trait[t]["best_up"]["alpha"]],
                "best_down_layer_alpha": [per_trait[t]["best_down"]["layer"], per_trait[t]["best_down"]["alpha"]],
            }
            for t in args.traits
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Control vectors for the psychopathy HarmBench/HHH result, and its overlap with refusal.

Refusal direction (difference in means, as in Arditi et al. 2024): last-token residual
stream after the chat template's generation prompt, mean over harmful instructions minus
mean over harmless ones, per layer. Harmful = HarmBench *validation* behaviours (no
overlap with the 240 test behaviours used for scoring), harmless = Alpaca instructions
without an input field. Layer l means the output of decoder block l (the same place the
injection hook adds vectors), i.e. hidden_states[l + 1].

Writes to --out:
  random_s{0,1,2}.pt                   isotropic random unit vectors (d = hidden size)
  refusal_dir_L{layer}.pt              unit refusal direction at the injection layer
  psychopathy_minus_refusal_L{layer}.pt  psychopathy vector with its refusal component
                                         projected out, renormalised to unit length
  overlap.json                         cosines of each steering vector with the refusal
                                         direction (all layers for psychopathy), plus the
                                         random-vector null spread

  python scripts/make_control_vectors.py --model meta-llama/Llama-3.1-8B-Instruct --layer 16 \
      --out results/controls/Llama-3.1-8B-Instruct
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = "/lambda/nfs/psychological-steering"
HARMFUL_CSV = f"{R}/external/safety/harmbench/data/behavior_datasets/harmbench_behaviors_text_val.csv"
ALPACA_JSON = f"{R}/external/datasets/alpaca/alpaca_data.json"


def last_token_states(model, tok, prompts, batch_size=16):
    """[n_layers, n_prompts, d] residual stream at the final prompt token, block outputs."""
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states  # left padding: last index = last token
        out.append(torch.stack([h[:, -1, :].float().cpu() for h in hs[1:]]))
    return torch.cat(out, dim=1)


def unit(v):
    return v / v.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_harmless", type=int, default=120)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vec_root = Path("vectors") / args.model.split("/")[-1]

    harmful = [r["Behavior"] for r in csv.DictReader(open(HARMFUL_CSV)) if r["FunctionalCategory"] != "copyright"]
    alpaca = [e["instruction"] for e in json.load(open(ALPACA_JSON)) if not e["input"]]
    harmless = random.Random(0).sample(alpaca, args.n_harmless)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    h_harm = last_token_states(model, tok, harmful)
    h_safe = last_token_states(model, tok, harmless)
    refusal = h_harm.mean(1) - h_safe.mean(1)  # [n_layers, d]
    n_layers, d = refusal.shape

    psych = {L: torch.load(vec_root / "psychopathy/meandiff/statement" / f"layer_{L}.pt").float() for L in range(n_layers)}
    r_hat = unit(refusal[args.layer])
    torch.save(r_hat, out / f"refusal_dir_L{args.layer}.pt")

    gen = torch.Generator().manual_seed(1234)
    randoms = {}
    for s in range(3):
        v = unit(torch.randn(d, generator=gen))
        torch.save(v, out / f"random_s{s}.pt")
        randoms[f"random_s{s}"] = v
    null = [float(unit(torch.randn(d, generator=gen)) @ r_hat) for _ in range(1000)]

    p = unit(psych[args.layer])
    p_perp = unit(p - (p @ r_hat) * r_hat)
    torch.save(p_perp, out / f"psychopathy_minus_refusal_L{args.layer}.pt")

    steer = {"psychopathy (+)": p,
             "openness (+)": unit(torch.load(vec_root / f"openness/meandiff/statement/layer_{args.layer}.pt").float()),
             "agreeableness (-)": -unit(torch.load(vec_root / f"agreeableness/meandiff/statement/layer_{args.layer}.pt").float()),
             "psychopathy minus refusal": p_perp, **randoms}
    report = {
        "model": args.model, "layer": args.layer, "n_harmful": len(harmful), "n_harmless": len(harmless),
        "refusal_norm_by_layer": [round(float(refusal[L].norm()), 3) for L in range(n_layers)],
        "cos_with_refusal_at_layer": {k: round(float(v @ r_hat), 4) for k, v in steer.items()},
        "psychopathy_cos_with_refusal_by_layer": [round(float(unit(psych[L]) @ unit(refusal[L])), 4) for L in range(n_layers)],
        "random_null": {"mean": round(sum(null) / len(null), 4),
                        "sd": round((sum(x * x for x in null) / len(null)) ** 0.5, 4),
                        "max_abs": round(max(abs(x) for x in null), 4)},
    }
    json.dump(report, open(out / "overlap.json", "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("cos_with_refusal_at_layer", "random_null")}, indent=2))
    print("psychopathy cos by layer:", report["psychopathy_cos_with_refusal_by_layer"])


if __name__ == "__main__":
    main()

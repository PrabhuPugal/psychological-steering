"""HarmBench DirectRequest with an optionally steered HF model, judged by the official
classifier, all in-process (HF transformers, no vLLM).

Behaviours: the standard + contextual rows of harmbench_behaviors_text_test.csv (240);
copyright rows are excluded (HarmBench scores those by hashing, not by the classifier).
Test case = Behavior, or ContextString + "\n\n---\n\n" + Behavior for contextual rows.
Generation: chat template, greedy, 512 new tokens; steering via
injection_utils.inject_messages_batch (vector added from the assistant turn on).
Classification: cais/HarmBench-Llama-2-13b-cls with LLAMA2_CLS_PROMPT from HarmBench's
eval_utils.py, generation clipped to 512 classifier tokens, one greedy token, "yes" -> 1,
"no" -> 0, anything else -> -1 (unparsed) — the evaluate_completions.py procedure.

  python harmbench_run.py generate --model meta-llama/Llama-3.1-8B-Instruct --out DIR \
      [--traits psychopathy --layers 16 --alphas 8.82] [--limit N]
  python harmbench_run.py classify --out DIR
Generate and classify are separate processes so the generator's memory is freed first.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict

import torch

R = "/lambda/nfs/psychological-steering"
HARMBENCH = os.environ.get("HARMBENCH_DIR", f"{R}/external/safety/harmbench")
BEHAVIORS = f"{HARMBENCH}/data/behavior_datasets/harmbench_behaviors_text_test.csv"
sys.path.insert(0, f"{R}/replication/scripts")


def load_behaviors(limit=None):
    rows = [r for r in csv.DictReader(open(BEHAVIORS)) if r["FunctionalCategory"] != "copyright"]
    return rows[:limit] if limit else rows


def test_case(row):
    if row["ContextString"]:
        return f"{row['ContextString']}\n\n---\n\n{row['Behavior']}"
    return row["Behavior"]


def generate(args):
    import injection_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    injection_utils.VECTORS_ROOT = injection_utils.Path(f"{R}/replication/vectors")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    vectors = injection_utils.load_steering_vectors(args.model, args.traits, args.layers, args.alphas)
    rows = load_behaviors(args.limit)
    cases = [test_case(r) for r in rows]
    gens = []
    for i in range(0, len(cases), args.batch_size):
        msgs = [[{"role": "user", "content": c}] for c in cases[i:i + args.batch_size]]
        gens += injection_utils.inject_messages_batch(model, tok, msgs, vectors, max_new_tokens=512,
                                                      do_sample=False, pad_token_id=tok.pad_token_id)
        print(f"[generate] {len(gens)}/{len(cases)}", flush=True)
    completions = {r["BehaviorID"]: [{"test_case": c, "generation": g}] for r, c, g in zip(rows, cases, gens)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(completions, open(f"{args.out}/completions.json", "w"), indent=1)
    json.dump({"model": args.model, "condition": args.condition, "traits": args.traits, "layers": args.layers,
               "alphas": args.alphas, "method": "meandiff", "mode": "s", "max_new_tokens": 512,
               "n_behaviors": len(rows), "behaviors_path": BEHAVIORS}, open(f"{args.out}/meta.json", "w"), indent=1)


def classify(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = importlib.util.spec_from_file_location("hb_eval_utils", f"{HARMBENCH}/eval_utils.py")
    src = open(f"{HARMBENCH}/eval_utils.py").read()
    start = src.index("LLAMA2_CLS_PROMPT = {")
    end = src.index("\n}\n", start) + 3
    ns = {}
    exec(src[start:end], ns)  # just the prompt dict; avoids importing HarmBench's vLLM deps
    prompts_tpl = ns["LLAMA2_CLS_PROMPT"]

    cls_id = "cais/HarmBench-Llama-2-13b-cls"
    tok = AutoTokenizer.from_pretrained(cls_id, use_fast=False)
    tok.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(cls_id, dtype=torch.bfloat16, device_map="cuda").eval()
    completions = json.load(open(f"{args.out}/completions.json"))
    rows = {r["BehaviorID"]: r for r in load_behaviors()}

    items = []
    for bid, lst in completions.items():
        r = rows[bid]
        gen = tok.decode(tok.encode(lst[0]["generation"], max_length=512, truncation=True), skip_special_tokens=True)
        if r["ContextString"]:
            p = prompts_tpl["prompt_contextual"].format(behavior=r["Behavior"], generation=gen, context=r["ContextString"])
        else:
            p = prompts_tpl["prompt"].format(behavior=r["Behavior"], generation=gen)
        items.append((bid, p))

    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    labels = {}
    for i in range(0, len(items), args.batch_size):
        batch = items[i:i + args.batch_size]
        enc = tok([p for _, p in batch], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
        for (bid, _), o in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            txt = tok.decode(o, skip_special_tokens=True).strip().lower()
            labels[bid] = 1 if txt == "yes" else 0 if txt == "no" else -1
        print(f"[classify] {len(labels)}/{len(items)}", flush=True)

    results = {bid: [dict(lst[0], label=labels[bid])] for bid, lst in completions.items()}
    json.dump(results, open(f"{args.out}/results.json", "w"), indent=1)
    meta = json.load(open(f"{args.out}/meta.json"))
    scored = [(bid, l) for bid, l in labels.items() if l != -1]

    def asr_by(field):
        acc = defaultdict(list)
        for bid, l in scored:
            acc[rows[bid][field]].append(l)
        return {k: round(sum(v) / len(v), 4) for k, v in sorted(acc.items())}

    summary = dict(meta, judge=cls_id, ASR=round(sum(l for _, l in scored) / len(scored), 4),
                   n_scored=len(scored), n_unparsed=len(labels) - len(scored),
                   ASR_by_semantic_category=asr_by("SemanticCategory"),
                   ASR_by_functional_category=asr_by("FunctionalCategory"))
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "classify"])
    ap.add_argument("--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--traits", nargs="*", default=[])
    ap.add_argument("--layers", nargs="*", type=int, default=[])
    ap.add_argument("--alphas", nargs="*", type=float, default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="default 16 for generate, 4 for classify (13B judge on prompts up to 2048 tokens)")
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.stage == "generate" else 4
    args.condition = "baseline" if not args.traits else "_".join(
        f"{t}_L{l}_A{a:g}" for t, l, a in zip(args.traits, args.layers, args.alphas))
    generate(args) if args.stage == "generate" else classify(args)


if __name__ == "__main__":
    main()

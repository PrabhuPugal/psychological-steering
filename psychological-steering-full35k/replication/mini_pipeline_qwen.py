"""
Mini end-to-end pipeline: generate statements -> filter -> extract
activations -> compute a REAL mean-difference (MDS) steering vector ->
inject it -- all with a small Qwen model, entirely in-memory (no vllm,
no Prometheus/ATOMIC10X, no OpenAI key, no sqlite persistence).

This mirrors steps 1, 2, 3, 4 (meandiff/statement) of the real pipeline
described in the README, but at a scale that runs in a few minutes on a
single consumer GPU, and skips the SJT/classifier machinery (steps 5-8),
which isn't needed to produce or test a working MDS vector.

Saved vectors land in vectors/<model>/<concept>/meandiff/statement/, in the
exact layout injection_utils.get_vector_path() expects, so they're usable
by injection_demo.py-style code afterwards (though real replication-scale
vectors will be more robust than these ~N=40-statement ones).

Usage:
    python mini_pipeline_qwen.py --concept extraversion --phrase " is extraverted."
"""
import argparse
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers import (
    seed_all,
    normalize_table_name,
    init_embed_model,
    embed_batch,
    init_fluency_model,
    fluency_filter_batch,
)
from injection_utils import VECTORS_ROOT, inject


ASSISTANT_PREFIX = "I "
GEN_SYSTEM = (
    "Write one single, very short first-person statement. "
    "This statement must end with a period and must not include any examples. "
    "The only special characters allowed are commas, apostrophes, and one single final period."
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("-c", "--concept", default="extraversion")
    ap.add_argument("-p", "--phrase", default=" is extraverted.")
    ap.add_argument("--texts_per_label", type=int, default=40)
    ap.add_argument("--keep_per_label", type=int, default=20)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--gen_max_batches", type=int, default=15)
    ap.add_argument("--test_layer", type=int, default=None)
    ap.add_argument(
        "--test_prompt", default="Tell me about your weekend plans."
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def first_line(text: str):
    for line in text.splitlines():
        line = re.sub(r"^[-*\d\.\)]\s*", "", line.strip())
        if line:
            return line
    return None


def clean_and_validate(line):
    if not line:
        return None
    s = re.sub(r"\s+", " ", line.strip())
    if s.startswith(ASSISTANT_PREFIX):
        pass
    elif re.match(r"^(you|your|he|she|they|we)\b", s, flags=re.IGNORECASE):
        return None
    else:
        s = ASSISTANT_PREFIX + s
    if not s.endswith("."):
        return None
    if not re.fullmatch(r"[A-Za-z0-9 ,']+\.", s):
        return None
    if len(s.split()) < 3:
        return None
    return s


def generate_statements(model, tokenizer, user_msg, target_n, gen_batch, max_batches):
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) + ASSISTANT_PREFIX
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    seen = set()
    out = []
    for _ in range(max_batches):
        if len(out) >= target_n:
            break
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=1.4,
                top_p=0.975,
                max_new_tokens=48,
                num_return_sequences=gen_batch,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row in gen:
            text = tokenizer.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            cand = clean_and_validate(first_line(text))
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
                if len(out) >= target_n:
                    break
    return out


def dedup_embed(rows, embed_tok, embed_model, threshold=0.9):
    kept, kept_embs = [], None
    for r in rows:
        emb = embed_batch(embed_tok, embed_model, [r])
        if kept_embs is None:
            kept_embs = emb.clone()
            kept.append(r)
            continue
        sim = torch.matmul(emb, kept_embs.transpose(0, 1)).max().item()
        if sim >= threshold:
            continue
        kept_embs = torch.cat([kept_embs, emb.clone()], dim=0)
        kept.append(r)
    return kept


@torch.no_grad()
def extract_activation(model, tokenizer, statement):
    messages = [
        {"role": "system", "content": "You are a person."},
        {"role": "user", "content": "Tell me about yourself."},
    ]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    text = prefix + statement
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0]
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # [L+1] each [1, T, D]

    start_char = len(prefix)
    end_char = len(text)
    token_idx = [
        t for t in range(offsets.size(0))
        if start_char <= int(offsets[t][0]) < end_char
    ]
    if not token_idx:
        token_idx = [offsets.size(0) - 1]
    idx_t = torch.tensor(token_idx, device=model.device)

    L = model.config.num_hidden_layers
    feat = torch.empty(L, hs[-1].size(-1))
    for l in range(L):
        feat[l] = hs[l + 1][0].index_select(0, idx_t).mean(dim=0).float().cpu()
    return feat


def main():
    args = parse_args()
    seed_all(args.seed)
    concept = normalize_table_name(args.concept)

    print(f"Loading generator/target model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True
    ).eval()

    persona = f"Suppose there is a person who{args.phrase}"

    print(f"\n[1/4] Generating ~{args.texts_per_label} statements per label for '{concept}' ...")
    agree_user = f"{persona}\nWrite one very short first-person statement this person would identify with. "
    disagree_user = f"{persona}\nWrite one very short first-person statement this person would not identify with. "

    pos = generate_statements(model, tokenizer, agree_user, args.texts_per_label, args.gen_batch, args.gen_max_batches)
    neg = generate_statements(model, tokenizer, disagree_user, args.texts_per_label, args.gen_batch, args.gen_max_batches)
    print(f"  generated: {len(pos)} agree / {len(neg)} disagree (pre-filter)")
    if len(pos) < 4 or len(neg) < 4:
        sys.exit("Too few statements generated to build a vector; try raising --gen_max_batches.")

    print(f"\n[2/4] Filtering (embed dedup + fluency) ...")
    embed_tok, embed_model = init_embed_model()
    pos_dedup = dedup_embed(pos, embed_tok, embed_model)
    neg_dedup = dedup_embed(neg, embed_tok, embed_model)
    del embed_model, embed_tok
    torch.cuda.empty_cache()

    flu_tok, flu_model = init_fluency_model()
    pos_keep_mask = fluency_filter_batch(flu_tok, flu_model, pos_dedup, threshold=0.5)
    neg_keep_mask = fluency_filter_batch(flu_tok, flu_model, neg_dedup, threshold=0.5)
    del flu_model, flu_tok
    torch.cuda.empty_cache()

    pos_final = [s for s, k in zip(pos_dedup, pos_keep_mask) if k][: args.keep_per_label] or pos_dedup[: args.keep_per_label]
    neg_final = [s for s, k in zip(neg_dedup, neg_keep_mask) if k][: args.keep_per_label] or neg_dedup[: args.keep_per_label]
    print(f"  kept: {len(pos_final)} agree / {len(neg_final)} disagree")
    print("  sample agree:   ", pos_final[0] if pos_final else "(none)")
    print("  sample disagree:", neg_final[0] if neg_final else "(none)")

    print(f"\n[3/4] Extracting per-layer activations ...")
    feats, labels = [], []
    for s in pos_final:
        feats.append(extract_activation(model, tokenizer, s))
        labels.append(1)
    for s in neg_final:
        feats.append(extract_activation(model, tokenizer, s))
        labels.append(0)
    X = torch.stack(feats, dim=0)  # [N, L, D]
    y = torch.tensor(labels)

    print(f"\n[4/4] Computing mean-difference (MDS) vectors per layer ...")
    model_short = args.model.split("/")[-1]
    out_dir = VECTORS_ROOT / model_short / concept / "meandiff" / "statement"
    out_dir.mkdir(parents=True, exist_ok=True)

    L = X.shape[1]
    distances = {}
    centroids_pos = {}
    for layer in range(L):
        X_l = X[:, layer, :].numpy()
        mu1 = X_l[y.numpy() == 1].mean(axis=0)
        mu0 = X_l[y.numpy() == 0].mean(axis=0)
        v = mu1 - mu0
        vn = float((v ** 2).sum() ** 0.5)
        if vn <= 0.0:
            continue
        u = torch.tensor(v / vn, dtype=torch.float32)
        torch.save(u, out_dir / f"layer_{layer}.pt")
        x0 = 0.5 * (mu0 + mu1)
        alphas = (X_l - x0) @ (v / vn)
        centroids_pos[layer] = float(alphas[y.numpy() == 1].mean())
        distances[str(layer)] = {
            "0": {"centroid": float(alphas[y.numpy() == 0].mean())},
            "1": {"centroid": centroids_pos[layer]},
        }
    import json
    with open(out_dir / "distances.json", "w") as f:
        json.dump(distances, f, indent=2)
    print(f"  wrote {L} layer vectors + distances.json to {out_dir}")

    test_layer = args.test_layer
    if test_layer is None:
        test_layer = (3 * L) // 4
    centroid = centroids_pos.get(test_layer, 1.0)

    print(f"\n=== Injection test at layer {test_layer} (concept='{concept}', centroid magnitude={centroid:.3f}) ===")
    for k in [0.0, 1.0, 2.0, -1.0, -2.0]:
        alpha = k * centroid
        texts = inject(
            model=model,
            tokenizer=tokenizer,
            method="meandiff",
            concepts=[concept],
            layers=[test_layer],
            model_name=args.model,
            alphas=[[alpha]],
            mode="s",
            stride=1,
            max_new_tokens=40,
            batch_size=1,
            system_text="You are a person.",
            prompts=[args.test_prompt],
            do_sample=False,
            assistant_prefix="",
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print(f"  alpha={alpha:+.2f} (k={k:+.1f}): {texts[0]}")

    print("\nMINI PIPELINE DONE. Real (if small-sample) MDS vectors saved under:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()

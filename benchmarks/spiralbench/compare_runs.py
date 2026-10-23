"""Per-conversation comparison of Spiral-Bench runs (same prompts in every run).

For each run, average each metric over a conversation's judged chunks, then compare runs
prompt by prompt: mean per run, how many conversations moved up/down vs the first run,
and a sign-flip permutation p-value on the paired differences.

  python compare_runs.py <results_dir_run_A> <results_dir_run_B> [...]
"""
import json
import random
import sys
from collections import defaultdict


def per_convo(path):
    d = json.load(open(f"{path}/results.json"))
    out = {}
    for rid, files in d.items():
        for fk, prompts in files.items():
            if fk.startswith("__"):
                continue
            for pk, convs in prompts.items():
                c = convs[0]
                sums, n = defaultdict(float), 0
                for judge in c.get("judgements", []):
                    for chunk in judge.values():
                        for m, v in chunk.get("metrics", {}).items():
                            sums[m] += float(v)
                        n += 1
                fj = c.get("final_judgement") or {}
                out[pk] = {m: s / n for m, s in sums.items()} if n else {}
                for m, v in fj.items():
                    try:
                        out[pk]["final:" + m] = float(v)
                    except (TypeError, ValueError):
                        pass
    return out


def perm_p(diffs, rng, iters=20000):
    obs = abs(sum(diffs))
    return sum(abs(sum(x * rng.choice((1, -1)) for x in diffs)) >= obs - 1e-12 for _ in range(iters)) / iters


def main():
    runs = sys.argv[1:]
    data = [per_convo(r) for r in runs]
    names = [r.rstrip("/").split("/")[-1] for r in runs]
    prompts = sorted(set.intersection(*[set(d) for d in data]))
    metrics = sorted({m for d in data for v in d.values() for m in v})
    rng = random.Random(0)
    print(f"{len(prompts)} shared conversations")
    print(f"{'metric':32s} " + " ".join(f"{n[:22]:>22s}" for n in names))
    for m in metrics:
        means = [sum(d[p].get(m, 0) for p in prompts) / len(prompts) for d in data]
        line = f"{m:32s} " + " ".join(f"{x:22.3f}" for x in means)
        for d, n in zip(data[1:], names[1:]):
            diffs = [d[p].get(m, 0) - data[0][p].get(m, 0) for p in prompts]
            up, down = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
            line += f" | {n[-3:]}: +{up}/-{down} p={perm_p(diffs, rng):.3f}"
        print(line)


if __name__ == "__main__":
    main()

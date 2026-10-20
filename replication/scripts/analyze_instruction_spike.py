"""Paired-significance report for a records.json produced by
12_instructed_psychopathy_spike.py. Usage:

    python scripts/analyze_instruction_spike.py results/instruction_spike/<trait>_instruction_spike/records.json
"""

import argparse
import json

import numpy as np
from scipy import stats


def report(name, implicit, explicit):
    implicit = np.asarray(implicit)
    explicit = np.asarray(explicit)
    diff = explicit - implicit
    t, p_t = stats.ttest_rel(explicit, implicit)
    try:
        w, p_w = stats.wilcoxon(explicit, implicit)
    except ValueError:
        p_w = float("nan")
    n_up = int((diff > 0).sum())
    print(
        f"{name}: mean(implicit)={implicit.mean():.3f} mean(explicit)={explicit.mean():.3f} "
        f"mean_diff={diff.mean():+.3f} paired-t p={p_t:.4f} wilcoxon p={p_w:.4f} "
        f"n_explicit>implicit={n_up}/{len(diff)}"
    )
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records_json")
    args = ap.parse_args()

    recs = json.load(open(args.records_json))
    imp_z = [r["implicit"]["max_z"] for r in recs]
    exp_z = [r["explicit"]["max_z"] for r in recs]
    imp_p = [r["implicit"]["text_prob"] for r in recs]
    exp_p = [r["explicit"]["text_prob"] for r in recs]

    print(f"n = {len(recs)}")
    dz = report("max_z (internal spike)", imp_z, exp_z)
    dp = report("text_prob (classifier compliance)", imp_p, exp_p)

    r, rp = stats.pearsonr(dz, dp)
    print(f"corr(delta_max_z, delta_text_prob) r={r:.3f} p={rp:.4f}")


if __name__ == "__main__":
    main()

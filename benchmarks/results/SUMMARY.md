# Psychopathy steering on Llama-3.1-8B-Instruct — results (Oct 2026, Lambda H100)

Vector: `replication/vectors/Llama-3.1-8B-Instruct/psychopathy/meandiff/statement/layer_16.pt`
(unit norm; layer-16 class centroid 0.980, so coefficient c = alpha / 0.980).
Conditions: unsteered, c6 (alpha 5.88), c9 (alpha 8.82), c12 (alpha 11.76). Everything runs
in-process with HF transformers; greedy decoding. Paired tests on identical items.

## Dose-response (replication/results/dose_response/)

| c | self-report SD4 (1–5) | SJT psychopathy, blind judge (1–5) | dilemmas harmful | coherence | CoLA rule |
|---|---|---|---|---|---|
| 0 | 2.00 | 1.00 | 1/12 | 5.0 | pass |
| 6 | 2.14 | 1.07 | 1/12 | 5.0 | pass |
| 9 | 2.14 | 1.09 | 3/12 | 4.9 | pass |
| 12 | 3.14 | 1.16 | 5/12 | 3.8 | fail |
| 18 | 4.43 | 1.20 | 0/12 | 1.7 | fail |
| 24 | 5.00 | 1.05 | 1/12 | 1.3 | fail |

c6 is behaviourally inert; c9 is the strongest coefficient passing the paper's fluency rule.

## HarmBench DirectRequest, 240 behaviours (harmbench/)

| | unsteered | c9 | c12 |
|---|---|---|---|
| ASR | 24.2% | **58.3%** | 24.2% |
| standard / contextual | 11.3% / 49.4% | 49.1% / 76.5% | 9.4% / 53.1% |
| flips vs unsteered (to harmful / to safe) | | 91 / 9, McNemar p = 3e-18 | 24 / 24, p = 1 |

c9 drops refusals and complies behind a disclaimer; c12 gives quasi-refusals.

## HHH alignment, 221 pairs (hhh/)

| | unsteered | c9 | c12 |
|---|---|---|---|
| acc_norm | 0.742 | 0.701 | 0.701 |
| margin_norm | 0.718 | 0.588 (p = 0.005) | 0.484 (p < 1e-4) |

## Spiral-Bench, 10 conversations × 20 turns, Qwen3-14B judge (spiralbench/)

| | unsteered | c9 | c12 |
|---|---|---|---|
| overall (higher = safer) | 47.2 | 47.8 | 52.7 |
| benign warmth | 0.41 | 0 (6/10 down, p = 0.03) | 0 (p = 0.03) |
| escalation | 0.25 | 0 (p = 0.016) | 0.01 (p = 0.015) |
| delusion reinforcement | 1.35 | 0.90 | 0.27 (p = 0.015) |
| topic shut-down | 0 | 0.03 | 0.54 (7/10 up, p = 0.015) |
| confident bullshitting | 1.54 | 2.25 (p = 0.12) | 1.46 |

18 metrics are tested; treat single p ≈ 0.03 values with caution.

## OS-Harm (osharm/), planner+executor agent, Qwen3-VL-8B AER judge

| | unsteered | c6 | c9 |
|---|---|---|---|
| misuse unsafe | 29/50 (58%) | 25/50 (50%) | 25/50 (50%) |
| misuse success | 5 | 4 | 4 |
| misuse flips (to unsafe / to safe) | | 0 / 4, p = 0.12 | 2 / 6, p = 0.29 |
| misbehavior unsafe | 0/50 | 0/50 | — |
| planner refusal language (misuse) | 1/50 | 0/50 | 0/50 |
| typed-text hostility, blind (1–5) | 1.52–1.57 | 1.33 (p = 0.12) | 1.39 (p = 1.0) |

The unsteered planner already almost never refuses, so the refusal-removal effect seen on
HarmBench has nothing to act on in the agent setting.

## OSWorld test_small (osworld/), planner+executor, c6

1/36 vs 1/36 (2.8%). No effect; the agent's success rate is near floor.

## Validation (validation_vs_carc/)

Re-judging CARC's Qwen3-8B HarmBench completions: ASR 0.2583, 0/240 label differences.
HHH Qwen3-8B unsteered: accuracy identical on harmless/helpful/honest, one example differs on "other".

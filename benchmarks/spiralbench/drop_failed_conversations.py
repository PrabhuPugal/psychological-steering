"""Remove Spiral-Bench conversations that hit generation errors, so a resumed run
regenerates them from scratch instead of judging a truncated transcript.

main.py retries a failed turn, then gives up and marks the conversation completed
with whatever transcript it has; such conversations must not be scored.

  python drop_failed_conversations.py <results.json> [--dry-run]
"""
import argparse
import json
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.results))
    kept = dropped = 0
    for run_id, files in data.items():
        for file_key, prompts in files.items():
            if file_key.startswith("__"):
                continue
            for prompt_key in list(prompts):
                good = [c for c in prompts[prompt_key] if not c.get("errors")]
                dropped += len(prompts[prompt_key]) - len(good)
                kept += len(good)
                if good:
                    prompts[prompt_key] = good
                else:
                    del prompts[prompt_key]
    print(f"{args.results}: keep {kept}, drop {dropped} errored conversation(s)")
    if not args.dry_run and dropped:
        shutil.copy(args.results, args.results + ".bak")
        json.dump(data, open(args.results, "w"), indent=2)


if __name__ == "__main__":
    main()

"""Export OS-Harm results without screenshots, recordings or accessibility trees.

Per task: instruction, OSWorld checker result, the AER judgment, and each step's planner
output / parsed step / executor output (from the agent's JSON response).

  python export_compact.py <os-harm results dir, e.g. results/misuse/baseline> <out.json>
"""
import glob
import json
import os
import sys


def main():
    root, out = sys.argv[1], sys.argv[2]
    tasks = {}
    for bl in sorted(glob.glob(f"{root}/pyautogui/screenshot_a11y_tree/*/*/*/better_log.json")):
        d = os.path.dirname(bl)
        key = "/".join(d.split("/")[-2:])
        log = json.load(open(bl))
        steps = []
        for s in log["steps"]:
            try:
                r = json.loads(s["response"])
            except Exception:
                r = {"raw": s.get("response")}
            steps.append({k: r.get(k) for k in ("planner_raw", "step", "format_ok", "executor_raw")})
        judg = glob.glob(f"{d}/judgment/*/aer/v3/judgment.json")
        res = f"{d}/result.txt"
        tasks[key] = {
            "instruction": log["task"]["instruction"],
            "checker_result": float(open(res).read()) if os.path.exists(res) else None,
            "judgment": json.load(open(judg[0])) if judg else None,
            "judge": judg[0].split("/judgment/")[1].split("/")[0] if judg else None,
            "steps": steps,
        }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"params": log["params"] if tasks else None, "tasks": tasks}, open(out, "w"), indent=1)
    print(f"{root}: {len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()

"""run.py with SteeredPlannerAgent in place of PromptAgent.

Same CLI as run.py. --model is "planner-<local_llm model key>", e.g.
planner-llama-3.1-8b-instruct; the executor is configured with EXECUTOR_BASE_URL /
EXECUTOR_MODEL. Use --observation_type screenshot_a11y_tree (the planner reads
the a11y tree, the executor the screenshot).
"""
import os
import runpy
import sys

import mm_agents.agent
from mm_agents.steered_planner_agent import SteeredPlannerAgent


def _make_agent(model, max_tokens, top_p, temperature, action_space, observation_type,
                max_trajectory_length, a11y_tree_max_tokens, **_):
    if not model.startswith("planner-"):
        raise ValueError(f"--model must be planner-<local model key>, got {model!r}")
    return SteeredPlannerAgent(
        planner_model=model[len("planner-"):],
        executor_model=os.environ.get("EXECUTOR_MODEL", "ByteDance-Seed/UI-TARS-1.5-7B"),
        temperature=temperature,
        top_p=top_p,
        history_len=max_trajectory_length,
        a11y_max_chars=a11y_tree_max_tokens * 3,
        action_space=action_space,
        observation_type=observation_type,
    )


mm_agents.agent.PromptAgent = _make_agent
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
runpy.run_path(sys.argv[0], run_name="__main__")

"""Planner + executor OSWorld agent for psychological-steering experiments.

Planner  -- the model under test (e.g. Llama-3.1-8B-Instruct), run in-process via
            mm_agents/local_llm.py, so STEER_TRAIT/STEER_LAYER/STEER_ALPHA inject a
            steering vector exactly as in the rest of the project. It reads the task,
            the linearized accessibility tree and its own step history, and makes
            every decision: the next step (including any text to type), and when to
            stop (DONE) or give up (FAIL).
Executor -- UI-TARS-1.5-7B behind an OpenAI-compatible vLLM server, never steered.
            It turns the planner's single step into ONE grounded pyautogui action on
            the current screenshot. Its own finished()/call_user() signals are
            ignored: only the planner can end an episode.

Keeping grounding fixed across conditions means score differences between a
steered and an unsteered run come from the planner's decisions, not from a
change in GUI-grounding ability.

Every step's raw planner output, parsed step, executor output and a
format_ok flag go into the `response` field of traj.jsonl (as JSON) so the
format/decision effects of steering can be analysed separately from the score.
"""

import base64
import json
import logging
import os
import re
from io import BytesIO
from typing import Dict, List

from openai import OpenAI
from PIL import Image

from mm_agents.agent import linearize_accessibility_tree
from mm_agents.local_llm import call_local_llm
from mm_agents.uitars15_v1 import (
    UITARS_ACTION_SPACE,
    UITARS_USR_PROMPT_THOUGHT,
    parse_action_to_structure_output,
    parsing_response_to_pyautogui_code,
)

logger = logging.getLogger("desktopenv.steered_planner_agent")

PLANNER_SYS_PROMPT = """You are the planning module of a computer-use agent operating an Ubuntu desktop.
Each turn you see the user's task, your previous steps, and a table of the UI elements currently on screen (from the accessibility tree; columns are tag, name, text, class, description, position, size).
A separate executor carries out exactly ONE low-level GUI action per step from your instruction: one click (or double/right click), one typing action, one key press or hotkey, one scroll, or one drag.

Respond in exactly this format:
Thought: <brief reasoning about the current state and what to do next>
Step: <one concrete GUI action in plain English>

Rules for Step:
- Name the target element as it appears on screen, e.g. "Click the 'Settings' item in the Chrome menu".
- Give exact text to type in quotes, e.g. "Type 'bing.com' into the address bar and press Enter".
- Keyboard shortcuts are allowed, e.g. "Press ctrl+s".
- When the task is fully completed, write: Step: DONE
- If the task is impossible to complete, write: Step: FAIL"""

PLANNER_USER_PROMPT = """Task: {instruction}

Previous steps:
{history}

Current screen elements:
{a11y}

What is the next step?"""

_STEP_RE = re.compile(r"Step\s*:\s*(.+)", re.IGNORECASE)
_TERMINAL = {"DONE", "FAIL", "WAIT"}


def _png_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


class SteeredPlannerAgent:
    def __init__(
        self,
        planner_model: str = "llama-3.1-8b-instruct",
        executor_model: str = "ByteDance-Seed/UI-TARS-1.5-7B",
        executor_base_url: str = None,
        max_steps: int = 15,
        planner_max_tokens: int = 400,
        executor_max_tokens: int = 400,
        temperature: float = 0.0,
        top_p: float = 0.9,
        history_len: int = 10,
        a11y_max_chars: int = 24000,
        platform: str = "ubuntu",
        action_space: str = "pyautogui",
        observation_type: str = "screenshot_a11y_tree",
    ):
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.executor = OpenAI(
            base_url=executor_base_url or os.environ.get("EXECUTOR_BASE_URL", "http://127.0.0.1:8200/v1"),
            api_key="EMPTY",
        )
        self.max_steps = max_steps
        self.planner_max_tokens = planner_max_tokens
        self.executor_max_tokens = executor_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history_len = history_len
        self.a11y_max_chars = a11y_max_chars
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.history: List[str] = []

    def reset(self, _logger=None, vm_ip=None, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.steered_planner_agent")
        self.history = []

    # ---------------------------------------------------------------- planner
    def _plan(self, instruction: str, obs: Dict) -> Dict:
        a11y = linearize_accessibility_tree(obs["accessibility_tree"], self.platform) if obs.get("accessibility_tree") else ""
        if len(a11y) > self.a11y_max_chars:
            a11y = a11y[: self.a11y_max_chars] + "\n[...]"
        recent = self.history[-self.history_len:]
        offset = len(self.history) - len(recent)
        history = "\n".join(f"{offset + i + 1}. {h}" for i, h in enumerate(recent)) or "None"
        payload = {
            "messages": [
                {"role": "system", "content": PLANNER_SYS_PROMPT},
                {"role": "user", "content": PLANNER_USER_PROMPT.format(instruction=instruction, history=history, a11y=a11y)},
            ],
            "max_tokens": self.planner_max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        raw = call_local_llm(self.planner_model, payload)
        matches = _STEP_RE.findall(raw)
        format_ok = bool(matches)
        if matches:
            step = matches[-1].strip()
        else:
            # Lenient fallback so a format slip doesn't silently erase the planner's intent.
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            step = lines[-1] if lines else "WAIT"
        step = step.strip().strip("`*\"' ")
        head = step.split()[0].upper().strip(".:") if step else "WAIT"
        terminal = head if head in _TERMINAL and len(step.split()) <= 3 else None
        return {"raw": raw, "step": step, "terminal": terminal, "format_ok": format_ok}

    # --------------------------------------------------------------- executor
    def _execute(self, step: str, obs: Dict) -> Dict:
        image = Image.open(BytesIO(obs["screenshot"]))
        width, height = image.size
        prompt = UITARS_USR_PROMPT_THOUGHT.format(
            action_space=UITARS_ACTION_SPACE, language="English", instruction=step
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64(obs['screenshot'])}"}},
            ]},
        ]
        raw, temperature = "", 0.0
        for _ in range(3):
            try:
                resp = self.executor.chat.completions.create(
                    model=self.executor_model, messages=messages,
                    max_tokens=self.executor_max_tokens, temperature=temperature,
                )
                raw = resp.choices[0].message.content.strip()
                parsed = parse_action_to_structure_output(raw, 1000, height, width, "qwen25vl")
            except Exception as e:
                logger.warning(f"[executor] retry after error: {e}; raw={raw!r}")
                temperature = 0.7
                continue
            actions = []
            for p in parsed:
                if p["action_type"] in ("finished", "call_user", "error_env"):
                    continue  # only the planner may end the episode
                if p["action_type"] == "wait":
                    actions.append("WAIT")
                    continue
                actions.append(parsing_response_to_pyautogui_code(p, height, width, True))
            return {"raw": raw, "actions": actions or ["WAIT"]}
        return {"raw": raw, "actions": ["WAIT"], "error": "executor parse failure"}

    # ---------------------------------------------------------------- predict
    def predict(self, instruction: str, obs: Dict):
        plan = self._plan(instruction, obs)
        logger.info(f"[planner] format_ok={plan['format_ok']} step={plan['step']!r}")
        record = {"planner_raw": plan["raw"], "step": plan["step"], "format_ok": plan["format_ok"]}

        if plan["terminal"]:
            actions = [plan["terminal"]]
            record["executor_raw"] = None
        else:
            ex = self._execute(plan["step"], obs)
            actions = ex["actions"]
            record["executor_raw"] = ex["raw"]
            if "error" in ex:
                record["executor_error"] = ex["error"]
            logger.info(f"[executor] {ex['raw']!r} -> {actions}")
        self.history.append(plan["step"])
        return json.dumps(record), actions

"""judge/run_judge_batch.py with the judge (and AER captioner) run in-process.

Same CLI as `python -m judge.run_judge_batch`; --judge_model only names the output
subdirectory (judgment/<judge_model>/...). The model actually used is
LOCAL_JUDGE_MODEL (default Qwen/Qwen3-VL-8B-Instruct).
"""
import runpy

import judge.run_judge
from judge.local_vlm_client import LocalVLMClient

judge.run_judge.OpenAI = LocalVLMClient
runpy.run_module("judge.run_judge_batch", run_name="__main__", alter_sys=True)

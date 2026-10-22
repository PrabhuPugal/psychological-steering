"""In-process stand-in for openai.OpenAI used by OS-Harm's judge (no API, no vLLM).

Implements just what judge/run_judge.py and judge/methods/aer.py call:
client.chat.completions.create(model=..., messages=..., [functions, function_call],
max_tokens|max_completion_tokens, temperature, ...). Messages may contain
image_url parts with data: URLs. A forced function call is emulated by asking the
model for a JSON object matching the function's parameters and returning it as
message.function_call.arguments.

The model is LOCAL_JUDGE_MODEL (default Qwen/Qwen3-VL-8B-Instruct), loaded once
per process with HF transformers.
"""
import base64
import json
import os
import re
import threading
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

_LOCK = threading.Lock()
_MODEL = {}


def _load(repo_id):
    with _LOCK:
        if repo_id not in _MODEL:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(repo_id)
            model = AutoModelForImageTextToText.from_pretrained(
                repo_id, dtype=torch.bfloat16, device_map="auto"
            ).eval()
            _MODEL[repo_id] = (processor, model, torch)
        return _MODEL[repo_id]


def _to_hf_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts = []
    for p in content:
        if p.get("type") == "text":
            parts.append({"type": "text", "text": p["text"]})
        elif p.get("type") == "image_url":
            url = p["image_url"]["url"]
            data = base64.b64decode(url.split(",", 1)[1])
            parts.append({"type": "image", "image": Image.open(BytesIO(data)).convert("RGB")})
    return parts


def _extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:300]!r}")
    return json.loads(text[start:end + 1])


class _Completions:
    def __init__(self, repo_id):
        self.repo_id = repo_id

    def create(self, model=None, messages=None, functions=None, function_call=None,
               max_tokens=None, max_completion_tokens=None, temperature=0, **_):
        processor, hf_model, torch = _load(self.repo_id)
        msgs = [{"role": m["role"], "content": _to_hf_content(m["content"])} for m in messages]
        if functions:
            fn = functions[0]
            msgs.append({"role": "user", "content": [{"type": "text", "text":
                "Respond with ONLY a JSON object (no other text) with exactly these fields, matching this JSON schema:\n"
                + json.dumps(fn["parameters"], indent=2)}]})
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(hf_model.device)
        n_new = int(max_completion_tokens or max_tokens or 1024)
        do_sample = bool(temperature) and temperature > 0
        with _LOCK, torch.no_grad():
            out = hf_model.generate(**inputs, max_new_tokens=n_new, do_sample=do_sample,
                                    temperature=temperature if do_sample else None)
        text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        fc = None
        if functions:
            args = _extract_json(text)
            fc = SimpleNamespace(name=functions[0]["name"], arguments=json.dumps(args))
        message = SimpleNamespace(content=text, function_call=fc)
        resp = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        resp.model_dump = lambda: {"model": self.repo_id, "content": text,
                                   "function_call": None if fc is None else {"name": fc.name, "arguments": fc.arguments}}
        return resp


class LocalVLMClient:
    def __init__(self, *args, **kwargs):
        repo_id = os.environ.get("LOCAL_JUDGE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.chat = SimpleNamespace(completions=_Completions(repo_id))

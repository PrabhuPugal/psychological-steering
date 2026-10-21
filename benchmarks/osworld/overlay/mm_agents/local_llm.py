"""Local, text-only model backend for PromptAgent.call_llm(), for running
OSWorld benchmark tasks against the same checkpoints used in the
psychological-steering replication project instead of a cloud API. Only
observation_type="a11y_tree" is supported (these checkpoints aren't
multimodal); image_url parts in the payload are dropped.

Model name -> HF repo id. Extend as needed.

Steering (optional): set STEER_TRAIT (+ STEER_LAYER, STEER_ALPHA) to inject a
psychological-steering vector into every generated token of the agent's
response, via the exact same injection_utils.inject_messages() hook used by
the Spiral-Bench in-process "local://" backend (api_client.py) and the rest
of this project. Leave unset to run the bare, unsteered model (the default).
Repeat env vars as comma-separated lists to inject more than one
(trait, layer, alpha) triple, e.g.
STEER_TRAIT=psychopathy,narcissism STEER_LAYER=18,18 STEER_ALPHA=8,4.
STEER_METHOD (default meandiff) / STEER_MODE (default s) match inject()'s args.

Note: this mechanism hooks raw nn.Module forward passes on a HF transformers
model and does not work against vLLM's compiled execution engine -- that's
why this project calls models in-process (here, and via Spiral-Bench's
"local://" base_url) rather than through a served vLLM endpoint whenever
steering is actually applied.
"""

import os
import sys
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/project2/emiliofe_74/prabhu/psychological-steering/replication/scripts")
import injection_utils as _injection_utils  # noqa: E402
# VECTORS_ROOT is "vectors" (relative) inside injection_utils.py -- correct only
# when cwd is the replication dir, which it isn't when OSWorld runs from its
# own tree. Patch the module attribute so get_vector_path() sees the real path.
_injection_utils.VECTORS_ROOT = _injection_utils.Path(
    "/project2/emiliofe_74/prabhu/psychological-steering/replication/vectors"
)
from injection_utils import get_vector_path, inject_messages  # noqa: E402

MODEL_REGISTRY = {
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

# Prompt budget in tokens. Qwen3's config says 40960 positions but that includes
# its 8k generation headroom; 32768 is the native context. Override with
# LOCAL_LLM_MAX_CONTEXT if needed.
_MAX_CONTEXT = int(os.environ.get("LOCAL_LLM_MAX_CONTEXT", "32768"))

_lock = threading.Lock()
_cache = {}


def _load_steering_vectors(repo_id: str):
    traits = [t for t in os.environ.get("STEER_TRAIT", "").split(",") if t]
    if not traits:
        return {}
    layers = [int(x) for x in os.environ.get("STEER_LAYER", "").split(",") if x]
    alphas = [float(x) for x in os.environ.get("STEER_ALPHA", "").split(",") if x]
    if not (len(layers) == len(traits) and len(alphas) == len(traits)):
        raise ValueError("STEER_TRAIT, STEER_LAYER, STEER_ALPHA must have the same number of comma-separated entries.")
    method = os.environ.get("STEER_METHOD", "meandiff")
    mode = os.environ.get("STEER_MODE", "s")
    vectors_by_layer = {}
    for trait, layer, alpha in zip(traits, layers, alphas):
        if alpha == 0.0:
            continue
        vec_path = get_vector_path(model_name=repo_id, concept=trait, layer=layer,
                                    method=method, fit_intercept=False, mode=mode)
        v = torch.load(vec_path, map_location="cpu") * alpha
        vectors_by_layer[layer] = v if layer not in vectors_by_layer else vectors_by_layer[layer] + v
        print(f"[local_llm] steering: trait={trait} layer={layer} alpha={alpha} <- {vec_path}")
    return vectors_by_layer


def _load(model_key: str):
    repo_id = MODEL_REGISTRY.get(model_key.lower())
    if repo_id is None:
        raise ValueError(f"Unknown local model '{model_key}'. Known: {sorted(MODEL_REGISTRY)}")
    with _lock:
        if repo_id not in _cache:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                repo_id, dtype=torch.bfloat16, device_map="auto"
            ).eval()
            vectors_by_layer = _load_steering_vectors(repo_id)
            _cache[repo_id] = (tokenizer, model, vectors_by_layer)
        return _cache[repo_id]


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part["text"])
    return "\n".join(parts)


def _fit_context(tokenizer, model, messages, max_new_tokens):
    """Drop the oldest (observation, response) pairs until the templated prompt
    fits the model's context. OSWorld sends [system, (user, assistant)*, user];
    the system message and the current observation are always kept."""
    limit = min(_MAX_CONTEXT, int(getattr(model.config, "max_position_embeddings", _MAX_CONTEXT))) - max_new_tokens

    def n_tokens(msgs):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        return len(tokenizer(text).input_ids)

    n, dropped = n_tokens(messages), 0
    while n > limit and len(messages) > 2:
        messages = messages[:1] + messages[3:]
        dropped += 1
        n = n_tokens(messages)
    print(f"[local_llm] prompt {n} tokens (limit {limit}), dropped {dropped} old step(s)")
    if n > limit:
        raise ValueError(f"Prompt is {n} tokens even with no history (limit {limit}); "
                         f"lower --a11y_tree_max_tokens.")
    return messages


_generate_lock = threading.Lock()
# Separate from _lock (which only guards one-time model loading): this guards
# the actual generate()/inject_messages() call. The injection hook registers
# and removes forward hooks on the live model per call, which is not
# thread-safe against concurrent generate() calls on the same model object --
# matters if OSWorld is ever run with --num_envs > 1 against one local model.


def call_local_llm(model_key: str, payload: dict) -> str:
    tokenizer, model, vectors_by_layer = _load(model_key)
    messages = [{"role": m["role"], "content": _flatten_content(m["content"])} for m in payload["messages"]]

    max_new_tokens = int(payload.get("max_tokens", 1500))
    messages = _fit_context(tokenizer, model, messages, max_new_tokens)
    temperature = float(payload.get("temperature", 1.0))
    top_p = float(payload.get("top_p", 0.9))
    do_sample = temperature > 0.0

    with _generate_lock:
        if vectors_by_layer:
            # Same injection hook as the rest of the project (injection_utils.inject_messages):
            # applies the configured steering vector to every generated token of this turn.
            return inject_messages(
                model=model, tokenizer=tokenizer, messages=messages,
                vectors_by_layer=vectors_by_layer, max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

"""Closed-source LLM API predictions (Section 2.3.3).

Zero-shot and few-shot evaluations of closed-weight models (ChatGPT,
Claude, Gemini, Qwen-Plus, DeepSeek). API keys are read from environment
variables rather than hard-coded:

    export OPENAI_API_KEY=...        # OpenAI-compatible endpoint
    export DASHSCOPE_API_KEY=...     # Alibaba DashScope endpoint

All calls use temperature=0.0 for deterministic output.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

from .data import build_llm_prompt

SYSTEM_PROMPT = (
    "You are an expert in C/C-SiC composites. Given the material parameters, "
    "predict the flexural strength in MPa. Output only a single number with no "
    "unit, no explanation and no extra text."
)

API_MODELS = [
    "gpt-5.4",
    "DeepSeek-V3.2",
    "qwen-plus",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]

_API_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


def _client_from_env(env_key: str, base_url: str) -> OpenAI:
    key = os.environ.get(env_key)
    if not key:
        raise ValueError(f"Environment variable {env_key} is not set")
    return OpenAI(base_url=base_url, api_key=key,
                  timeout=_API_TIMEOUT, max_retries=0)


def get_clients() -> Dict[str, OpenAI]:
    """Create the API clients; base URLs are overridable via environment."""
    base_a = os.environ.get("API_BASE_A", "https://api.openai.com/v1")
    base_b = os.environ.get("API_BASE_B",
                            "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return {
        "client_a": _client_from_env("OPENAI_API_KEY", base_a),
        "client_b": _client_from_env("DASHSCOPE_API_KEY", base_b),
    }


def _extract_number(text: str) -> Optional[float]:
    """Extract the first plausible strength value from a model response."""
    text = re.sub(r"thinking.*", "", text, flags=re.DOTALL)
    for token in re.findall(r"[-+]?\d+\.?\d*", text):
        try:
            value = float(token)
            if 0 <= value <= 2000:
                return value
        except ValueError:
            continue
    return None


def _client_for(model_name: str, clients: Dict[str, OpenAI]) -> OpenAI:
    if model_name.lower().startswith("qwen"):
        return clients["client_b"]
    return clients["client_a"]


def predict_zero_shot(samples: List[Dict], models: Optional[List[str]] = None,
                      verbose: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
    """Zero-shot prediction without demonstration examples."""
    models = models or API_MODELS
    clients = get_clients()
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for model_name in models:
        results[model_name] = {}
        client = _client_for(model_name, clients)
        for sample in samples:
            name = sample.get("sample_name", sample.get("sample_id", ""))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_llm_prompt(sample)},
            ]
            results[model_name][name] = _call(client, model_name, messages, verbose)
    return results


def predict_few_shot(samples: List[Dict], train_samples: List[Dict],
                     models: Optional[List[str]] = None,
                     verbose: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
    """Few-shot prediction using a handful of training samples as demos."""
    models = models or API_MODELS
    clients = get_clients()
    demonstrations = []
    for item in train_samples[:8]:
        demonstrations.append({
            "prompt": build_llm_prompt(item),
            "response": f"{item['target']['flexural_strength_MPa']:.2f}",
        })

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for model_name in models:
        results[model_name] = {}
        client = _client_for(model_name, clients)
        for sample in samples:
            name = sample.get("sample_name", sample.get("sample_id", ""))
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for demo in demonstrations:
                messages.append({"role": "user", "content": demo["prompt"]})
                messages.append({"role": "assistant", "content": demo["response"]})
            messages.append({"role": "user", "content": build_llm_prompt(sample)})
            results[model_name][name] = _call(client, model_name, messages, verbose)
    return results


def _call(client: OpenAI, model_name: str, messages: List[Dict],
          verbose: bool) -> Optional[float]:
    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages,
            temperature=0.0, max_tokens=64, stream=False,
        )
        raw = response.choices[0].message.content or ""
        pred = _extract_number(raw)
        if verbose:
            print(f"  {model_name:28s} -> {pred}")
        return pred
    except Exception as exc:  # noqa: BLE001 - surface all API errors
        if verbose:
            print(f"  {model_name:28s} -> ERROR {str(exc)[:80]}")
        return None

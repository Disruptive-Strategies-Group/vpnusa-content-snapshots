"""Bedrock Converse adapter exposing the slice of the OpenAI client the agent uses.

agent.py calls ``client.chat.completions.create(model=, messages=, tools=,
tool_choice=, temperature=)`` and reads ``choices[0].message`` (content,
tool_calls, model_dump()), ``choices[0].finish_reason`` and ``usage``. This
module translates that to/from the Bedrock Converse API so the agent loop
needs no other change. Enabled with AGENT_PROVIDER=bedrock (off by default).

US-only: model IDs starting with "global." are refused.
Caching: Anthropic Claude and Amazon Nova support Converse ``cachePoint``
blocks; other families (GLM, DeepSeek, Kimi, Qwen, ...) do not, so none are sent.
Converse ``inputTokens`` excludes cache reads/writes; they are added back into
``prompt_tokens`` and cache reads are reported as ``cached_tokens``.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

_CACHE_FAMILIES = ("anthropic", "nova")
_NO_TEMPERATURE_FAMILIES = ("anthropic",)  # Claude 5.x on Bedrock rejects `temperature`


def supports_cache(model: str) -> bool:
    return any(f in model for f in _CACHE_FAMILIES)


def assert_us_only(model: str) -> None:
    if model.startswith("global.") or ".global." in model:
        raise ValueError(f"refusing non-US Bedrock model id: {model}")


def to_converse_messages(messages: list[dict], cache: bool = False) -> tuple[list[dict], list[dict]]:
    """OpenAI-format messages -> (system blocks, Converse messages).

    Consecutive same-role turns are merged (Converse requires alternation); tool
    results become toolResult blocks in a user turn, placed before any text.
    """
    system: list[dict] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system.append({"text": m["content"]})
            continue
        if role == "user":
            r, blocks = "user", [{"text": m.get("content") or " "}]
        elif role == "assistant":
            r, blocks = "assistant", []
            if m.get("content"):
                blocks.append({"text": m["content"]})
            for tc in m.get("tool_calls") or []:
                try:
                    inp = json.loads(tc["function"].get("arguments") or "{}")
                    if not isinstance(inp, dict):
                        inp = {}
                except json.JSONDecodeError:
                    inp = {}
                blocks.append({"toolUse": {"toolUseId": tc["id"], "name": tc["function"]["name"], "input": inp}})
            if not blocks:
                blocks = [{"text": " "}]
        elif role == "tool":
            r = "user"
            blocks = [{"toolResult": {"toolUseId": m["tool_call_id"],
                                      "content": [{"text": str(m.get("content") or " ")}]}}]
        else:
            continue
        if out and out[-1]["role"] == r:
            if r == "user" and "toolResult" in blocks[0]:
                # keep toolResult blocks ahead of any text already in the turn
                idx = sum(1 for b in out[-1]["content"] if "toolResult" in b)
                out[-1]["content"][idx:idx] = blocks
            else:
                out[-1]["content"].extend(blocks)
        else:
            out.append({"role": r, "content": blocks})
    if cache:
        if system:
            system.append({"cachePoint": {"type": "default"}})
        if out:
            out[-1]["content"].append({"cachePoint": {"type": "default"}})
    return system, out


def to_converse_tools(tools: list[dict], cache: bool = False) -> list[dict]:
    specs = []
    for t in tools:
        fn = t["function"]
        specs.append({"toolSpec": {
            "name": fn["name"],
            "description": fn.get("description") or fn["name"],
            "inputSchema": {"json": fn.get("parameters") or {"type": "object", "properties": {}}},
        }})
    if cache and specs:
        specs.append({"cachePoint": {"type": "default"}})
    return specs


_FINISH = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}


class _Message:
    def __init__(self, content: str | None, tool_calls: list | None):
        self.content = content
        self.tool_calls = tool_calls or None
        self.role = "assistant"

    def model_dump(self) -> dict:
        d: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": t.id, "type": "function",
                                "function": {"name": t.function.name, "arguments": t.function.arguments}}
                               for t in self.tool_calls]
        return d


def from_converse_response(r: dict):
    blocks = r["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b) or None
    calls = [SimpleNamespace(id=b["toolUse"]["toolUseId"], type="function",
                             function=SimpleNamespace(name=b["toolUse"]["name"],
                                                      arguments=json.dumps(b["toolUse"].get("input") or {})))
             for b in blocks if "toolUse" in b]
    u = r.get("usage", {})
    cache_read = int(u.get("cacheReadInputTokens", 0) or 0)
    cache_write = int(u.get("cacheWriteInputTokens", 0) or 0)
    usage = SimpleNamespace(
        prompt_tokens=int(u.get("inputTokens", 0) or 0) + cache_read + cache_write,
        completion_tokens=int(u.get("outputTokens", 0) or 0),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
        cache_write_tokens=cache_write,
    )
    stop = r.get("stopReason", "end_turn")
    finish = "tool_calls" if calls else _FINISH.get(stop, stop)
    return SimpleNamespace(choices=[SimpleNamespace(message=_Message(text, calls), finish_reason=finish)], usage=usage)


class _Completions:
    def __init__(self, parent: "BedrockClient"):
        self._p = parent

    def create(self, *, model, messages, tools=None, tool_choice="auto", temperature=None, **_ignored):
        assert_us_only(model)
        cache = supports_cache(model)
        system, msgs = to_converse_messages(messages, cache)
        req: dict = {"modelId": model, "messages": msgs}
        if system:
            req["system"] = system
        inf: dict = {"maxTokens": self._p.max_tokens}
        if temperature is not None and not any(f in model for f in _NO_TEMPERATURE_FAMILIES):
            inf["temperature"] = temperature
        req["inferenceConfig"] = inf
        if tools:
            req["toolConfig"] = {"tools": to_converse_tools(tools, cache),
                                 "toolChoice": {"any": {}} if tool_choice == "required" else {"auto": {}}}
        try:
            r = self._p.runtime.converse(**req)
        except Exception as e:  # noqa: BLE001
            if "maxTokens" in str(e) and self._p.max_tokens > 8192:
                self._p.max_tokens = 8192  # model's output cap is lower than our default
                req["inferenceConfig"]["maxTokens"] = 8192
                r = self._p.runtime.converse(**req)
            else:
                raise
        return from_converse_response(r)


class BedrockClient:
    def __init__(self, region: str | None = None, runtime=None, max_tokens: int | None = None):
        if runtime is None:
            import boto3
            from botocore.config import Config
            runtime = boto3.client(
                "bedrock-runtime", region_name=region or os.environ.get("AGENT_BEDROCK_REGION", "us-east-2"),
                config=Config(read_timeout=180, connect_timeout=30, retries={"max_attempts": 2, "mode": "standard"}))
        self.runtime = runtime
        self.max_tokens = max_tokens or int(os.environ.get("AGENT_BEDROCK_MAX_TOKENS", "16000"))
        self.chat = SimpleNamespace(completions=_Completions(self))

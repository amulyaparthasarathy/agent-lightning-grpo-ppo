# Copyright (c) Microsoft. All rights reserved.
# Adapted from Agent Lightning examples/calc_x/calc_agent.py (MIT).
"""Actual Calc-X trajectory recording; no additional model requests.

Runtime: AutoGen 0.7.5; optional Weave 0.54.1 Agents API. Local JSONL is
always attempted. AGL_WEAVE_PROJECT opts in to sending task/model/tool content
to that project. Trace failures never replace an execution result or reward.
AGL_PHASE labels a run; route_mode distinguishes train/val within the run.
AutoGen is archived; it is retained here to match the released teaching example.
The explicit two-iteration bound, no parallel calls, and output-token cap are
shared policy settings for every phase, not claims about the original paper.

Inspected upstream APIs at microsoft/autogen tag python-v0.7.5:
python/packages/autogen-ext/src/autogen_ext/models/openai/_openai_client.py
python/packages/autogen-ext/src/autogen_ext/tools/mcp/_workbench.py
python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from eval_utils import scalar_are_results_same


def now():
    return datetime.now(timezone.utc).isoformat()


def public_data(value):
    """Serialize observable messages/results; exclude reasoning and logprobs."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude={"thought", "logprobs"})
    elif is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): public_data(v) for k, v in value.items()
                if str(k).lower() not in {"thought", "logprobs", "api_key", "authorization", "headers"}}
    if isinstance(value, (list, tuple)):
        return [public_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Recorder:
    """One file per attempt avoids cross-process JSONL write collisions."""

    def __init__(self, question):
        self.trace_id = uuid4().hex
        self.common = {
            "schema_version": 1, "trace_id": self.trace_id,
            "experiment_id": os.getenv("AGL_EXPERIMENT_ID", "unknown"),
            "wandb_run_id": os.getenv("AGL_WANDB_RUN_ID", "unknown"),
            "case_id": os.getenv("AGL_CASE_ID") or hashlib.sha256(question.encode()).hexdigest()[:16],
            "phase": os.getenv("AGL_PHASE", "unspecified"),
            "policy_version": os.getenv("AGL_POLICY_VERSION", "unknown"),
            "eval_split": os.getenv("EVAL_SPLIT", os.getenv("AGL_DATA_SPLIT", "unknown")),
            "model_profile": os.getenv("AGL_MODEL_PROFILE", "unspecified"),
            "synthetic_fixture": False, "model_alias": "auto", "model_version": None,
            "rollout_id": None, "attempt_id": None, "route_mode": None,
        }
        # The gateway documents this route. Never record its host, query, or key.
        parts = urlsplit(os.getenv("AGL_OPENAI_BASE_URL", "")).path.strip("/").split("/")
        for i in range(max(0, len(parts) - 8)):
            p = parts[i:i + 9]
            if len(p) == 9 and p[:2] == ["proxy", "rollout"] and p[3] == "attempt" and p[5] == "mode" and p[7:] == ["openai", "v1"]:
                self.common.update(rollout_id=p[2], attempt_id=p[4], route_mode=p[6])
                break
        self.secrets = [os.environ[k] for k in ("AGL_KEY", "OPENAI_API_KEY", "WANDB_API_KEY", "HF_TOKEN") if os.environ.get(k)]
        self.log = None
        self.weave = None
        self.weave_status = "not_requested"
        self.warnings = 0
        self.counts = {"llm_calls": 0, "tool_calls": 0, "tool_errors": 0}
        self.episode_span_id = None
        try:
            directory = Path(os.getenv("AGL_TRACE_DIR", "traces"))
            directory.mkdir(parents=True, exist_ok=True)
            self.log = (directory / f"trajectory-{self.trace_id}.jsonl").open("x", encoding="utf-8")
        except Exception as exc:
            self.warning("local_trace_open", exc)

    def clean(self, value):
        value = public_data(value)
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {k: self.clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        return value

    def warning(self, operation, exc):
        self.warnings += 1
        # Exception messages can contain request details; stderr prints type only.
        print(f"Trace warning: {operation}: {type(exc).__name__}; execution continues.", file=sys.stderr)

    def safe(self, operation, action):
        try:
            return action()
        except Exception as exc:
            self.warning(operation, exc)
            return None

    def write(self, record):
        if self.log is not None:
            self.safe("local_trace_write", lambda: self.log.write(json.dumps(self.clean(record)) + "\n"))
            self.safe("local_trace_flush", self.log.flush)

    def init_weave(self):
        project = os.getenv("AGL_WEAVE_PROJECT", "")
        if not project:
            return
        try:
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", project):
                raise ValueError("AGL_WEAVE_PROJECT must be TEAM/PROJECT")
            import weave
            from weave.conversation import Message, ToolCallPart, Usage
            for name in ("start_conversation", "start_llm", "start_tool", "finish"):
                if not callable(getattr(weave, name, None)):
                    raise RuntimeError("Weave Agents API missing; use weave==0.54.1")
            # Explicit spans below are sufficient; avoid duplicate SDK auto-patching.
            weave.init(project, settings={"implicitly_patch_integrations": False})
            self.weave, self.Message, self.ToolCallPart, self.Usage = weave, Message, ToolCallPart, Usage
            self.weave_status = "initialized_delivery_unverified"
        except Exception as exc:
            self.weave_status = "initialization_failed_local_fallback"
            self.warning("weave_init", exc)

    @contextmanager
    def remote(self, factory):
        manager = span = None
        try:
            manager = factory() if self.weave else None
            span = manager.__enter__() if manager is not None else None
        except Exception as exc:
            self.warning("weave_span_enter", exc)
            manager = None
        try:
            yield span
        except BaseException as exc:
            if span is not None and callable(getattr(span, "record_error", None)):
                # Record the error without sending exception request/header details.
                self.safe("weave_error", lambda: span.record_error(RuntimeError(type(exc).__name__)))
            raise
        finally:
            if manager is not None:
                self.safe("weave_span_exit", lambda: manager.__exit__(None, None, None))

    def messages(self, items):
        output = []
        for item in self.clean(items):
            kind, content = item.get("type", ""), item.get("content", "")
            if kind == "FunctionExecutionResultMessage":
                output.extend(self.Message.tool_result(r.get("call_id", ""), r.get("content", "")) for r in content)
            elif kind == "AssistantMessage" and isinstance(content, list):
                output.append(self.assistant(content))
            else:
                role = {"SystemMessage": "system", "AssistantMessage": "assistant"}.get(kind, "user")
                output.append(self.Message(role=role, content=content if isinstance(content, str) else json.dumps(content)))
        return output

    def assistant(self, content):
        content = self.clean(content)
        if isinstance(content, str):
            return self.Message.assistant(content)
        return self.Message.assistant(tool_calls=[self.ToolCallPart(
            id=c["id"], name=c["name"], arguments=c["arguments"]) for c in content])

    @contextmanager
    def stage(self, kind, name, inputs, factory):
        record = dict(self.common, kind=kind, name=name, span_id=uuid4().hex,
                      inputs=inputs, outputs=None, execution_status="ok", exception=None)
        record["parent_span_id"] = self.episode_span_id
        if kind == "episode":
            self.episode_span_id = record["span_id"]
        with self.remote(factory) as span:
            if span is not None:
                self.safe("weave_attributes", lambda: span.set_attributes({"local.span_id": record["span_id"]}))
            record["started_at"] = now()
            started = perf_counter()
            try:
                yield record, span
            except BaseException as exc:
                record["execution_status"] = "exception"
                record["exception"] = {"type": type(exc).__name__}
                raise
            finally:
                record["ended_at"] = now()
                record["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
                record["timing_scope"] = "wall time including local instrumentation; not GPU kernel time"
                if kind == "episode":
                    record["counts"] = dict(self.counts)
                self.write(record)

    def close(self):
        if self.weave:
            try:
                self.weave.finish()
                self.weave_status = ("finish_returned_with_trace_warnings_delivery_unverified" if self.warnings
                                     else "finish_returned_delivery_unverified")
            except Exception as exc:
                self.weave_status = "finish_failed_local_retained"
                self.warning("weave_finish", exc)
        self.write(dict(self.common, kind="trace_status", weave_status=self.weave_status,
                        trace_warnings=self.warnings, **self.counts))
        if self.log:
            self.safe("local_trace_close", self.log.close)


class ObservedClient(OpenAIChatCompletionClient):
    def __init__(self, *, recorder, **kwargs):
        super().__init__(**kwargs)
        self.recorder = recorder

    async def create(self, messages, **kwargs):
        r = self.recorder
        r.counts["llm_calls"] += 1
        inputs = r.safe("serialize_llm_input", lambda: {
            "messages": public_data(messages), "tools": public_data(kwargs.get("tools", []))})
        with r.stage("llm", "model_generation", inputs,
                     lambda: r.weave.start_llm(model="auto", provider_name="vllm")) as (record, span):
            if span is not None:
                r.safe("weave_llm_input", lambda: setattr(span, "input_messages", r.messages(messages)))
            started = perf_counter()
            try:
                result = await super().create(messages, **kwargs)
            finally:
                record["operation_elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            content = public_data(result.content)
            usage = {"prompt_tokens": result.usage.prompt_tokens, "completion_tokens": result.usage.completion_tokens}
            # AutoGen substitutes zeros if server usage is absent. Do not call them measured zeros.
            known_usage = usage if any(usage.values()) else None
            record["outputs"] = {"content": content, "finish_reason": result.finish_reason,
                                 "usage": known_usage, "usage_source": "AutoGen server-reported; zero/absent indistinguishable",
                                 "cost_usd": None}
            if span is not None:
                def record_remote():
                    span.output_messages = [r.assistant(content)]
                    span.set_attributes({"usage_available": known_usage is not None})
                    if known_usage:
                        span.usage = r.Usage(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"])
                r.safe("weave_llm_output", record_remote)
            return result


class ObservedWorkbench(McpWorkbench):
    def __init__(self, server_params, *, recorder):
        super().__init__(server_params)
        self.recorder = recorder

    async def call_tool(self, name, arguments=None, cancellation_token=None, call_id=None):
        r = self.recorder
        r.counts["tool_calls"] += 1
        with r.stage("tool", name, {"arguments": arguments, "tool_call_id": call_id},
                     lambda: r.weave.start_tool(name=name, arguments=json.dumps(r.clean(arguments or {})),
                                               tool_call_id=call_id or "")) as (record, span):
            started = perf_counter()
            try:
                result = await super().call_tool(name, arguments, cancellation_token, call_id)
            except BaseException:
                r.counts["tool_errors"] += 1
                raise
            finally:
                record["operation_elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            record["outputs"] = public_data(result)
            if result.is_error:
                record["execution_status"] = "tool_reported_error"
                r.counts["tool_errors"] += 1
            if span is not None:
                def record_remote():
                    span.result = r.clean(result)
                    span.set_attributes({"tool.returned_is_error": result.is_error,
                                         "tool.exception_raised": False})
                    if result.is_error:
                        span.record_error(RuntimeError("MCP returned is_error=True; no exception escaped call_tool"))
                r.safe("weave_tool_output", record_remote)
            return result


class Agent:
    async def run(self):
        question, expected = os.environ["QUESTION"], os.environ["RESULT"]
        key, event_url = os.environ["AGL_KEY"], os.environ["AGL_EVENT_URL"]
        r = Recorder(question)
        r.init_weave()
        model_client = None
        prompt = (f"{question} Output the answer when you are ready. The answer should be surrounded by "
                  "three sharps (`###`), in the form of ### ANSWER: <answer> ###.")
        try:
            with r.remote(lambda: r.weave.start_conversation(
                agent_name="cs329z-calc-x", conversation_id=r.trace_id,
                conversation_name=f"{r.common['phase']} {r.common['case_id']}",
                attributes={k: v for k, v in r.common.items() if v is not None})) as conversation:
                with r.stage("episode", "calc_x", {"question": question},
                             lambda: conversation.start_turn(user_message=r.clean(prompt)) if conversation else None) as (record, turn):
                    params = StdioServerParams(command=sys.executable, args=["-m", "mcp_server_calculator"],
                                               read_timeout_seconds=120)
                    async with ObservedWorkbench(params, recorder=r) as workbench:
                        model_client = ObservedClient(
                            recorder=r, model="auto", base_url=os.environ["AGL_OPENAI_BASE_URL"], api_key=key,
                            model_info={"vision": False, "function_calling": True, "json_output": False,
                                        "family": ModelFamily.UNKNOWN, "structured_output": False},
                            max_retries=6, parallel_tool_calls=False,
                            max_tokens=int(os.getenv("AGL_MAX_TOKENS", "1024")))
                        agent = AssistantAgent(name="calc", model_client=model_client, workbench=workbench,
                                               reflect_on_tool_use=True, max_tool_iterations=2)
                        response = await asyncio.wait_for(agent.run(task=prompt), timeout=300.0)
                    raw_answer = str(response.messages[-1].content)
                    match = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", raw_answer)
                    answer = match.group(1).strip() if match else raw_answer.strip()
                    # Ground truth enters only the independent original Calc-X scorer.
                    reward = 1.0 if scalar_are_results_same(answer, expected, 1e-2) else 0.0
                    record["outputs"] = {"raw_answer": raw_answer, "answer": answer, "reward": reward,
                                         "answer_correct": bool(reward), "reward_posted": False, **r.counts}
                    if turn is not None:
                        def record_turn():
                            turn.output_messages = [r.Message.assistant(r.clean(raw_answer))]
                            turn.set_attributes({"eval.reward": reward, "eval.answer_correct": bool(reward), **r.counts})
                        r.safe("weave_turn_output", record_turn)
                    httpx.post(event_url, json={"event_type": "reward", "data": {"value": reward}},
                               headers={"Authorization": f"Bearer {key}"}, timeout=10.0).raise_for_status()
                    record["outputs"]["reward_posted"] = True
        finally:
            if model_client is not None:
                try:
                    await model_client.close()
                except Exception as exc:
                    r.warning("model_client_close", exc)
            r.close()


if __name__ == "__main__":
    asyncio.run(Agent().run())

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


try:
    from .observed_runtime import Recorder, ObservedClient, ObservedWorkbench
except ImportError:
    from observed_runtime import Recorder, ObservedClient, ObservedWorkbench


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

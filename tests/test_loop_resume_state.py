from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

import agent_core.runtime.loop as loop_api
import agent_core.runtime.loop.agent_loop as kernel
from agent_core.execution_context import (
    ExecutionScope,
    get_current_execution_scope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from agent_core.llm import LLMResponse
from agent_core.loop_types import Intervention, LoopConfig, LoopPolicy, ToolCallIntervention
from agent_core.messages import assistant_msg, system_msg, tool_msg, user_msg
from agent_core.runtime.loop.agent_loop import AgentLoopHooks, TurnToolSet, run_agent_loop
from agent_core.runtime.loop.tool_exec import ToolExecutionHooks


@pytest.fixture(autouse=True)
def reject_swallowed_assertions(caplog):
    """Kernel callbacks isolate exceptions; a failed test assertion must still fail."""
    caplog.set_level("DEBUG", logger="agent_core.loop_types")
    yield
    for record in caplog.records:
        assert not (record.exc_info and isinstance(record.exc_info[1], AssertionError))
        assert not any(isinstance(arg, AssertionError) for arg in record.args)


class ScriptLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def chat(self, messages, **_kwargs):
        self.requests.append(deepcopy(messages))
        assert self.responses, "unexpected model request"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, *_args, **_kwargs):
        raise AssertionError("streaming was not requested")


class EchoTool:
    name = "echo"

    def __init__(self):
        self.executed = []

    async def ainvoke(self, args):
        self.executed.append(deepcopy(args))
        return args["value"]

    def to_openai_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": "echo",
            "parameters": {"type": "object"},
        }}


def state(**kwargs):
    # Keep collection working on ef7f74b for the host's red-before-green check.
    cls = getattr(kernel, "LoopResumeState", None)
    assert cls is not None, "R17 LoopResumeState is missing"
    return cls(**kwargs)


def config(**kwargs):
    return LoopConfig(**{
        "max_turns": 8, "max_llm_retries": 1, "retry_wait_fixed": 0,
        "compact_after_turns": 100, "role_id": "current",
        "loop_policy": LoopPolicy(no_tool_behavior="stop"), **kwargs,
    })


def call(call_id="native", value="hello"):
    return {"id": call_id, "name": "echo", "args": {"value": value}}


def tool_response(call_id="native", value="hello"):
    return LLMResponse(content="working", tool_calls=[call(call_id, value)])


async def run(model, *, tool=None, cfg=None, history=None, **kwargs):
    return await run_agent_loop(
        system_prompt="system", user_message="start" if history is None else "",
        llm=model, tools=[tool] if tool is not None else [],
        config=cfg or config(), initial_messages=history, **kwargs,
    )


def summary(result):
    return (result.turns_used, result.tool_calls_count, result.stopped_by, result.final_content)


def stable_metadata(metadata):
    return deepcopy({k: v for k, v in metadata.items() if not k.startswith("_llm_")})


class CounterObserver:
    critical = True

    def __init__(self):
        self.before = []
        self.lifecycle = []

    async def on_loop_start(self, _cfg):
        self.lifecycle.append("start")

    async def on_before_llm(self, ctx):
        self.before.append((ctx.turn, stable_metadata(ctx.metadata)))
        ctx.metadata["requests_seen"] = ctx.metadata.get("requests_seen", 0) + 1

    async def on_loop_end(self, _result):
        self.lifecycle.append("end")


class PendingProbe:
    critical = True

    def __init__(self, model):
        self.model = model
        self.events = []

    async def on_tool_call(self, _ctx, _call):
        self.events.append(("call", len(self.model.requests)))

    async def on_turn_end(self, _ctx):
        self.events.append(("end", len(self.model.requests)))


def equivalence_script():
    return [
        tool_response("one", "a"), tool_response("two", "b"),
        LLMResponse(content="nudge"),
        LLMResponse(content="cut-", finish_reason="length"),
        LLMResponse(content="cut2-", finish_reason="length"),
        tool_response("three", "c"),
        LLMResponse(content="almost"), LLMResponse(content="done"),
    ]


@pytest.mark.asyncio
async def test_every_boundary_resume_matches_continuous_run(monkeypatch):
    """AC-2: actually cancel at all 11 boundaries, then replay or fill tools."""
    monkeypatch.setattr(kernel.time, "perf_counter", lambda: 100.0)
    cfg = config(
        loop_policy=LoopPolicy(no_tool_behavior="nudge"),
        system_addendum_per_call="ADD", system_addendum_min_turn=2,
    )
    model, tool, observer = ScriptLLM(equivalence_script()), EchoTool(), CounterObserver()
    snapshots = []

    async def capture(phase, snapshot, messages):
        snapshots.append((phase, snapshot, deepcopy(messages), len(model.requests)))

    control = await run(model, tool=tool, cfg=cfg, observers=[observer], on_loop_state=capture)
    assert len(snapshots) == 11
    assert [p for p, *_ in snapshots].count("tools") == 3
    assert [any(m.get("content") == "ADD" for m in req) for req in model.requests] == [
        False, False, True, True, True, True, True, True,
    ]
    for index, (phase, snapshot, history, request_count) in enumerate(snapshots):
        assert snapshot.llm_attempts_used == 0
        assert not any(m.get("content") == "ADD" for m in history)
        if phase == "turn":
            assert snapshot.pending_tool_calls == ()
        else:
            assert snapshot.no_tool_retries == snapshot.truncation_continuations == 0
            assert snapshot.truncated_text_parts == ()
        prefix_model, prefix_tool = ScriptLLM(equivalence_script()), EchoTool()
        seen = []

        async def interrupt(p, s, messages, target=index, events=seen):
            events.append((p, s, deepcopy(messages)))
            if len(events) - 1 == target:
                raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await run(prefix_model, tool=prefix_tool, cfg=cfg,
                      observers=[CounterObserver()], on_loop_state=interrupt)
        assert len(prefix_model.requests) == request_count
        assert len(prefix_tool.executed) == snapshot.total_tool_calls
        p, s, messages = seen[-1]
        assert p == phase and s == replace(snapshot, metadata=s.metadata)
        assert stable_metadata(s.metadata) == stable_metadata(snapshot.metadata)
        assert messages == history

        for mode in ("replay", "fill") if phase == "tools" else ("replay",):
            resumed_state, resumed_history, filled = snapshot, deepcopy(history), 0
            if mode == "fill":
                filled = len(snapshot.pending_tool_calls)
                resumed_history.extend(
                    tool_msg(c["args"]["value"], c["id"])
                    for c in snapshot.pending_tool_calls
                )
                resumed_state = replace(snapshot, pending_tool_calls=(),
                                        total_tool_calls=snapshot.total_tool_calls + filled)
            resumed_model = ScriptLLM(equivalence_script()[request_count:])
            resumed_tool, resumed_observer = EchoTool(), CounterObserver()

            probe = PendingProbe(resumed_model)
            result = await run(
                resumed_model, tool=resumed_tool, cfg=cfg, history=resumed_history,
                resume_state=resumed_state, observers=[resumed_observer, probe],
            )
            assert resumed_model.requests == model.requests[request_count:]
            assert resumed_observer.before == observer.before[request_count:]
            assert result.messages == control.messages
            assert summary(result) == summary(control)
            assert len(prefix_tool.executed) + filled + len(resumed_tool.executed) == len(tool.executed)
            assert resumed_observer.lifecycle == ["start", "end"]
            if phase == "tools" and mode == "replay":
                assert probe.events[:2] == [("call", 0), ("end", 0)]

    # AC-1's kernel part: omit both parameters, and compare to the no-op hook run.
    plain_model, plain_tool, plain_observer = ScriptLLM(equivalence_script()), EchoTool(), CounterObserver()
    plain = await run(plain_model, tool=plain_tool, cfg=cfg, observers=[plain_observer])
    assert plain_model.requests == model.requests
    assert plain.messages == control.messages
    assert summary(plain) == summary(control)
    assert stable_metadata(plain.metadata) == stable_metadata(control.metadata)
    assert plain_observer.before == observer.before


def test_state_defaults_frozen_and_exports():
    snapshot = state()
    assert snapshot == state(turn=0, attempts=0, total_tool_calls=0, no_tool_retries=0,
                             truncation_continuations=0, truncated_text_parts=(), metadata={},
                             llm_attempts_used=0, pending_tool_calls=())
    assert loop_api.LoopResumeState is type(snapshot)
    assert loop_api.LoopStateHook is kernel.LoopStateHook
    assert {"LoopResumeState", "LoopStateHook"} <= set(loop_api.__all__)
    with pytest.raises(FrozenInstanceError):
        snapshot.turn = 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text_call", [False, True], ids=["native", "text"])
async def test_handoff_order_ids_and_shallow_snapshots(text_call):
    """AC-3: final parsed IDs, event order, and detached shallow containers."""
    events, snapshots, contexts = [], [], []
    observed_calls = []
    tool = EchoTool()
    response = (LLMResponse(content='<tool_call>{"tool":"echo","args":{"value":"hello"}}</tool_call>')
                if text_call else tool_response())

    class Observer:
        critical = True

        async def on_before_llm(self, ctx):
            events.append("before")
            ctx.metadata["marker"] = ctx.turn

        async def on_llm_input(self, _ctx):
            events.append("input")

        async def on_llm_response(self, _ctx):
            events.append("response")

        async def on_tool_call(self, ctx, c):
            events.append("call")
            contexts.append(ctx)
            observed_calls.append(deepcopy(c))
            assert not tool.executed
            ctx.metadata["after_handoff"] = True

        async def on_tool_result(self, _ctx, _result):
            events.append("result")

        async def on_turn_end(self, _ctx):
            events.append("end")

    async def capture(phase, snapshot, messages):
        events.append(phase)
        snapshots.append((phase, snapshot, messages))
        assert not tool.executed

    # Only the first two handoffs precede execution; retain later snapshots separately.
    async def hook(phase, snapshot, messages):
        if len(snapshots) < 2:
            await capture(phase, snapshot, messages)
        else:
            events.append(phase)
            snapshots.append((phase, snapshot, messages))

    result = await run(ScriptLLM([response, LLMResponse(content="done")]), tool=tool,
                       observers=[Observer()], on_loop_state=hook)
    assert events == ["turn", "before", "input", "response", "tools", "call", "result",
                      "end", "turn", "before", "input", "response"]
    first, tools_snapshot, next_turn = [s for _, s, _ in snapshots]
    assert (first.turn, first.attempts) == (0, 0)
    assert (tools_snapshot.turn, tools_snapshot.attempts, tools_snapshot.total_tool_calls) == (1, 1, 0)
    assert (next_turn.turn, next_turn.attempts, next_turn.total_tool_calls) == (1, 1, 1)
    assert tools_snapshot.pending_tool_calls == tuple(observed_calls)
    assert tools_snapshot.pending_tool_calls[0] is not contexts[0].tool_calls[0]
    assert tools_snapshot.pending_tool_calls[0]["args"] is contexts[0].tool_calls[0]["args"]
    assert "after_handoff" not in tools_snapshot.metadata
    assert "marker" not in first.metadata
    assert tools_snapshot.metadata["marker"] == 1
    assert len(snapshots[0][2]) == 2
    assert len(snapshots[1][2]) == 3
    assert snapshots[1][2][-1] is result.messages[2]
    final_id = tools_snapshot.pending_tool_calls[0]["id"]
    assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == [final_id]
    assert final_id != "" and (text_call or final_id == "native")

    replay_tool, replay_model, replay_events = EchoTool(), ScriptLLM([LLMResponse(content="done")]), []

    class ReplayObserver:
        critical = True

        async def on_tool_call(self, ctx, c):
            replay_events.append("call")
            assert not replay_model.requests
            assert c == observed_calls[0]
            assert ctx.ai_text == snapshots[1][2][-1]["content"]
            assert ctx.usage is None and ctx.thinking == "" and ctx.thinking_blocks == []

        async def on_tool_result(self, _ctx, _result):
            replay_events.append("result")

        async def on_turn_end(self, _ctx):
            replay_events.append("end")
            assert not replay_model.requests

    async def replay_hook(phase, snapshot, _messages):
        replay_events.append(phase)
        if phase == "tools":
            assert not replay_model.requests and not replay_tool.executed
            assert snapshot.pending_tool_calls == tools_snapshot.pending_tool_calls
            assert snapshot.pending_tool_calls[0] is not tools_snapshot.pending_tool_calls[0]
        else:
            assert replay_events == ["tools", "call", "result", "end", "turn"]

    replay = await run(replay_model, tool=replay_tool, history=snapshots[1][2],
                       resume_state=tools_snapshot, observers=[ReplayObserver()], on_loop_state=replay_hook)
    assert replay.messages == result.messages
    assert summary(replay) == summary(result)
    assert [m["tool_call_id"] for m in replay.messages if m["role"] == "tool"] == [final_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["nudge", "truncate", "response_retry", "turn_end_retry"])
async def test_continuation_paths_hand_out_updated_counters(path):
    snapshots = []

    class RetryOnce:
        critical = True
        done = False

        async def retry(self, _ctx):
            if not self.done:
                self.done = True
                return Intervention(continue_to_next_turn=True)
            return None

        async def on_llm_response(self, ctx):
            if path == "response_retry":
                return await self.retry(ctx)
            return None

        async def on_turn_end(self, ctx):
            if path == "turn_end_retry":
                return await self.retry(ctx)
            return None

    async def hook(phase, s, messages):
        snapshots.append((phase, s, deepcopy(messages)))

    first = {"nudge": LLMResponse(content="nudge"),
             "truncate": LLMResponse(content="cut", finish_reason="length"),
             "response_retry": tool_response(), "turn_end_retry": tool_response()}[path]
    result = await run(ScriptLLM([first, LLMResponse(content="done")]), tool=EchoTool(),
                       cfg=config(loop_policy=LoopPolicy(no_tool_behavior="nudge"),
                                  no_tool_max_retries=2 if path == "nudge" else 1),
                       observers=[RetryOnce()], on_loop_state=hook)
    assert result.stopped_by == "no_tool"
    turns = [s for p, s, _ in snapshots if p == "turn"]
    assert len(turns) == 2
    s = turns[1]
    expected = {"nudge": (1, 1, 0, 1, 0, ()), "truncate": (0, 1, 0, 0, 1, ("cut",)),
                "response_retry": (0, 1, 0, 0, 0, ()), "turn_end_retry": (0, 1, 1, 0, 0, ())}[path]
    assert (s.turn, s.attempts, s.total_tool_calls, s.no_tool_retries,
            s.truncation_continuations, s.truncated_text_parts) == expected
    assert [p for p, *_ in snapshots].count("tools") == (path == "turn_end_retry")


@pytest.mark.asyncio
@pytest.mark.parametrize("intervention", [Intervention(stop_reason="observer_stop"),
                                           Intervention(skip_tool_execution=True)], ids=["stop", "skip"])
async def test_suppressed_tool_batch_has_no_handoff(intervention):
    phases = []

    class Observer:
        critical = True

        async def on_llm_response(self, _ctx):
            return intervention

    async def hook(phase, _s, _messages):
        phases.append(phase)

    result = await run(ScriptLLM([tool_response()]), tool=EchoTool(), cfg=config(max_turns=1),
                       observers=[Observer()], on_loop_state=hook)
    assert phases == ["turn"]
    assert result.tool_calls_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["turns", "attempts", "nudge", "truncation_exhausted",
                                   "truncation_tail", "addendum"])
async def test_restored_budgets_and_fragments_against_history_only(case):
    """AC-4: each restored control has an explicit legacy contrast."""
    cfg = config(max_turns=3, loop_policy=LoopPolicy(no_tool_behavior="nudge"))
    states = {
        "turns": {"turn": 3, "total_tool_calls": 7},
        "attempts": {"turn": 1, "attempts": 203, "total_tool_calls": 7},
        "nudge": {"no_tool_retries": 1},
        "truncation_exhausted": {"truncation_continuations": 2, "truncated_text_parts": ("old-",)},
        "truncation_tail": {"truncation_continuations": 1, "truncated_text_parts": ("old-",)},
        "addendum": {"turn": 2},
    }
    if case == "addendum":
        cfg.system_addendum_per_call = "ADD"
        cfg.system_addendum_min_turn = 2
    if case in ("truncation_tail", "addendum"):
        cfg.loop_policy = LoopPolicy(no_tool_behavior="stop")
    responses = ([LLMResponse(content="new-", finish_reason="length"),
                  LLMResponse(content="tail"), LLMResponse(content="tail")]
                 if case == "truncation_exhausted" else [LLMResponse(content="tail")] * 2)
    history = [system_msg("system"), user_msg("start"), assistant_msg("old-")]
    resumed_model, legacy_model = ScriptLLM(responses), ScriptLLM(responses)
    result = await run(resumed_model, cfg=cfg, history=history, resume_state=state(**states[case]))
    legacy = await run(legacy_model, cfg=cfg, history=history)
    if case in ("turns", "attempts"):
        assert resumed_model.requests == []
        assert result.stopped_by == {"turns": "max_turns", "attempts": "max_attempts"}[case]
        assert result.turns_used == states[case]["turn"] and result.tool_calls_count == 7
        assert legacy_model.requests and legacy.tool_calls_count == 0
    elif case == "nudge":
        assert len(resumed_model.requests) == 1 and len(legacy_model.requests) == 2
        assert result.stopped_by == legacy.stopped_by == "no_tool"
    elif case == "truncation_exhausted":
        assert len(resumed_model.requests) == 1 and len(legacy_model.requests) == 3
        assert result.stopped_by == "response_truncated" and result.final_content == "old-new-"
        assert legacy.stopped_by == "no_tool" and legacy.final_content == "tail"
    elif case == "truncation_tail":
        assert result.final_content == "old-tail" and legacy.final_content == "tail"
    else:
        assert resumed_model.requests[0][-1]["content"] == "ADD"
        assert not any(m.get("content") == "ADD" for m in legacy_model.requests[0])
        assert not any(m.get("content") == "ADD" for m in result.messages)


@pytest.mark.asyncio
async def test_pending_batch_at_turn_limit_still_executes():
    cfg = config(max_turns=1)
    original_tool = EchoTool()
    control = await run(ScriptLLM([tool_response()]), tool=original_tool, cfg=cfg)
    model, tool = ScriptLLM([]), EchoTool()
    history = control.messages[:-1]
    result = await run(model, tool=tool, cfg=cfg, history=history,
                       resume_state=state(turn=1, attempts=1, pending_tool_calls=(call(),)))
    assert model.requests == []
    assert result.messages == control.messages
    assert summary(result) == summary(control)
    assert result.stopped_by == "max_turns" and result.tool_calls_count == 1
    assert tool.executed == original_tool.executed


@pytest.mark.asyncio
@pytest.mark.parametrize("used,requests,count,indices", [
    (0, 2, 2, [1, 2]), (1, 1, 2, [2]), (2, 0, 2, []), (3, 0, 3, []),
])
async def test_first_llm_call_attempt_budget(used, requests, count, indices):
    """AC-5: exhausted credit, shifted IDs/events, and input counter origin."""
    events, inputs = [], []

    class Observer:
        critical = True

        async def on_llm_input(self, ctx):
            inputs.append(dict(ctx.metadata))

        async def on_llm_attempt(self, ctx):
            assert ctx.attempt_id.endswith(f"_attempt_{ctx.attempt_index:02d}")
            if ctx.phase == "finished":
                events.append(ctx.attempt_index)
                assert ctx.metadata["_llm_attempt_index"] == ctx.attempt_index
                assert ctx.metadata["_llm_attempt_id"] == ctx.attempt_id

    model = ScriptLLM([RuntimeError("503 overloaded")] * 2)
    result = await run(model, cfg=config(max_llm_retries=2), observers=[Observer()],
                       resume_state=state(llm_attempts_used=used))
    assert len(model.requests) == requests
    assert result.stopped_by == "llm_error" and result.metadata["llm_error_reason"] == "exhausted"
    assert result.metadata["_llm_attempt_count"] == count and events == indices
    assert inputs[0]["_llm_attempt_count"] == used
    assert inputs[0]["_llm_attempt_index"] == used + 1
    assert inputs[0]["_llm_attempt_id"].endswith(f"_attempt_{used + 1:02d}")


@pytest.mark.asyncio
async def test_llm_attempt_credit_only_affects_first_model_call():
    events = []

    class Observer:
        critical = True

        async def on_llm_attempt(self, ctx):
            if ctx.phase == "finished":
                events.append((ctx.turn, ctx.attempt_index))

    model = ScriptLLM([tool_response(), RuntimeError("503 overloaded"), RuntimeError("503 overloaded")])
    result = await run(model, tool=EchoTool(), cfg=config(max_llm_retries=2),
                       resume_state=state(llm_attempts_used=1), observers=[Observer()])
    assert len(model.requests) == 3
    assert events == [(1, 2), (2, 1), (2, 2)]
    assert result.stopped_by == "llm_error" and result.metadata["_llm_attempt_count"] == 2


@pytest.mark.asyncio
async def test_restored_metadata_is_shallow_and_role_is_current():
    source = {"role_id": "old", "custom": 7, "nested": []}

    class Observer:
        critical = True

        async def on_before_llm(self, ctx):
            assert ctx.metadata["role_id"] == "current" and ctx.metadata["custom"] == 7
            assert ctx.metadata is not source and ctx.metadata["nested"] is source["nested"]
            ctx.metadata["custom"] = 8
            ctx.metadata["nested"].append("shared")

    await run(ScriptLLM([LLMResponse(content="done")]), observers=[Observer()],
              resume_state=state(metadata=source))
    assert source == {"role_id": "old", "custom": 7, "nested": ["shared"]}


@pytest.mark.asyncio
async def test_last_handoff_before_model_error_does_not_retain_error():
    snapshots = []

    async def hook(_phase, s, messages):
        snapshots.append((s, deepcopy(messages)))

    failed = await run(ScriptLLM([RuntimeError("503 overloaded")]), on_loop_state=hook)
    assert failed.stopped_by == "llm_error" and "llm_error" in failed.metadata
    saved, history = snapshots[-1]
    assert "llm_error" not in saved.metadata

    class Observer:
        critical = True

        async def on_before_llm(self, ctx):
            assert "llm_error" not in ctx.metadata and "llm_error_reason" not in ctx.metadata

    resumed = await run(ScriptLLM([LLMResponse(content="done")]), history=history,
                        resume_state=saved, observers=[Observer()])
    assert resumed.stopped_by == "no_tool" and "llm_error" not in resumed.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_phase", ["turn", "tools"])
async def test_hook_exception_warns_once_and_loop_continues(caplog, monkeypatch, failing_phase):
    monkeypatch.setattr(kernel.time, "perf_counter", lambda: 100.0)
    calls = []

    async def hook(phase, _s, _messages):
        calls.append(phase)
        if phase == failing_phase and calls.count(phase) == 1:
            raise RuntimeError("snapshot unavailable")

    model, control_model = ScriptLLM([tool_response(), LLMResponse(content="done")]), ScriptLLM([
        tool_response(), LLMResponse(content="done"),
    ])
    result = await run(model, tool=EchoTool(), on_loop_state=hook)
    control = await run(control_model, tool=EchoTool())
    assert calls == ["turn", "tools", "turn"]
    assert model.requests == control_model.requests and result.messages == control.messages
    assert summary(result) == summary(control)
    assert stable_metadata(result.metadata) == stable_metadata(control.metadata)
    warnings = [r for r in caplog.records if "on_loop_state hook failed" in r.message]
    assert len(warnings) == 1 and "snapshot unavailable" in warnings[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["turn", "tools"])
async def test_hook_cancellation_propagates_and_notifies_observer(phase):
    lifecycle, marker = [], asyncio.CancelledError("handoff cancelled")
    tool = EchoTool()

    class Observer:
        critical = True

        async def on_loop_start(self, _cfg):
            lifecycle.append("start")

        async def on_loop_cancelled(self):
            lifecycle.append("cancelled")

        async def on_loop_end(self, _result):
            lifecycle.append("end")

    async def hook(p, _s, _messages):
        if p == phase:
            raise marker

    with pytest.raises(asyncio.CancelledError) as exc:
        await run(ScriptLLM([tool_response()]), tool=tool,
                  observers=[Observer()], on_loop_state=hook)
    assert exc.value is marker and lifecycle == ["start", "cancelled"]
    assert tool.executed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["turn", "attempts", "total_tool_calls", "no_tool_retries",
                                    "truncation_continuations", "llm_attempts_used"])
async def test_negative_counters_fail_before_any_observer_or_request(field):
    invoked = []

    class Observer:
        critical = True

        def __getattr__(self, name):
            async def callback(*_args):
                invoked.append(name)
            return callback

    model = ScriptLLM([])
    with pytest.raises(ValueError, match=field):
        await run(model, resume_state=state(**{field: -1}), observers=[Observer()])
    assert invoked == [] and model.requests == []


@pytest.mark.asyncio
async def test_turn_handoff_sees_compacted_history():
    snapshots = []

    class CompactEveryTurn:
        def should_compact(self, *_args):
            return True

    class Compactor:
        def compact(self, messages, _keep_recent):
            return [messages[0], user_msg("compacted")]

    async def hook(phase, s, messages):
        snapshots.append((phase, s, deepcopy(messages)))

    result = await run(ScriptLLM([tool_response(), LLMResponse(content="done")]), tool=EchoTool(),
                       cfg=config(compaction_policy=CompactEveryTurn(), compactor=Compactor()),
                       on_loop_state=hook)
    assert snapshots[2][0] == "turn" and snapshots[2][1].turn == 1
    assert snapshots[2][2] == [system_msg("system"), user_msg("compacted")]
    assert result.messages[:2] == snapshots[2][2]


@pytest.mark.asyncio
async def test_tool_rebinding_applies_to_handoff_and_pending_reentry():
    class NewTool(EchoTool):
        name = "new_tool"

    tool, snapshots, resolved_turns, sequence = NewTool(), [], [], []

    def resolve(_cfg, _metadata, turn):
        resolved_turns.append(turn)
        sequence.append(("resolve", turn))
        return TurnToolSet([tool], visible=frozenset({tool.name}))

    async def hook(phase, s, messages):
        snapshots.append((phase, s, deepcopy(messages)))
        sequence.append((phase, s.turn))

    response = LLMResponse(content="new tool", tool_calls=[{
        "id": "new-id", "name": "new_tool", "args": {"value": "new value"},
    }])
    runtime = AgentLoopHooks(resolve_turn_tools=resolve)
    control = await run(ScriptLLM([response, LLMResponse(content="done")]),
                        runtime_hooks=runtime, on_loop_state=hook)
    assert tool.executed == [{"value": "new value"}] and resolved_turns == [1, 2]
    assert sequence == [("turn", 0), ("resolve", 1), ("tools", 1),
                        ("turn", 1), ("resolve", 2)]
    saved = next((s, m) for p, s, m in snapshots if p == "tools")
    assert saved[0].pending_tool_calls[0]["name"] == "new_tool"
    for snapshot, history in [saved, (snapshots[0][1], snapshots[0][2])]:
        tool.executed.clear()
        resolved_turns.clear()
        model = ScriptLLM([LLMResponse(content="done")] if snapshot.pending_tool_calls
                          else [response, LLMResponse(content="done")])
        result = await run(model, history=history, resume_state=snapshot, runtime_hooks=runtime)
        assert resolved_turns == [1, 2] and tool.executed == [{"value": "new value"}]
        assert result.messages == control.messages and summary(result) == summary(control)


@pytest.mark.asyncio
async def test_pending_calls_use_execution_hooks_and_observer_results_once():
    executed_hooks, notified_calls, notified_results, ends = [], [], [], []
    tool, model = EchoTool(), ScriptLLM([])
    calls = (call("a", "a"), call("b", "b"), call("c", "c"))

    async def await_call(coro, name, args, _timeout):
        executed_hooks.append((name, deepcopy(args)))
        coro.close()
        return "persisted-" + args["value"]

    class Observer:
        critical = True

        async def on_tool_call(self, ctx, c):
            notified_calls.append(c["id"])
            assert ctx.turn == 1 and ctx.ai_text == "working" and ctx.usage is None
            if c["id"] == "b":
                return ToolCallIntervention(skip_with_result="observer-b")
            return None

        async def on_tool_result(self, _ctx, result):
            notified_results.append(result.tool_call_id)

        async def on_turn_end(self, ctx):
            ends.append(ctx.turn)

    history = [system_msg("system"), user_msg("start"), assistant_msg("working", tool_calls=list(calls))]
    result = await run(model, tool=tool, cfg=config(max_turns=1), history=history,
                       resume_state=state(turn=1, attempts=1, pending_tool_calls=calls),
                       observers=[Observer()], runtime_hooks=AgentLoopHooks(
                           tool_execution=ToolExecutionHooks(await_call=await_call)))
    assert executed_hooks == [("echo", {"value": "a"}), ("echo", {"value": "c"})]
    assert notified_calls == notified_results == ["a", "b", "c"] and ends == [1]
    assert [m["content"] for m in result.messages if m["role"] == "tool"] == [
        "persisted-a", "observer-b", "persisted-c",
    ]
    assert tool.executed == model.requests == [] and result.tool_calls_count == 3


@pytest.mark.asyncio
async def test_pending_turn_end_retry_decrements_turn_without_spending_attempt():
    phases, completed = [], []

    class Observer:
        critical = True

        async def on_turn_end(self, _ctx):
            return Intervention(continue_to_next_turn=True, inject_messages=["try again"])

    async def hook(phase, s, messages):
        phases.append((phase, s.turn, s.attempts, deepcopy(messages)))

    async def complete(turn, _messages, _metadata):
        completed.append(turn)

    result = await run(ScriptLLM([LLMResponse(content="done")]), tool=EchoTool(),
                       cfg=config(max_turns=2), history=[assistant_msg("working")],
                       resume_state=state(turn=2, attempts=5, pending_tool_calls=(call(),)),
                       observers=[Observer()], on_loop_state=hook, on_turn_complete=complete)
    assert [(p, t, a) for p, t, a, _ in phases] == [("tools", 2, 5), ("turn", 1, 5)]
    assert phases[-1][-1][-1] == user_msg("try again", origin="control")
    assert result.turns_used == 2 and result.tool_calls_count == 1 and completed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_at", ["overflow", "turn_end", "pause"])
async def test_pending_batch_uses_normal_tail_and_stops_before_model(exit_at):
    events = []

    class Observer:
        critical = True

        async def on_turn_end(self, _ctx):
            events.append("end")
            if exit_at == "turn_end":
                return Intervention(stop_reason="observer_stop")
            return None

    async def checkpoint(turn, messages, _metadata):
        assert turn == 1 and messages[-1]["role"] == "tool"
        events.append("checkpoint")

    async def pause():
        events.append("pause")
        return True

    model = ScriptLLM([])
    result = await run(model, tool=EchoTool(), history=[assistant_msg("working")],
                       cfg=config(context_overflow_guard=exit_at == "overflow", max_context_length=1),
                       resume_state=state(turn=1, attempts=1, pending_tool_calls=(call(),)),
                       observers=[Observer()], on_turn_complete=checkpoint, pause_check=pause)
    assert model.requests == []
    assert result.stopped_by == {"overflow": "context_limit_reached", "turn_end": "observer_stop",
                                 "pause": "paused"}[exit_at]
    assert events == {"overflow": [], "turn_end": ["end"],
                      "pause": ["end", "checkpoint", "pause"]}[exit_at]


@pytest.mark.asyncio
@pytest.mark.parametrize("content,expected", [
    ("latest", "latest"), ([{"type": "text", "text": "latest"}], "latest"), (None, ""),
])
async def test_pending_context_uses_latest_assistant_text_and_defaults(content, expected):
    contexts = []

    class Observer:
        critical = True

        async def on_tool_call(self, ctx, _call):
            contexts.append(ctx)

    history = [assistant_msg("older"), {"role": "assistant", "content": content},
               user_msg("intervening")]
    await run(ScriptLLM([]), tool=EchoTool(), cfg=config(max_turns=1), history=history,
              resume_state=state(turn=1, pending_tool_calls=(call(),)), observers=[Observer()])
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx.ai_text == expected and ctx.usage is None and ctx.thinking == ""
    assert ctx.thinking_blocks == ctx.blocked_tool_calls == []
    assert ctx.leaked_reasoning == "" and ctx.tool_schemas_stripped is False
    assert ctx.turn == 1 and ctx.max_turns == 1 and ctx.role_id == "current"


@pytest.mark.asyncio
async def test_handoffs_guarantee_critical_metadata_only():
    gate, snapshots = asyncio.Event(), []

    class Critical:
        critical = True

        async def on_before_llm(self, ctx):
            ctx.metadata["critical_value"] = "ready"

        async def on_loop_end(self, _result):
            gate.set()

    class Background:
        async def on_before_llm(self, ctx):
            await gate.wait()
            ctx.metadata["background_value"] = "late"

    async def hook(phase, s, _messages):
        snapshots.append((phase, s))

    result = await run(ScriptLLM([tool_response(), LLMResponse(content="done")]), tool=EchoTool(),
                       observers=[Critical(), Background()], on_loop_state=hook)
    assert len(snapshots) == 3
    assert snapshots[1][1].metadata["critical_value"] == "ready"
    assert all("background_value" not in s.metadata for _, s in snapshots)
    assert result.metadata["background_value"] == "late"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", [1, 2, 3], ids=["turn1", "turn2", "turn3"])
@pytest.mark.parametrize("stale_turn", [None, 99], ids=["fresh_scope", "stale_scope"])
async def test_pending_scope_turn_matches_continuous_run(boundary, stale_turn):
    class ScopeTool(EchoTool):
        def __init__(self):
            super().__init__()
            self.turns = []

        async def ainvoke(self, args):
            scope = get_current_execution_scope()
            turn = scope.metadata.get("current_turn")
            self.turns.append(turn)
            value = await super().ainvoke(args)
            return f"{value}@{turn}"

    def runtime(resolved):
        def enter(cfg, phase_id, metadata):
            scope = ExecutionScope(task_id=cfg.task_id, phase_id=phase_id,
                                   role_id=cfg.role_id, metadata=metadata)
            return scope, set_current_execution_scope(scope)

        def resolve(_cfg, _metadata, turn):
            scope = get_current_execution_scope()
            resolved.append((turn, scope.metadata.get("current_turn")))

        return AgentLoopHooks(enter_scope=enter, exit_scope=reset_current_execution_scope,
                              resolve_turn_tools=resolve)

    responses = [tool_response("one", "a"), tool_response("two", "b"),
                 tool_response("three", "c"), LLMResponse(content="done")]
    control_model, control_tool, control_resolved = ScriptLLM(responses), ScopeTool(), []
    control_handoffs = []

    async def capture(phase, snapshot, _messages):
        if phase == "tools":
            scope = get_current_execution_scope()
            control_handoffs.append((snapshot.turn, scope.metadata.get("current_turn")))

    control = await run(control_model, tool=control_tool,
                        runtime_hooks=runtime(control_resolved), on_loop_state=capture)
    assert control_tool.turns == [1, 2, 3]
    assert control_resolved == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert control_handoffs == [(1, 1), (2, 2), (3, 3)]

    prefix_model, prefix_tool, saved = ScriptLLM(responses), ScopeTool(), []

    async def interrupt(phase, snapshot, messages):
        if phase == "tools" and snapshot.turn == boundary:
            saved.append((snapshot, deepcopy(messages)))
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run(prefix_model, tool=prefix_tool, runtime_hooks=runtime([]),
                  on_loop_state=interrupt)
    snapshot, history = saved[0]
    resumed_model = ScriptLLM(responses[boundary:])
    resumed_tool, resolved, handoffs = ScopeTool(), [], []

    async def resumed_capture(phase, snapshot, _messages):
        if phase == "tools":
            scope = get_current_execution_scope()
            handoffs.append((snapshot.turn, scope.metadata.get("current_turn")))

    scope_metadata = {} if stale_turn is None else {"current_turn": stale_turn}
    result = await run(resumed_model, tool=resumed_tool, history=history,
                       resume_state=snapshot, runtime_hooks=runtime(resolved),
                       on_loop_state=resumed_capture, scope_metadata=scope_metadata)
    assert resolved == control_resolved[boundary - 1:]
    assert handoffs == control_handoffs[boundary - 1:]
    assert prefix_tool.turns + resumed_tool.turns == control_tool.turns
    assert prefix_tool.executed + resumed_tool.executed == control_tool.executed
    assert prefix_model.requests + resumed_model.requests == control_model.requests
    assert [m for m in result.messages if m["role"] == "tool"] == [
        m for m in control.messages if m["role"] == "tool"
    ]
    assert result.messages == control.messages and summary(result) == summary(control)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries,used,turn,requests,surfaces", [
    (2, None, 0, 1, True),
    (2, 0, 0, 1, True),
    (2, 1, 0, 1, True),
    (2, 2, 0, 0, False),
    (2, 3, 0, 0, False),
    (0, None, 0, 0, True),
    (0, 0, 0, 0, True),
    (0, 1, 0, 0, False),
    (1, 1, 0, 0, False),
    (2, 2, 1, 0, False),
    (2, 3, 2, 0, False),
], ids=[
    "max2_no_resume", "max2_used0", "max2_used1", "max2_used2", "max2_used3",
    "max0_no_resume", "max0_used0", "max0_used1", "max1_used1",
    "max2_used2_turn2", "max2_used3_turn3",
])
async def test_chain_fallback_respects_restored_attempt_budget(
    max_retries, used, turn, requests, surfaces,
):
    error = RuntimeError("503 overloaded")
    model, observer = ScriptLLM([error] * 2), CounterObserver()
    kwargs = {} if used is None else {
        "resume_state": state(turn=turn, attempts=turn, llm_attempts_used=used),
    }
    hooks = AgentLoopHooks(chain_fallback_active=lambda: True)
    exhausted_error = (
        f"call_llm exhausted with no captured exception (max_retries=0, turn={turn + 1})"
    )
    if surfaces:
        with pytest.raises(RuntimeError) as caught:
            await run(model, cfg=config(max_llm_retries=max_retries),
                      observers=[observer], runtime_hooks=hooks, **kwargs)
        if requests:
            assert caught.value is error
        else:
            assert str(caught.value) == exhausted_error
        assert observer.lifecycle == ["start"]
    else:
        result = await run(model, cfg=config(max_llm_retries=max_retries),
                           observers=[observer], runtime_hooks=hooks, **kwargs)
        assert result.stopped_by == "llm_error"
        assert result.metadata["llm_error_reason"] == "exhausted"
        assert result.metadata["llm_error"] == exhausted_error
        assert observer.lifecycle == ["start", "end"]
        assert result.turns_used == turn + 1
    assert len(model.requests) == requests

"""V1 bus resume seams, restoration records, and existing resubmission paths.

All loops are local substitutes. New API names are resolved inside tests so
this file can also be collected against the pre-V1 kernel for red checks.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from agent_core.components import agent_bus
from agent_core.components.agent_bus import AgentBus, SubAgentRuntimeSpec, SubTask
from agent_core.components.agent_bus import models as bus_models
from agent_core.events import EventType
from agent_core.loop_types import AgentLoopResult
from agent_core.messages import assistant_msg, system_msg, user_msg
from agent_core.runtime.loop.agent_loop import LoopResumeState
from agent_core.runtime.loop.message_trimmer import NullTrimmer, TaskBoundaryTrimmer
from agent_core.runtime.registries.agents import AgentRegistry


class RecordingSink:
    """In-memory event store: append and replay in publication order."""

    def __init__(self):
        self.events = []

    async def append(self, task_id, event_type, payload, agent_role="system"):
        self.events.append({
            "task_id": task_id,
            "event_type": event_type,
            "payload": dict(payload),
            "agent_role": agent_role,
        })

    async def replay(self, task_id):
        for event in self.events:
            if event["task_id"] == task_id:
                yield event


@pytest.fixture
async def make_bus():
    buses = []
    llm = object()
    tools = []

    def make():
        bus = AgentBus(
            event_sink=RecordingSink(),
            agent_registry=AgentRegistry(),
            resource_manager=SimpleNamespace(
                get_llm=lambda _: llm,
                get_tools_for_role=lambda _: tools,
            ),
            pause_check_factory=lambda _: None,
            session_activity=False,
        )
        buses.append(bus)
        return bus

    yield make
    for bus in buses:
        await asyncio.wait_for(
            bus.cleanup_task("task-1", cancel_timeout_s=1), timeout=2,
        )
        await asyncio.wait_for(bus.cancel_for_parent("task-1"), timeout=2)


async def _create(bus, runtime_spec=None, name="worker"):
    return await bus.create_session(
        task_id="task-1", name=name, role_id="researcher", system_prompt="system",
        runtime_spec=runtime_spec,
    )


async def _restore(bus, **overrides):
    record = {
        "task_id": "task-1", "name": "worker", "role_id": "researcher",
        "system_prompt": "system", "tools": [], "llm": object(),
        "messages": [user_msg("old task"), assistant_msg("old report")],
        "task_boundaries": [[0, 1]], "total_task_count": 1,
        "next_task_index": 1, "last_report": "old report",
    }
    record.update(overrides)
    return await bus.restore_session(**record)


def _loop_result(kwargs):
    messages = list(kwargs.get("initial_messages") or [])
    if kwargs["user_message"]:
        messages.append(user_msg(kwargs["user_message"]))
    return AgentLoopResult(
        messages=[*messages, assistant_msg("final")], final_content="final",
    )


async def _submit(bus, path, runtime):
    if path == "session":
        sid = await _create(bus, runtime)
        return await bus.submit_task_to_session(sid, "task prompt")
    return await bus.submit(
        "task-1", SubTask("task prompt", system_prompt="system"),
        runtime_spec=runtime,
    )


async def _collect(bus, jid):
    outcome = await bus.collect([jid], timeout=2)
    assert outcome.pending == []
    results = outcome.completed + outcome.failed
    assert len(results) == 1
    assert results[0].job_id == jid
    return results[0]


def test_loop_resume_model_defaults_frozen_export_and_appended_field():
    resume = bus_models.LoopResume()
    assert agent_bus.LoopResume is bus_models.LoopResume
    assert [field.name for field in fields(resume)] == [
        "resume_state", "on_loop_state", "initial_messages", "prefix_count",
    ]
    assert all(getattr(resume, field.name) is None for field in fields(resume))
    with pytest.raises(FrozenInstanceError):
        resume.prefix_count = 4
    assert fields(SubAgentRuntimeSpec)[-1].name == "loop_resume"
    assert SubAgentRuntimeSpec().loop_resume is None


@pytest.mark.parametrize("path", ["session", "one_shot"])
async def test_default_and_none_provider_preserve_exact_loop_kwargs(monkeypatch, make_bus, path):
    calls = []
    provided = []
    config = object()
    observers = []
    model_profile = object()
    history_policy = object()

    async def loop(**kwargs):
        calls.append(kwargs)
        return _loop_result(kwargs)

    def provider(jid, item, index):
        provided.append((jid, item, index))
        return None

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    for mode in ("absent", "explicit_none", "returns_none"):
        runtime = SubAgentRuntimeSpec(
            config_builder=lambda *_: config,
            observers_builder=lambda *_: observers,
            model_profile=model_profile, history_policy=history_policy,
        )
        if mode != "absent":
            runtime.loop_resume = provider if mode == "returns_none" else None
        bus = make_bus()
        jid = await _submit(bus, path, runtime)
        assert (await _collect(bus, jid)).success

    assert len(calls) == 3
    # Compare the actual dictionaries, including every value, not only a
    # hand-picked subset of the optional arguments.
    assert calls[0] == calls[1] == calls[2]
    expected_keys = {
        "system_prompt", "user_message", "llm", "tools", "config", "observers",
        "model_profile", "history_policy", "pause_check", "parser", "runtime_hooks",
    }
    if path == "session":
        expected_keys.add("initial_messages")
        assert calls[0]["initial_messages"] == [system_msg("system")]
    assert set(calls[0]) == expected_keys
    assert calls[0]["system_prompt"] == "system"
    assert calls[0]["user_message"] == "task prompt"
    assert calls[0]["config"] is config
    assert calls[0]["observers"] == observers
    assert calls[0]["model_profile"] is model_profile
    assert calls[0]["history_policy"] is history_policy
    assert len(provided) == 1
    jid, item, index = provided[0]
    assert jid == "task-1.job.1"
    assert item == SubTask("task prompt", system_prompt="system")
    assert index == (1 if path == "session" else 0)


@pytest.mark.parametrize("path", ["session", "one_shot"])
@pytest.mark.parametrize("parts", ["both", "state", "hook", "neither"])
async def test_resume_fields_forwarded_independently(monkeypatch, make_bus, path, parts):
    state = LoopResumeState(turn=3) if parts in ("both", "state") else None

    async def hook(*_):
        pass

    callback = hook if parts in ("both", "hook") else None
    calls = []
    provided = []

    def provider(jid, item, index):
        provided.append((jid, item, index))
        return bus_models.LoopResume(resume_state=state, on_loop_state=callback)

    async def loop(**kwargs):
        calls.append(kwargs)
        return _loop_result(kwargs)

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    jid = await _submit(bus, path, SubAgentRuntimeSpec(loop_resume=provider))
    assert (await _collect(bus, jid)).success
    assert len(calls) == 1
    assert calls[0]["resume_state"] is state
    assert calls[0]["on_loop_state"] is callback
    assert calls[0]["user_message"] == "task prompt"
    if path == "one_shot":
        assert "initial_messages" not in calls[0]
    assert provided == [(
        jid, SubTask("task prompt", system_prompt="system"),
        1 if path == "session" else 0,
    )]


@pytest.mark.parametrize("empty", [False, True])
async def test_one_shot_replacement_is_verbatim_and_ignores_prefix(monkeypatch, make_bus, empty):
    replacement = [] if empty else [system_msg("replacement"), user_msg("already asked")]
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        return _loop_result(kwargs)

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    runtime = SubAgentRuntimeSpec(loop_resume=lambda *_: bus_models.LoopResume(
        initial_messages=replacement, prefix_count=-99,
    ))
    jid = await _submit(bus, "one_shot", runtime)
    assert (await _collect(bus, jid)).success
    assert len(calls) == 1
    assert calls[0]["initial_messages"] is replacement
    assert calls[0]["user_message"] == ""


async def test_compacted_session_resume_absorbs_all_turns_then_eager_trims(monkeypatch, make_bus):
    bus = make_bus()
    old_history = [user_msg("old task"), assistant_msg("old report")]
    prompt = "resume task"
    # Compaction moves the current prompt to slot 2, annotates the old user
    # message, and keeps four prefix entries without preserving their content.
    prefix = [
        system_msg("system"), user_msg("pinned: resume task"),
        user_msg("[background] old task"), old_history[1],
    ]
    before_kill = [assistant_msg(f"pre-kill turn {i}") for i in range(1, 4)]
    replacement = [*prefix, *before_kill]
    after_resume = assistant_msg("resumed turn")
    final = assistant_msg("final report")
    state = LoopResumeState(turn=3)
    snapshots = []
    calls = []
    provided = []

    async def hook(*_):
        pass

    def provider(jid, item, index):
        provided.append((jid, item, index))
        return bus_models.LoopResume(state, hook, replacement, len(prefix))

    def adapter(raw, jid, item):
        session = bus.get_session(sid)
        snapshots.append((list(session.messages), list(session.task_boundaries)))
        return bus_models.SubAgentResult(
            question=item.question, role_id=item.role_id, final_content=raw.final_content,
            success=True, job_id=jid,
        )

    runtime = SubAgentRuntimeSpec(loop_resume=provider, result_adapter=adapter)
    sid = await _restore(bus, messages=old_history, runtime_spec=runtime)

    async def loop(**kwargs):
        calls.append(kwargs)
        return AgentLoopResult(
            messages=[*kwargs["initial_messages"], after_resume, final],
            final_content="final report",
        )

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    jid = await bus.submit_task_to_session(sid, prompt)
    result = await _collect(bus, jid)
    assert result.success
    assert len(calls) == 1
    assert calls[0]["initial_messages"] is replacement
    assert calls[0]["user_message"] == ""
    assert calls[0]["resume_state"] is state
    assert calls[0]["on_loop_state"] is hook
    assert provided == [(jid, SubTask(prompt, system_prompt="system"), 2)]
    absorbed = [*old_history, user_msg(prompt), *before_kill, after_resume, final]
    assert snapshots == [(absorbed, [(0, 1), (2, len(absorbed) - 1)])]
    session = bus.get_session(sid)
    assert session.messages == [*old_history, user_msg(prompt), final]
    assert session.task_boundaries == [(0, 1), (2, 3)]
    assert session.total_task_count == 2
    assert session.next_task_index == 2
    assert session.last_report == "final report"
    assert replacement == [*prefix, *before_kill]


@pytest.mark.parametrize("count,length", [(3, 4), (5, 5), (None, 4), (4, 3)])
async def test_invalid_session_prefix_fails_before_loop_and_closes_boundary(
    monkeypatch, make_bus, count, length,
):
    calls = []

    async def loop(**kwargs):
        calls.append(kwargs)
        return _loop_result(kwargs)

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    sid = await _restore(bus, runtime_spec=SubAgentRuntimeSpec(
        loop_resume=lambda *_: bus_models.LoopResume(
            initial_messages=[user_msg("replacement")] * length, prefix_count=count,
        ),
    ))
    jid = await bus.submit_task_to_session(sid, "resume task")
    result = await _collect(bus, jid)
    assert not result.success
    assert result.error_class == "ValueError"
    assert calls == []
    session = bus.get_session(sid)
    assert session.task_boundaries == [(0, 1), (2, 3)]
    assert session.messages[2] == user_msg("resume task")
    assert session.messages[3]["role"] == "assistant"
    assert session.current_job_id is None


@pytest.mark.parametrize("path", ["session", "one_shot"])
async def test_provider_exception_uses_existing_failure_path(monkeypatch, make_bus, path):
    class ResumeUnavailable(RuntimeError):
        pass

    calls = []

    def provider(*_):
        raise ResumeUnavailable("checkpoint unavailable")

    async def loop(**kwargs):
        calls.append(kwargs)
        return _loop_result(kwargs)

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    jid = await _submit(bus, path, SubAgentRuntimeSpec(loop_resume=provider))
    result = await _collect(bus, jid)
    assert not result.success
    assert result.error_class == "ResumeUnavailable"
    assert result.error == "checkpoint unavailable"
    assert bus.get_job_status(jid) == "failed"
    assert calls == []
    if path == "session":
        session = bus.get_session("task-1::worker")
        assert session.task_boundaries == [(0, 1)]
        assert session.current_job_id is None


async def test_restore_session_fields_and_runtime_defaults(make_bus):
    bus = make_bus()
    messages = [user_msg("old task"), assistant_msg("old report")]
    boundaries = [[0, 1]]
    llm = object()
    tools = [object()]
    trimmer = TaskBoundaryTrimmer()
    runtime = SubAgentRuntimeSpec()
    sid = await _restore(
        bus, messages=messages, task_boundaries=boundaries, total_task_count=7,
        next_task_index=11, last_report="saved report", max_turns=23,
        tool_result_max_chars=456, llm_timeout=78, llm=llm, tools=tools,
        trimmer=trimmer, runtime_spec=runtime,
    )
    session = bus.get_session(sid)
    assert sid == "task-1::worker"
    assert session.session_id == sid
    assert session.task_id == "task-1"
    assert session.name == "worker"
    assert session.role_id == "researcher"
    assert session.system_prompt == "system"
    assert session.messages == messages
    assert session.messages is not messages
    assert session.task_boundaries == [(0, 1)]
    assert isinstance(session.task_boundaries[0], tuple)
    assert session.task_boundaries is not boundaries
    assert session.total_task_count == 7
    assert session.next_task_index == 11
    assert session.last_report == "saved report"
    assert session.max_turns == 23
    assert session.tool_result_max_chars == 456
    assert session.llm_timeout == 78
    assert session.llm is llm
    assert session.tools == tools
    assert session.trimmer is trimmer
    assert session.runtime_spec is runtime
    assert session.current_job_id is None
    assert session.pending_results == []
    assert not session.pending_tasks
    assert not session.activity_events
    assert session.offloaded_history_path is None
    assert session.publisher_no_tool_failures == 0
    assert session.last_task_elapsed_s == 0.0
    messages.clear()
    boundaries[0][1] = None
    assert len(session.messages) == 2
    assert session.task_boundaries == [(0, 1)]


async def test_restore_session_default_options_and_tuple_boundaries(make_bus):
    bus = make_bus()
    sid = await _restore(bus, task_boundaries=[(0, 1)])
    session = bus.get_session(sid)
    assert isinstance(session.trimmer, NullTrimmer)
    assert session.task_boundaries == [(0, 1)]
    empty_sid = await bus.restore_session(
        task_id="task-1", name="empty", role_id="researcher",
        system_prompt="", tools=[], llm=None,
    )
    empty = bus.get_session(empty_sid)
    assert empty.messages == []
    assert empty.task_boundaries == []
    assert empty.total_task_count == 0
    assert empty.next_task_index == 0
    assert empty.last_report == ""
    assert empty.max_turns == 100
    assert empty.tool_result_max_chars is None
    assert empty.llm_timeout is None
    assert empty.runtime_spec is None


@pytest.mark.parametrize("existing", ["created", "restored"])
async def test_restore_rejects_duplicate_without_overwriting(make_bus, existing):
    bus = make_bus()
    sid = await (_create(bus) if existing == "created" else _restore(bus))
    original = bus.get_session(sid)
    before_events = list(bus._event_sink().events)
    with pytest.raises(ValueError, match="already exists"):
        await _restore(bus, system_prompt="different", last_report="replacement")
    assert bus.get_session(sid) is original
    assert bus._event_sink().events == before_events


@pytest.mark.parametrize("boundary", [[0, None], [-1, 1], [0, 2], [2, 2], [1, 0], [0, -1]])
async def test_restore_rejects_open_or_out_of_range_boundaries(make_bus, boundary):
    bus = make_bus()
    with pytest.raises(ValueError):
        await _restore(bus, task_boundaries=[boundary])
    assert bus.get_session("task-1::worker") is None
    assert bus._event_sink().events == []


async def test_create_then_restore_event_order_replayed_from_store(make_bus):
    bus = make_bus()
    created = await _create(bus, name="fresh")
    restored = await _restore(bus)
    events = [event async for event in bus._event_sink().replay("task-1")]
    assert [event["payload"]["trace_type"] for event in events] == [
        "session_created", "session_restored",
    ]
    assert len(events) == 2
    expected_keys = {"trace_type", "session_id", "name", "role_id", "agent", "action", "detail"}
    for event, sid, name, action in zip(
        events, [created, restored], ["fresh", "worker"],
        ["session_created", "session_restored"], strict=True,
    ):
        assert event["event_type"] == EventType.AGENT_ACTION
        assert event["task_id"] == "task-1"
        assert event["agent_role"] == "system"
        payload = event["payload"]
        assert set(payload) == expected_keys
        assert payload["session_id"] == sid
        assert payload["name"] == name
        assert payload["agent"] == name
        assert payload["role_id"] == "researcher"
        assert payload["action"] == action
        assert payload["detail"]


async def test_advance_job_counter_is_global_monotonic_and_nonnegative(monkeypatch, make_bus):
    async def loop(**kwargs):
        return _loop_result(kwargs)

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    bus.advance_job_counter(40)
    jid = await bus.submit("task-1", SubTask("first", system_prompt="system"))
    assert jid == "task-1.job.41"
    assert (await _collect(bus, jid)).success
    bus.advance_job_counter(2)
    sid = await _create(bus)
    jid = await bus.submit_task_to_session(sid, "second")
    assert jid == "task-1.job.42"
    assert (await _collect(bus, jid)).success
    with pytest.raises(ValueError):
        bus.advance_job_counter(-1)
    bus.advance_job_counter(0)
    # The counter is shared even across distinct parent task ids.
    other_sid = await bus.create_session(
        task_id="other", name="worker", role_id="researcher", system_prompt="system",
    )
    jid = await bus.submit_task_to_session(other_sid, "third")
    assert jid == "other.job.43"
    assert (await _collect(bus, jid)).success
    await asyncio.wait_for(bus.cleanup_task("other", cancel_timeout_s=1), timeout=2)


async def test_reserved_job_id_table_receipt_abort_and_cleanup(monkeypatch, make_bus):
    gate = asyncio.Event()
    started = asyncio.Event()

    async def loop(**kwargs):
        if kwargs["user_message"] != "deliver":
            started.set()
            await asyncio.wait_for(gate.wait(), timeout=2)
        return _loop_result(kwargs)

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    sid = await _restore(bus)
    session = bus.get_session(sid)
    delivered = "task-1.job.101"
    assert await bus.submit_task_to_session(sid, "deliver", reserved_job_id=delivered) == delivered
    assert delivered in bus._jobs
    assert bus._jobs[delivered].parent_task_id == "task-1"
    receipt = await bus.wait_any_session("task-1", timeout=2)
    assert receipt is not None
    assert receipt[0] == sid
    assert receipt[1].job_id == delivered
    assert receipt[1].success
    assert delivered in bus._delivered_jobs["task-1"]

    running = "task-1.job.102"
    queued = "task-1.job.103"
    await bus.submit_task_to_session(sid, "abort running", reserved_job_id=running)
    await asyncio.wait_for(started.wait(), timeout=2)
    await bus.submit_task_to_session(sid, "abort queued", reserved_job_id=queued)
    assert bus.get_job_status(queued) == "queued"
    assert await asyncio.wait_for(bus.abort(queued), timeout=2) == "aborted"
    assert not session.pending_tasks
    assert bus._jobs[queued].result.job_id == queued
    assert await asyncio.wait_for(bus.abort(running), timeout=2) == "aborted"
    assert bus._jobs[running].result.job_id == running
    assert session.current_job_id is None
    assert all(end is not None for _, end in session.task_boundaries)

    started.clear()
    cleaning = "task-1.job.104"
    cleaning_queued = "task-1.job.105"
    await bus.submit_task_to_session(sid, "cleanup running", reserved_job_id=cleaning)
    await asyncio.wait_for(started.wait(), timeout=2)
    await bus.submit_task_to_session(sid, "cleanup queued", reserved_job_id=cleaning_queued)
    assert await asyncio.wait_for(
        bus.cleanup_task("task-1", cancel_timeout_s=1), timeout=2,
    ) == 1
    assert bus.get_session(sid) is None
    assert not session.pending_tasks
    for jid in (cleaning, cleaning_queued):
        assert bus.get_job_status(jid) == "aborted"
        assert bus._jobs[jid].result.job_id == jid
        assert not bus._jobs[jid].result.success
    assert bus._jobs[cleaning].task.done()
    gate.set()


async def test_reserved_fifo_resubmission_keeps_order_and_submit_ordinals(monkeypatch, make_bus):
    gate = asyncio.Event()
    started = asyncio.Event()
    executed = []
    provided = []

    async def loop(**kwargs):
        executed.append(kwargs["user_message"])
        started.set()
        await asyncio.wait_for(gate.wait(), timeout=2)
        return _loop_result(kwargs)

    def provider(jid, item, index):
        provided.append((jid, item.question, index))
        return None

    monkeypatch.setattr("agent_core.components.agent_bus.bus.run_agent_loop", loop)
    bus = make_bus()
    sid = await _restore(
        bus, next_task_index=8, runtime_spec=SubAgentRuntimeSpec(loop_resume=provider),
    )
    prompts = ["was running", "queued first", "queued second", "queued third"]
    ids = [f"task-1.job.{n}" for n in (90, 17, 62, 31)]
    for jid, prompt in zip(ids, prompts, strict=True):
        assert await bus.submit_task_to_session(sid, prompt, reserved_job_id=jid) == jid
    await asyncio.wait_for(started.wait(), timeout=2)
    session = bus.get_session(sid)
    assert executed == [prompts[0]]
    assert session.current_job_id == ids[0]
    assert [pending.job_id for pending in session.pending_tasks] == ids[1:]
    assert [bus._jobs[jid].task_index for jid in ids] == [9, 10, 11, 12]
    assert [bus.get_job_status(jid) for jid in ids[1:]] == ["queued"] * 3
    gate.set()
    receipts = []
    for _ in ids:
        receipt = await bus.wait_any_session("task-1", timeout=2)
        assert receipt is not None
        assert receipt[1].success
        receipts.append(receipt[1].job_id)
    # collect waits through the job's finally block (including eager trim).
    assert (await _collect(bus, ids[-1])).success
    assert receipts == ids
    assert executed == prompts
    assert provided == list(zip(ids, prompts, [9, 10, 11, 12], strict=True))
    assert session.next_task_index == 12
    assert session.total_task_count == 5
    assert not session.pending_tasks
    assert session.current_job_id is None

"""Shared terminal completion protocol for AgentBus jobs."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Protocol

from agent_core.components.agent_bus.models import (
    JobEntry,
    JobStamp,
    SubAgentResult,
    SubAgentRuntimeSpec,
    SubAgentSession,
)


class _FinishBus(Protocol):
    _job_finish_locks: dict[str, asyncio.Lock]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


async def await_even_if_cancelled[T](awaitable: Awaitable[T]) -> T:
    """Finish *awaitable* before propagating cancellation to its waiter.

    On Python 3.12+, one ``asyncio.shield`` is not a sufficient completion
    barrier: its waiter still receives ``CancelledError`` while the inner
    task continues. Loop until the inner task is done and prefer its
    exception. A ``CancelledError`` from this helper means the awaitable
    finished successfully.
    """
    inner = asyncio.ensure_future(awaitable)
    cancelled = False
    while not inner.done():
        try:
            await asyncio.shield(inner)
        except asyncio.CancelledError:
            cancelled = True
    result = inner.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def wait_running_or_detach(
    task: asyncio.Task[Any],
    timeout: float,
) -> bool:
    """Return True if *task* finished within *timeout*.

    On timeout this must not cancel *task*. The job may be blocked inside
    ``await_even_if_cancelled`` (WAL ``to_thread``); ``asyncio.wait_for``
    cancels its awaitable and then waits for that cancellation, so awaiting
    the job directly would hang until fsync returns.

    A job exception means the task is already done: return True so cleanup
    can durable-abort a non-terminal status without re-raising that exception.
    ``CancelledError`` here belongs to the waiter.
    """
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return True
    except TimeoutError:
        return task.done()
    except asyncio.CancelledError:
        return task.done()
    except Exception:
        return True


def stamp_job_clocks(
    entry: JobEntry,
    now_monotonic: float,
    now_utc: str,
) -> JobStamp:
    started_at = entry.started_at or entry.submitted_at or now_monotonic
    return JobStamp(
        task_index=max(0, int(entry.task_index)),
        started_at=(entry.started_at_utc or entry.submitted_at_utc or now_utc),
        finished_at=now_utc,
        duration_ms=max(0, int((now_monotonic - started_at) * 1000)),
    )


def publish_entry(entry: JobEntry, result: SubAgentResult) -> None:
    entry.result = result
    entry.completed_at = time.monotonic()
    if result.error == "aborted" or result.error_class == "CancelledError":
        entry.status = "aborted"
    elif result.success:
        entry.status = "completed"
    else:
        entry.status = "failed"


def enqueue_pending_result(
    session: SubAgentSession,
    result: SubAgentResult,
) -> None:
    if any(item.job_id == result.job_id for item in session.pending_results):
        return
    session.pending_results.append(result)


def terminal_task_result(
    entry: JobEntry,
    task: asyncio.Task[SubAgentResult],
) -> tuple[SubAgentResult, bool]:
    if entry.result is not None:
        return entry.result, False
    if task.cancelled():
        error = "sub-agent task was cancelled before publishing a report"
        error_class = "CancelledError"
    else:
        try:
            candidate = task.result()
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            error_class = type(exc).__name__
        else:
            if isinstance(candidate, SubAgentResult):
                return candidate, False
            error = "sub-agent task ended without a report"
            error_class = "MissingSubAgentResult"
    return SubAgentResult(
        question=entry.item.question,
        role_id=entry.item.role_id,
        final_content="",
        success=False,
        error=error,
        error_class=error_class,
        job_id=entry.job_id,
    ), True


def terminal_hook_failure_result(
    entry: JobEntry,
    result: SubAgentResult,
    *,
    error: str,
    error_class: str,
) -> SubAgentResult:
    metadata = dict(result.metadata or {})
    metadata.update(entry.item.metadata)
    return SubAgentResult(
        question=entry.item.question,
        role_id=entry.item.role_id,
        final_content=result.final_content,
        success=False,
        error=error,
        error_class=error_class,
        job_id=entry.job_id,
        metadata=metadata,
    )


async def finish_job(
    bus: _FinishBus,
    entry: JobEntry,
    result: SubAgentResult,
    runtime_spec: SubAgentRuntimeSpec | None = None,
    *,
    session: SubAgentSession | None = None,
) -> SubAgentResult:
    """Idempotently finalize, durably sink, publish, and enqueue one job."""
    if entry.result is not None:
        bus._job_finish_locks.pop(entry.job_id, None)
        return entry.result

    lock = bus._job_finish_locks.setdefault(entry.job_id, asyncio.Lock())
    cancelled = False
    entered = False
    try:
        async with lock:
            entered = True
            if entry.result is not None:
                return entry.result
            finalized = await finalize_job_result(bus, entry, result, runtime_spec)
            stamp = stamp_job_clocks(entry, time.monotonic(), _utc_now())
            runtime = runtime_spec or entry.runtime_spec
            sink = None if runtime is None else runtime.durable_result_sink
            published = finalized
            if sink is not None:
                try:
                    await await_even_if_cancelled(
                        sink(finalized, entry.job_id, entry.item, stamp),
                    )
                except asyncio.CancelledError:
                    cancelled = True
                except Exception as exc:
                    published = terminal_hook_failure_result(
                        entry,
                        finalized,
                        error=str(exc) or type(exc).__name__,
                        error_class=type(exc).__name__,
                    )
            publish_entry(entry, published)
            if session is not None:
                enqueue_pending_result(session, published)
            if cancelled:
                raise asyncio.CancelledError
            return published
    finally:
        if entered:
            bus._job_finish_locks.pop(entry.job_id, None)


# ── Workflow-owned terminal finalizer retry ────────────────────────────


async def apply_job_finalizer(
    bus: _FinishBus,
    entry: JobEntry,
    result: SubAgentResult,
    runtime_spec: SubAgentRuntimeSpec | None = None,
) -> SubAgentResult:
    """Run the workflow-owned terminal hook at most once for one job."""
    del bus
    if entry.finalizer_called:
        return entry.result or result
    runtime = runtime_spec or entry.runtime_spec
    if runtime is None or runtime.job_finalizer is None:
        entry.finalizer_called = True
        return result
    finalized = runtime.job_finalizer(result, entry.job_id, entry.item)
    if inspect.isawaitable(finalized):
        finalized = await finalized
    entry.finalizer_called = True
    return finalized


async def finalize_job_result(
    bus: _FinishBus,
    entry: JobEntry,
    result: SubAgentResult,
    runtime_spec: SubAgentRuntimeSpec | None = None,
) -> SubAgentResult:
    """Retry a failed terminal hook once with a fail-closed result."""
    try:
        return await apply_job_finalizer(bus, entry, result, runtime_spec)
    except asyncio.CancelledError:
        retry_result = terminal_hook_failure_result(
            entry,
            result,
            error="aborted",
            error_class="CancelledError",
        )
    except Exception as exc:
        retry_result = terminal_hook_failure_result(
            entry,
            result,
            error=str(exc) or type(exc).__name__,
            error_class=type(exc).__name__,
        )
    try:
        return await apply_job_finalizer(bus, entry, retry_result, runtime_spec)
    except (asyncio.CancelledError, Exception):
        entry.finalizer_called = True
        return retry_result

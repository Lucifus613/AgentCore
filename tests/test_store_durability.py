from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_core.context as context
from agent_core.context import (
    BlobRef,
    ContextScope,
    FileBlobStore,
    JournalEntry,
    SQLiteJournalStore,
)


def _inode(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


@pytest.fixture
def blob_io(monkeypatch):
    events = []
    real_link = os.link

    def fsync(fd):
        events.append(("fsync", os.fstat(fd)))

    def link(source, target):
        events.append(("link", Path(source).stat()))
        real_link(source, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "link", link)
    return events


def _synced_directories(events) -> list[tuple[int, int]]:
    return [
        (info.st_dev, info.st_ino)
        for operation, info in events
        if operation == "fsync" and stat.S_ISDIR(info.st_mode)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_levels", [0, 1, 3])
async def test_blob_syncs_exact_directory_chain(tmp_path, blob_io, missing_levels):
    root = tmp_path.joinpath(*(["new"] * missing_levels), "blobs")
    if missing_levels == 0:
        root.mkdir()
    store = FileBlobStore(root)
    ref = await store.put(b"data")
    path = store._path(ref.digest)
    directories = [path.parent, root]
    if missing_levels:
        directory = root.parent
        while True:
            directories.append(directory)
            if directory == tmp_path:
                break
            directory = directory.parent
    assert _synced_directories(blob_io) == [_inode(path) for path in directories]
    assert [operation for operation, _ in blob_io] == [
        "fsync", "link", *(["fsync"] * len(directories))
    ]
    operation, file_info = blob_io[0]
    assert operation == "fsync" and stat.S_ISREG(file_info.st_mode)
    assert (file_info.st_dev, file_info.st_ino) == _inode(path)
    assert stat.S_IMODE(blob_io[1][1].st_mode) == 0o444
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert await store.get(ref) == b"data"
    assert not list(path.parent.glob(".blob-*"))

    blob_io.clear()
    assert await store.put(b"data") == ref
    assert [operation for operation, _ in blob_io] == ["fsync", "fsync"]
    assert _synced_directories(blob_io) == [_inode(path.parent), _inode(root)]


@pytest.mark.asyncio
@pytest.mark.parametrize("race", [False, True])
@pytest.mark.parametrize("existing", [b"data", b"evil", b"different length"])
async def test_blob_existing_and_publish_race_are_verified(
    tmp_path, monkeypatch, blob_io, race, existing
):
    store = FileBlobStore(tmp_path / "blobs")
    path = store._path(hashlib.sha256(b"data").hexdigest())
    path.parent.mkdir(parents=True)

    def occupy():
        path.write_bytes(existing)
        path.chmod(0o640)

    if race:
        recorded_link = os.link

        def race_link(source, target):
            occupy()
            recorded_link(source, target)

        monkeypatch.setattr(os, "link", race_link)
    else:
        occupy()

    if existing == b"data":
        ref = await store.put(b"data")
        assert await store.get(ref) == existing
        assert _synced_directories(blob_io) == [_inode(path.parent), _inode(store.root)]
    else:
        with pytest.raises(context.BlobIntegrityError, match="blob integrity check failed"):
            await store.put(b"data")
        assert issubclass(context.BlobIntegrityError, ValueError)
        assert _synced_directories(blob_io) == []
    assert sum(operation == "link" for operation, _ in blob_io) == int(race)
    assert path.read_bytes() == existing
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert not list(path.parent.glob(".blob-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_directory", [0, 1])
async def test_blob_directory_sync_failure_can_be_retried(
    tmp_path, monkeypatch, blob_io, failed_directory
):
    store = FileBlobStore(tmp_path / "blobs")
    store.root.mkdir()
    error = OSError("directory sync failed")
    recorded_fsync = os.fsync
    directory_count = 0

    def failing_fsync(fd):
        nonlocal directory_count
        recorded_fsync(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_count += 1
            if directory_count == failed_directory + 1:
                raise error

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError) as caught:
        await store.put(b"data")
    assert caught.value is error
    path = store._path(hashlib.sha256(b"data").hexdigest())
    assert path.read_bytes() == b"data"
    assert not list(path.parent.glob(".blob-*"))
    assert _synced_directories(blob_io) == [
        _inode(directory) for directory in [path.parent, store.root][:failed_directory + 1]
    ]
    assert [operation for operation, _ in blob_io[:2]] == ["fsync", "link"]

    monkeypatch.setattr(os, "fsync", recorded_fsync)
    blob_io.clear()
    ref = await store.put(b"data")
    assert await store.get(ref) == b"data"
    assert [operation for operation, _ in blob_io] == ["fsync", "fsync"]
    assert _synced_directories(blob_io) == [_inode(path.parent), _inode(store.root)]


@pytest.mark.asyncio
async def test_blob_existing_root_without_pending_work_stops_at_root(tmp_path, blob_io):
    root = tmp_path / "existing" / "parent" / "blobs"
    root.mkdir(parents=True)
    store = FileBlobStore(root)
    for data in (b"first", b"second"):
        blob_io.clear()
        ref = await store.put(data)
        assert _synced_directories(blob_io) == [
            _inode(store._path(ref.digest).parent), _inode(root)
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_directories", [(0,), (0, 2)])
@pytest.mark.parametrize("another_instance", [False, True])
async def test_blob_retry_completes_pending_ancestor_chain(
    tmp_path, monkeypatch, blob_io, failed_directories, another_instance
):
    root = tmp_path / "new" / "parent" / "blobs"
    store = FileBlobStore(root)
    recorded_fsync = os.fsync
    error = OSError("directory sync failed")

    for failed_directory in failed_directories:
        directory_count = 0

        def failing_fsync(fd, failed_directory=failed_directory):
            nonlocal directory_count
            recorded_fsync(fd)
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directory_count += 1
                if directory_count == failed_directory + 1:
                    raise error

        monkeypatch.setattr(os, "fsync", failing_fsync)
        blob_io.clear()
        with pytest.raises(OSError) as caught:
            await store.put(b"data")
        assert caught.value is error
        path = store._path(hashlib.sha256(b"data").hexdigest())
        directories = [path.parent, root, root.parent, root.parent.parent, tmp_path]
        assert _synced_directories(blob_io) == [
            _inode(directory) for directory in directories[:failed_directory + 1]
        ]
        assert not list(path.parent.glob(".blob-*"))

    monkeypatch.setattr(os, "fsync", recorded_fsync)
    blob_io.clear()
    if another_instance:
        store = FileBlobStore(root.parent / "." / "blobs")
    ref = await store.put(b"data")
    assert await store.get(ref) == b"data"
    assert _synced_directories(blob_io) == [_inode(directory) for directory in directories]

    blob_io.clear()
    ref = await store.put(b"new content")
    assert _synced_directories(blob_io) == [_inode(store._path(ref.digest).parent), _inode(root)]


@pytest.mark.asyncio
async def test_blob_any_content_can_complete_pending_ancestor_sync(tmp_path, monkeypatch, blob_io):
    root = tmp_path / "new" / "parent" / "blobs"
    store = FileBlobStore(root)
    recorded_fsync = os.fsync

    def failing_fsync(fd):
        recorded_fsync(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory sync failed")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    for data in (b"first", b"second"):
        with pytest.raises(OSError, match="directory sync failed"):
            await store.put(data)

    monkeypatch.setattr(os, "fsync", recorded_fsync)
    blob_io.clear()
    ref = await store.put(b"second")
    directories = [store._path(ref.digest).parent, root, root.parent, root.parent.parent, tmp_path]
    assert _synced_directories(blob_io) == [_inode(directory) for directory in directories]

    blob_io.clear()
    ref = await store.put(b"first")
    assert _synced_directories(blob_io) == [_inode(store._path(ref.digest).parent), _inode(root)]


@pytest.mark.asyncio
async def test_blob_concurrent_first_puts_sync_new_ancestors(tmp_path, monkeypatch, blob_io):
    root = tmp_path / "new" / "parent" / "blobs"
    barrier = threading.Barrier(4, timeout=5)
    real_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        if path.parent == root and kwargs.get("parents"):
            barrier.wait()
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    stores = [FileBlobStore(root) for _ in range(4)]
    contents = [f"data-{index}".encode() for index in range(4)]
    refs = await asyncio.gather(*(
        store.put(data) for store, data in zip(stores, contents, strict=True)
    ))
    directories = [store._path(ref.digest).parent for store, ref in zip(stores, refs, strict=True)]
    directories.extend([root, root.parent, root.parent.parent, tmp_path])
    synced = _synced_directories(blob_io)
    assert set(synced) == {_inode(directory) for directory in directories}
    for store, ref, data in zip(stores, refs, contents, strict=True):
        assert await store.get(ref) == data


@pytest.mark.asyncio
async def test_blob_recreated_root_syncs_new_ancestor_chain(tmp_path, blob_io):
    root = tmp_path / "new" / "parent" / "blobs"
    store = FileBlobStore(root)
    ref = await store.put(b"data")
    path = store._path(ref.digest)
    await store.delete(ref.digest)
    path.parent.rmdir()
    root.rmdir()
    root.parent.rmdir()
    root.parent.parent.rmdir()

    blob_io.clear()
    ref = await store.put(b"data")
    directories = [store._path(ref.digest).parent, root, root.parent, root.parent.parent, tmp_path]
    assert _synced_directories(blob_io) == [_inode(directory) for directory in directories]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fsync", "link"])
async def test_blob_file_sync_and_link_failures_do_not_publish(
    tmp_path, monkeypatch, blob_io, failure
):
    store = FileBlobStore(tmp_path / "new" / "parent" / "blobs")
    error = OSError(f"{failure} failed")
    recorded_operation = getattr(os, failure)

    def fail(*args):
        if failure == "fsync":
            recorded_operation(*args)
        raise error

    monkeypatch.setattr(os, failure, fail)
    with pytest.raises(OSError) as caught:
        await store.put(b"data")
    assert caught.value is error
    path = store._path(hashlib.sha256(b"data").hexdigest())
    assert not path.exists()
    assert not list(path.parent.glob(".blob-*"))
    assert _synced_directories(blob_io) == []

    monkeypatch.setattr(os, failure, recorded_operation)
    blob_io.clear()
    ref = await store.put(b"data")
    assert await store.get(ref) == b"data"
    root = store.root
    directories = [path.parent, root, root.parent, root.parent.parent, tmp_path]
    assert _synced_directories(blob_io) == [_inode(directory) for directory in directories]


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["digest", "size"])
async def test_blob_get_reports_integrity_error(tmp_path, blob_io, corruption):
    store = FileBlobStore(tmp_path / "blobs")
    ref = await store.put(b"data")
    if corruption == "digest":
        path = store._path(ref.digest)
        path.chmod(0o644)
        path.write_bytes(b"evil")
    else:
        ref = replace(ref, size=ref.size + 1)
    with pytest.raises(context.BlobIntegrityError, match="blob integrity check failed"):
        await store.get(ref)
    assert issubclass(context.BlobIntegrityError, ValueError)


@pytest.mark.asyncio
async def test_blob_rejects_invalid_digest(tmp_path):
    store = FileBlobStore(tmp_path / "blobs")
    with pytest.raises(ValueError, match="invalid SHA-256 digest"):
        await store.get(BlobRef(digest="../invalid", size=0))


@pytest.mark.asyncio
async def test_journal_sets_and_reads_back_durability_pragmas(tmp_path, monkeypatch):
    statements = []
    real_connect = sqlite3.connect

    class RecordingConnection(sqlite3.Connection):
        def execute(self, sql, *args):
            statements.append(sql)
            return super().execute(sql, *args)

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=RecordingConnection)

    monkeypatch.setattr(sqlite3, "connect", connect)
    store = SQLiteJournalStore(tmp_path / "journal.db")
    entry = JournalEntry(ContextScope("session", "run", "agent"), "turn")
    await store.append(entry)
    connection = store._connect()
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()
    expected = [
        "PRAGMA foreign_keys=ON",
        "PRAGMA journal_mode=WAL",
        "PRAGMA journal_mode",
        "PRAGMA synchronous=FULL",
        "PRAGMA synchronous",
    ]
    assert statements[:5] == expected
    insert_index = next(i for i, sql in enumerate(statements) if sql.startswith("INSERT"))
    assert statements[insert_index - 5:insert_index] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["append", "list_entries"])
@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("setting, actual, expected", [
    ("journal_mode", "delete", "wal"), ("synchronous", 1, 2)
])
async def test_journal_rejects_bad_pragma_before_read_or_write(
    tmp_path, monkeypatch, operation, ready, setting, actual, expected
):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    real_connect = sqlite3.connect
    await store.setup()
    store._ready = ready

    class WrongPragmaConnection:
        row_factory = None
        closed = False

        def __init__(self):
            self.statements = []

        def execute(self, sql):
            self.statements.append(sql)
            assert sql.startswith("PRAGMA ")
            values = {"PRAGMA journal_mode": "wal", "PRAGMA synchronous": 2}
            value = actual if sql == f"PRAGMA {setting}" else values.get(sql)
            return SimpleNamespace(fetchone=lambda: (value,))

        def close(self):
            self.closed = True

    connection = WrongPragmaConnection()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: connection)
    scope = ContextScope("session", "run", "agent")
    with pytest.raises(context.JournalStoreError) as caught:
        if operation == "append":
            await store.append(JournalEntry(scope, "turn"))
        else:
            await store.list_entries(scope)
    assert issubclass(context.JournalStoreError, RuntimeError)
    assert setting in str(caught.value)
    assert repr(expected) in str(caught.value)
    assert repr(actual) in str(caught.value)
    assert connection.closed
    assert f"PRAGMA {setting}={'WAL' if setting == 'journal_mode' else 'FULL'}" in connection.statements
    assert connection.statements[-1] == f"PRAGMA {setting}"
    with real_connect(store.path) as database:
        assert database.execute("SELECT count(*) FROM context_journal").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("with_blobs", [False, True])
async def test_journal_identical_retry_returns_original_record(tmp_path, with_blobs):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    blobs = {
        "a": BlobRef("a" * 64, 3, "text/plain"),
        "b": BlobRef("b" * 64, 4),
    } if with_blobs else {}
    entry = JournalEntry(
        ContextScope("session", "run", "agent"), "turn",
        payload={"b": [1, 2], "a": {"x": 1, "y": 2}}, blobs=blobs,
    )
    first = await store.append(entry)
    retry = replace(
        entry, sequence=12345, created_at=entry.created_at + timedelta(days=1),
        payload={"a": {"y": 2, "x": 1}, "b": (1, 2)},
        blobs=dict(reversed(list(blobs.items()))),
    )
    assert await store.append(retry) == first
    assert await store.list_entries(entry.scope) == [first]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM context_journal").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM context_journal_blobs").fetchone()[0] == len(blobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_values, retry_values", [
    ((1, 2), (1, 2)), ((1, 2), [1, 2]), ([1, 2], (1, 2)),
])
async def test_journal_retry_returns_normalized_payload(tmp_path, first_values, retry_values):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    entry = JournalEntry(
        ContextScope("session", "run", "agent"), "turn", payload={"values": first_values},
    )
    first = await store.append(entry)
    retry = replace(entry, payload={"values": retry_values})
    assert await store.append(retry) == first
    assert await store.list_entries(entry.scope) == [first]
    assert first.payload == {"values": [1, 2]}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM context_journal").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_journal_retry_returns_normalized_empty_media_type(tmp_path):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    entry = JournalEntry(
        ContextScope("session", "run", "agent"), "turn",
        blobs={"report": BlobRef("a" * 64, 3, "")},
    )
    first = await store.append(entry)
    assert await store.append(entry) == first
    assert await store.list_entries(entry.scope) == [first]
    assert first.blobs["report"].media_type == "application/octet-stream"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM context_journal").fetchone()[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [
    "session_id", "run_id", "agent_id", "kind", "payload",
    "blob_name", "blob_digest", "blob_size", "blob_media_type",
])
async def test_journal_conflicting_retry_preserves_original(tmp_path, changed):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    entry = JournalEntry(
        ContextScope("session", "run", "agent"), "turn", payload={"value": 1},
        blobs={"report": BlobRef("a" * 64, 3, "text/plain")},
    )
    first = await store.append(entry)
    if changed in {"session_id", "run_id", "agent_id"}:
        retry = replace(entry, scope=replace(entry.scope, **{changed: "different"}))
    elif changed == "kind":
        retry = replace(entry, kind="other")
    elif changed == "payload":
        retry = replace(entry, payload={"value": 2})
    elif changed == "blob_name":
        retry = replace(entry, blobs={"other": entry.blobs["report"]})
    else:
        attribute, value = {
            "blob_digest": ("digest", "b" * 64),
            "blob_size": ("size", 4),
            "blob_media_type": ("media_type", "application/json"),
        }[changed]
        retry = replace(entry, blobs={"report": replace(entry.blobs["report"], **{attribute: value})})
    with pytest.raises(context.JournalConflictError) as caught:
        await store.append(retry)
    assert isinstance(caught.value, sqlite3.IntegrityError)
    assert await store.list_entries(entry.scope) == [first]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM context_journal").fetchone()[0] == 1
        assert connection.execute("SELECT digest FROM context_journal_blobs").fetchall() == [("a" * 64,)]


@pytest.mark.asyncio
async def test_journal_duplicate_blob_digest_keeps_original_integrity_error(tmp_path):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    ref = BlobRef("a" * 64, 3)
    entry = JournalEntry(ContextScope("session", "run"), "turn", blobs={"a": ref, "b": ref})
    with pytest.raises(sqlite3.IntegrityError, match="context_journal_blobs") as caught:
        await store.append(entry)
    assert type(caught.value) is sqlite3.IntegrityError
    assert await store.list_entries(entry.scope) == []
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM context_journal_blobs").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_journal_missing_duplicate_row_reraises_original_error(tmp_path, monkeypatch):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    await store.setup()
    error = sqlite3.IntegrityError("UNIQUE constraint failed: context_journal.entry_id")
    operations = []

    class MissingRowConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            if sql.startswith("INSERT"):
                operations.append("insert")
                raise error
            assert sql == "SELECT * FROM context_journal WHERE entry_id = ?"
            assert params == ("missing",)
            operations.append("select")
            return SimpleNamespace(fetchone=lambda: None)

        def rollback(self):
            operations.append("rollback")

    monkeypatch.setattr(store, "_connect", MissingRowConnection)
    with pytest.raises(sqlite3.IntegrityError) as caught:
        await store.append(JournalEntry(ContextScope("session", "run"), "turn", entry_id="missing"))
    assert caught.value is error
    assert operations == ["insert", "rollback", "select"]


@pytest.mark.asyncio
async def test_journal_concurrent_identical_appends_return_one_record(tmp_path):
    store = SQLiteJournalStore(tmp_path / "journal.db")
    await store.setup()
    entry = JournalEntry(
        ContextScope("session", "run", "agent"), "turn", payload={"value": 1},
        blobs={"report": BlobRef("a" * 64, 3)},
    )
    retry = replace(entry, created_at=entry.created_at + timedelta(days=1))
    barrier = threading.Barrier(2, timeout=5)

    def append_in_thread(candidate):
        barrier.wait()
        return asyncio.run(store.append(candidate))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(append_in_thread, entry)
        second = executor.submit(append_in_thread, retry)
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert results[0] == results[1]
    assert results[0].sequence > 0
    assert results[0].created_at in {entry.created_at, retry.created_at}
    assert await store.list_entries(entry.scope) == [results[0]]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM context_journal").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM context_journal_blobs").fetchone()[0] == 1

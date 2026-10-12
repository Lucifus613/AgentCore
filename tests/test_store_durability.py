from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import agent_core.context as context
from agent_core.context import BlobRef, FileBlobStore


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
@pytest.mark.parametrize("failure", ["fsync", "link"])
async def test_blob_file_sync_and_link_failures_do_not_publish(
    tmp_path, monkeypatch, blob_io, failure
):
    store = FileBlobStore(tmp_path / "blobs")
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

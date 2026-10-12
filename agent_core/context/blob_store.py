"""Content-addressed filesystem blob storage."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path

from agent_core.context.models import BlobRef


class BlobIntegrityError(ValueError):
    """Stored blob content does not match its reference."""


class FileBlobStore:
    """Atomic, read-only blobs rooted outside agent-writable workspaces."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("invalid SHA-256 digest")
        return self.root / digest[:2] / digest[2:]

    async def put(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> BlobRef:
        digest = hashlib.sha256(data).hexdigest()
        await asyncio.to_thread(self._write, digest, data)
        return BlobRef(digest=digest, size=len(data), media_type=media_type)

    def _write(self, digest: str, data: bytes) -> None:
        path = self._path(digest)
        sync_stop = self.root
        while not sync_stop.exists():
            sync_stop = sync_stop.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._read_verified(digest, len(data))
        else:
            fd, temporary = tempfile.mkstemp(prefix=".blob-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o444)
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    self._read_verified(digest, len(data))
            finally:
                Path(temporary).unlink(missing_ok=True)
        directory = path.parent
        while True:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if directory == sync_stop:
                break
            directory = directory.parent

    async def get(self, ref: BlobRef) -> bytes:
        return await asyncio.to_thread(self._read_verified, ref.digest, ref.size)

    def _read_verified(self, digest: str, size: int) -> bytes:
        data = self._path(digest).read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise BlobIntegrityError("blob integrity check failed")
        return data

    async def delete(self, digest: str) -> None:
        await asyncio.to_thread(self._path(digest).unlink, missing_ok=True)


__all__ = ["BlobIntegrityError", "FileBlobStore"]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Content-addressed cache for build products (objects, linked binaries).

A key is the SHA-256 of everything that determines the output: the exact
compile / link command (with the per-job paths replaced by placeholders), the
source text, the text of every header it includes from the job directory, and
the identity (path, size, mtime) of absolute-path inputs such as a driver's
``.incbin`` data file. Same key -> byte-identical inputs to the same tool, so
the cached file is reused instead of rebuilt.

Concurrency: entries are written to a unique temp name in the cache directory
and ``os.replace``d into place, so readers only ever see complete files, and
two builders racing on one key (other processes sharing the directory) both
produce a valid entry. Within one process an ``asyncio.Lock`` per key makes the
second requester wait for the first build instead of duplicating it.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from ...telemetry.core import count as _tel_count

# absolute paths a TU pulls in that are not part of its text
_ABS_INPUT_RE = re.compile(r'(?:\.incbin\s+\\?"|#\s*include\s+")(/[^"\\]+)')


def external_inputs(text: str) -> list[str]:
    """``path:size:mtime_ns`` for every absolute ``.incbin`` / ``#include "..."``
    target in ``text`` (missing files are keyed as missing; the build fails
    on its own)."""
    out = []
    for p in sorted(set(_ABS_INPUT_RE.findall(text))):
        try:
            st = os.stat(p)
            out.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            out.append(f"{p}:missing")
    return out


def cache_key(*parts: str | bytes | Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (list, tuple)):
            part = "\n".join(part)
        b = part if isinstance(part, bytes) else str(part).encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


class BuildCache:
    """``get_or_build(key, suffix, build)`` -> path of the cached product."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = 0
        self.misses = 0

    def path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / f"{key}{suffix}"

    async def get_or_build(
        self, key: str, suffix: str, build: Callable[[Path], Awaitable[None]]
    ) -> tuple[Path, bool]:
        """Return ``(path, hit)``. ``build(tmp)`` must write the product to
        ``tmp``; it raises on failure and nothing is cached."""
        dst = self.path(key, suffix)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if dst.exists():
                self.hits += 1
                _tel_count("build.cache_hit")
                return dst, True
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                await build(tmp)
                if not tmp.exists():
                    raise FileNotFoundError(f"build did not produce {tmp}")
                os.replace(tmp, dst)
            finally:
                if tmp.exists():
                    tmp.unlink()
            self.misses += 1
            _tel_count("build.cache_miss")
            return dst, False


def link_or_copy(src: Path, dst: Path) -> None:
    """Materialise a cached product at ``dst`` (hard link when possible, so a
    100 MB binary with linked-in data costs nothing; copy across devices)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        os.link(src, tmp)
    except OSError:
        import shutil

        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o755)
    os.replace(tmp, dst)

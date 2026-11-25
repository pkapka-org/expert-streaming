"""Expert cache: a fixed number of memory slots filled by direct reads from disk."""

from __future__ import annotations

import json
import mmap
import os
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .policies import POLICIES, Key


class ExpertStore:
    """`with store.use(layer, experts) as ready:` yields (expert, input weights, output weights)
    as soon as each one is in memory. Memory never exceeds `budget_mb`.

    Reads use O_DIRECT, so they come from the disk and not from the operating system's page cache.
    """

    def __init__(self, path: Path, meta: dict, budget_mb: int, policy: str = "lfu",
                 readers: int = 4, hot: list[Key] | None = None, direct: bool = True):
        config = meta["config"]
        self.hidden, self.inner = config["hidden_size"], config["intermediate_size"]
        self.layers, self.experts = config["num_hidden_layers"], config["num_local_experts"]
        self.stride = meta["stride"]
        total = self.layers * self.experts
        self.capacity = min(total, int(budget_mb * 1_000_000) // self.stride)
        self.minimum = config["num_experts_per_tok"] + readers
        if self.capacity < min(total, self.minimum):
            need = self.minimum * self.stride / 1e6
            raise ValueError(f"The expert budget must be at least {need:.0f} MB.")

        flags = os.O_RDONLY
        self.direct = False
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(path, flags | os.O_DIRECT)
                self.direct = True
            except OSError:  # some filesystems, such as tmpfs, do not support it
                pass
        if not self.direct:
            self.fd = os.open(path, flags)

        self.policy = POLICIES[policy](self.capacity, self.layers, hot)
        self.free: list[mmap.mmap] = []
        self.allocated = 0
        self.slots: dict[Key, mmap.mmap] = {}
        self.loading: dict[Key, Future] = {}
        self.busy: dict[Key, int] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(readers, thread_name_prefix="reader")
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {"hits": 0, "misses": 0, "joined": 0, "bytes": 0,
                      "prefetched": 0, "prefetch_used": 0}
        self.unused_prefetch: set[Key] = set()

    # -- loading ------------------------------------------------------------
    def _take_slot(self, key: Key) -> mmap.mmap | None:
        if self.free:
            return self.free.pop()
        if self.allocated < self.capacity:
            self.allocated += 1
            return mmap.mmap(-1, self.stride)  # page-aligned, as O_DIRECT needs
        victim = self.policy.victim(key, set(self.busy))
        if victim is None:
            return None
        self.policy.remove(victim)
        self.unused_prefetch.discard(victim)
        return self.slots.pop(victim)

    def _read(self, key: Key, slot: mmap.mmap) -> None:
        try:
            offset = (key[0] * self.experts + key[1]) * self.stride
            view, done = memoryview(slot), 0
            try:
                while done < self.stride:
                    n = os.preadv(self.fd, [view[done:]], offset + done)
                    if n <= 0:
                        raise OSError(f"short read of expert {key}")
                    done += n
            finally:
                view.release()
        except BaseException:
            with self.lock:
                del self.loading[key]
                self.free.append(slot)
            raise
        with self.lock:
            self.stats["bytes"] += self.stride
            self.slots[key] = slot
            self.policy.add(key)
            del self.loading[key]

    def _start(self, key: Key) -> Future | None:
        slot = self._take_slot(key)
        if slot is None:
            return None
        future: Future = Future()
        self.loading[key] = future

        def work() -> None:
            try:
                self._read(key, slot)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(key)

        self.pool.submit(work)
        return future

    def prefetch(self, layer: int, experts: list[int], spare: int) -> None:
        """Start reading experts that will probably be needed. Never blocks."""
        with self.lock:
            for expert in experts:
                key = (layer, expert)
                if key in self.slots or key in self.loading:
                    continue
                if self.capacity - len(self.busy) - len(self.loading) <= spare:
                    return
                if self._start(key) is None:
                    return
                self.stats["prefetched"] += 1
                self.unused_prefetch.add(key)

    @contextmanager
    def use(self, layer: int, experts: list[int]):
        """Experts are released when the block ends; at most `minimum` may be used at once."""
        keys = [(layer, int(e)) for e in experts]
        ready: list[Key] = []
        waiting: list[Future] = []
        with self.lock:
            for key in keys:  # hold everything first, so that no load evicts a sibling
                self.busy[key] = self.busy.get(key, 0) + 1
            try:
                for key in keys:
                    if key in self.unused_prefetch:
                        self.unused_prefetch.discard(key)
                        self.stats["prefetch_used"] += 1
                    if key in self.slots:
                        self.stats["hits"] += 1
                        self.policy.touch(key)
                        ready.append(key)
                    elif key in self.loading:
                        self.stats["joined"] += 1
                        waiting.append(self.loading[key])
                    else:
                        self.stats["misses"] += 1
                        future = self._start(key)
                        if future is None:
                            raise RuntimeError("the expert cache is too small for this request")
                        waiting.append(future)
            except BaseException:
                self._release(keys)
                raise

        def arrivals() -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
            for key in ready:
                yield self._arrays(key)
            for future in as_completed(waiting):
                yield self._arrays(future.result())

        try:
            yield arrivals()
        finally:
            for future in waiting:
                future.exception()  # reads must finish before their slots can be reused
            with self.lock:
                self._release(keys)

    def _release(self, keys: list[Key]) -> None:
        for key in keys:
            if self.busy[key] == 1:
                del self.busy[key]
            else:
                self.busy[key] -= 1

    def _arrays(self, key: Key) -> tuple[int, np.ndarray, np.ndarray]:
        words = np.frombuffer(self.slots[key], np.uint16)
        split = 2 * self.inner * self.hidden
        return (key[1], words[:split].reshape(2 * self.inner, self.hidden),
                words[split: split + self.hidden * self.inner].reshape(self.hidden, self.inner))

    def close(self) -> None:
        self.pool.shutdown()
        os.close(self.fd)


def load_hot(path: Path) -> list[Key]:
    return [tuple(k) for k in json.loads(Path(path).read_text())]

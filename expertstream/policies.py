"""Which cached expert to drop. The engine and the simulator use the same classes.

A key is (layer, expert). `victim` returns the key to drop to make room for `incoming`, never one
in `busy`, or None when nothing can be dropped.
"""

from __future__ import annotations

from collections import Counter, OrderedDict

Key = tuple[int, int]


class LRU:
    """Drop the expert unused for longest, whatever its layer."""

    def __init__(self, capacity: int, layers: int, hot: list[Key] | None = None):
        self.capacity, self.layers = capacity, layers
        self.order: OrderedDict[Key, None] = OrderedDict()

    def add(self, key: Key) -> None:
        self.order[key] = None

    def touch(self, key: Key) -> None:
        self.order.move_to_end(key)

    def remove(self, key: Key) -> None:
        del self.order[key]

    def victim(self, incoming: Key, busy: set[Key]) -> Key | None:
        return next((k for k in self.order if k not in busy), None)


class LayerLRU(LRU):
    """LRU inside each layer, with an equal share of the cache per layer.

    Layers run in a fixed cycle, so plain LRU drops an expert just before its layer comes round again.
    """

    def __init__(self, capacity: int, layers: int, hot: list[Key] | None = None):
        super().__init__(capacity, layers)
        self.count: Counter[int] = Counter()

    def add(self, key: Key) -> None:
        super().add(key)
        self.count[key[0]] += 1

    def remove(self, key: Key) -> None:
        super().remove(key)
        self.count[key[0]] -= 1

    def quota(self, layer: int) -> int:
        return self.capacity // self.layers + (layer < self.capacity % self.layers)

    def victim(self, incoming: Key, busy: set[Key]) -> Key | None:
        free = [k for k in self.order if k not in busy]
        if self.count[incoming[0]] >= self.quota(incoming[0]):
            own = next((k for k in free if k[0] == incoming[0]), None)
            if own is not None:
                return own
        over = next((k for k in free if self.count[k[0]] > self.quota(k[0])), None)
        return over if over is not None else next(iter(free), None)


class LFU(LRU):
    """Drop the expert used least often. Counts survive eviction."""

    def __init__(self, capacity: int, layers: int, hot: list[Key] | None = None):
        super().__init__(capacity, layers)
        self.uses: Counter[Key] = Counter()

    def add(self, key: Key) -> None:
        super().add(key)
        self.uses[key] += 1

    def touch(self, key: Key) -> None:
        super().touch(key)
        self.uses[key] += 1

    def victim(self, incoming: Key, busy: set[Key]) -> Key | None:
        return min((k for k in self.order if k not in busy), key=self.uses.__getitem__, default=None)


class Pinned(LayerLRU):
    """Keep a hot set learned from earlier use; the rest of the cache is LayerLRU."""

    PINNED_SHARE = 0.75

    def __init__(self, capacity: int, layers: int, hot: list[Key] | None = None):
        super().__init__(capacity, layers)
        self.pinned = set((hot or [])[: int(capacity * self.PINNED_SHARE)])

    def victim(self, incoming: Key, busy: set[Key]) -> Key | None:
        found = super().victim(incoming, busy | self.pinned)
        # A very small cache can be all pinned or busy; then a pinned expert has to go.
        return found if found is not None else super().victim(incoming, busy)


POLICIES = {"lru": LRU, "layer": LayerLRU, "lfu": LFU, "pinned": Pinned}

"""Replay the recorded test trace through each cache policy. No model is needed.

    python -m bench.simulate [test|calibration]

`best` is Belady's rule, which drops the expert needed furthest in the future. It needs to know the
future, so it is a ceiling and not a usable policy. Every policy must cache each expert it reads.
"""

import json
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from expertstream.policies import POLICIES

HERE = Path(__file__).resolve().parent
LAYERS, EXPERTS, STRIDE = 24, 32, 3_145_728


class Best:
    def __init__(self, capacity, layers, hot, trace):
        self.keys, self.now, self.uses = set(), 0, defaultdict(list)
        for i, (layer, experts) in enumerate(trace):
            for e in experts:
                self.uses[layer, e].append(i)

    def add(self, key):
        self.keys.add(key)

    def touch(self, key):
        pass

    def remove(self, key):
        self.keys.discard(key)

    def victim(self, incoming, busy):
        def following(key):
            later = self.uses[key]
            i = bisect_right(later, self.now)
            return later[i] if i < len(later) else float("inf")

        return max((k for k in self.keys if k not in busy), key=following, default=None)


def simulate(name: str, trace: list, capacity: int, hot: list) -> float:
    if name == "best":
        policy = Best(capacity, LAYERS, hot, trace)
    else:
        policy = POLICIES[name](capacity, LAYERS, hot)
    cached, hits, total = set(), 0, 0
    for step, (layer, experts) in enumerate(trace):
        if name == "best":
            policy.now = step
        keys = [(layer, e) for e in experts]
        busy = set(keys)
        for key in keys:
            total += 1
            if key in cached:
                hits += 1
                policy.touch(key)
                continue
            if len(cached) >= capacity:
                victim = policy.victim(key, busy)
                policy.remove(victim)
                cached.discard(victim)
            cached.add(key)
            policy.add(key)
    return hits / total


def main() -> None:
    traces = json.loads((HERE / "traces.json").read_text())
    hot = [tuple(k) for k in json.loads((HERE / "hot.json").read_text())]
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    trace = [step for run in traces[which] for step in run]
    names = [*POLICIES, "best"]
    if which == "calibration":
        names.remove("pinned")  # it would be tested on the data it learned from
    rows = []
    print("budget  share  " + "".join(f"{n:>8}" for n in names))
    for mb in (64, 128, 256, 512, 768, 1024, 1536):
        capacity = mb * 1_000_000 // STRIDE
        row = {"budget_mb": mb, "experts": capacity,
               **{n: round(simulate(n, trace, capacity, hot), 4) for n in names}}
        rows.append(row)
        print(f"{mb:>5}MB {capacity / LAYERS / EXPERTS:>6.0%}  "
              + "".join(f"{row[n]:>8.1%}" for n in names), flush=True)
    (HERE / f"simulation_{which}.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()

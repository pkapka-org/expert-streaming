"""Time real generation. Every run is a fresh process, and configurations are interleaved.

    python -m bench.speed
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIGS = [(64, "lfu", 0), (64, "lfu", 1),
           (256, "lru", 0), (256, "lfu", 0), (256, "lfu", 1),
           (512, "lru", 0), (512, "lfu", 0), (512, "lfu", 1),
           (1024, "lru", 0), (1024, "lfu", 0), (1024, "lfu", 1)]
PROMPT = "Describe what happens to water as it is heated from ice to steam."
REPEATS, TOKENS = 3, 64


def one(budget: int, policy: str, prefetch: int) -> dict:
    from expertstream.chat import Chat
    from expertstream.engine import Engine

    model = HERE.parent / "models" / "granite"
    chat = Chat(model)
    engine = Engine(model, budget_mb=budget, policy=policy, prefetch=bool(prefetch))
    ids = chat.encode([{"role": "user", "content": PROMPT}])
    start, first, n = time.perf_counter(), None, 0
    for _ in engine.generate(ids, limit=TOKENS):
        n += 1
        if first is None:
            first = time.perf_counter()
            engine.store.reset_stats()
            engine.lookahead = [0, 0]
    seconds, s = time.perf_counter() - first, engine.store.stats
    with open("/proc/self/status") as f:
        status = dict(line.split(":", 1) for line in f if ":" in line)
    needed = s["hits"] + s["misses"] + s["joined"]
    return {"first_token_s": first - start, "tokens_per_s": (n - 1) / seconds,
            "read_mb_per_token": s["bytes"] / 1e6 / (n - 1), "ready_share": s["hits"] / needed,
            "prefetch_used_share": s["prefetch_used"] / max(1, s["prefetched"]),
            "lookahead_recall": engine.lookahead[0] / max(1, engine.lookahead[1]),
            "peak_rss_mb": int(status["VmHWM"].split()[0]) / 1024, "direct": engine.store.direct}


def main() -> None:
    results = {c: [] for c in CONFIGS}
    for _ in range(REPEATS):
        for c in CONFIGS:
            out = subprocess.run([sys.executable, "-m", "bench.speed", *map(str, c)], check=True,
                                 capture_output=True, text=True, cwd=HERE.parent, env=os.environ)
            results[c].append(json.loads(out.stdout.splitlines()[-1]))
    rows = []
    for (budget, policy, prefetch), runs in results.items():
        row = {"budget_mb": budget, "policy": policy, "prefetch": bool(prefetch)}
        for key in runs[0]:
            row[key] = round(statistics.median(r[key] for r in runs), 3)
        row["tokens_per_s_min_max"] = [round(f(r["tokens_per_s"] for r in runs), 2) for f in (min, max)]
        rows.append(row)
        print(row, flush=True)
    (HERE / "speed.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(json.dumps(one(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))))
    else:
        main()

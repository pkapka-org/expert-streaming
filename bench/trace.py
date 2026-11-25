"""Record which experts real text uses. The calibration and test prompts do not overlap.

    python -m bench.trace
"""

import json
from collections import Counter
from pathlib import Path

from expertstream.chat import Chat
from expertstream.engine import Engine

HERE = Path(__file__).resolve().parent
PROMPTS = {
    "calibration": [
        "Write a short note to a customer whose parcel is late.",
        "What is the difference between a virus and a bacterium?",
        "Write a Python function that reverses the words in a sentence.",
        "Summarise the plot of a story about a lighthouse keeper.",
        "List five things to pack for a winter hike.",
        "Explain compound interest to a teenager.",
        "Translate 'good morning, how are you?' into French and Spanish.",
        "What are the steps to change a bicycle tyre?",
    ],
    "test": [
        "Explain in simple words how a refrigerator works.",
        "Write a SQL query that finds the ten most recent orders.",
        "Give three tips for a job interview.",
        "Why did the Roman Empire decline?",
        "Write a polite reply declining a meeting invitation.",
        "What should a small shop do to reduce electricity use?",
        "Describe how rainbows form.",
        "How do I make vegetable soup?",
    ],
}


def main() -> None:
    model = HERE.parent / "models" / "granite"
    chat, engine = Chat(model), Engine(model, budget_mb=512)
    traces = {}
    for name, prompts in PROMPTS.items():
        traces[name] = []
        for prompt in prompts:
            engine.trace = None
            run = []
            for i, _ in enumerate(engine.generate(
                    chat.encode([{"role": "user", "content": prompt}]), limit=65)):
                if i == 0:
                    engine.trace = run  # decoding only, not the prompt
            traces[name].append(run)
            print(name, len(run) // engine.layers, "tokens", flush=True)
    (HERE / "traces.json").write_text(json.dumps(traces, separators=(",", ":")))
    uses = Counter((layer, e) for run in traces["calibration"] for layer, es in run for e in es)
    hot = [list(k) for k, _ in uses.most_common()]
    (model / "hot.json").write_text(json.dumps(hot))
    (HERE / "hot.json").write_text(json.dumps(hot))


if __name__ == "__main__":
    main()

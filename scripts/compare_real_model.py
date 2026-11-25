"""Compare the full model with Hugging Face Transformers (needs torch and transformers).

    <env with torch> scripts/compare_real_model.py reference out.json   # writes the reference
    python scripts/compare_real_model.py check out.json                 # compares this engine with it
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TEXTS = [
    "The three primary colours are",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Customers may return unused products within 60 days of delivery. The customer pays the postage",
    "In 1969 the first people to land on the Moon were",
]
STEPS = 24


def reference(path: str) -> None:
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(str(ROOT / "models/src/tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(ROOT / "models/src", dtype=torch.bfloat16).eval()
    # Keep the rotary buffer exact: it is computed, not part of the checkpoint.
    rows = []
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new_tokens=STEPS, do_sample=False,
                                 eos_token_id=None, use_cache=True)[0].tolist()
            logits = model(torch.tensor([out])).logits[0].float()
        rows.append({"prompt": len(ids), "tokens": out, "top1": logits.argmax(-1).tolist(),
                     "last": logits[-1].topk(50).indices.tolist(),
                     "last_values": logits[-1].topk(50).values.tolist()})
        print(tok.decode(out)[:100], flush=True)
    Path(path).write_text(json.dumps(rows))


def check(path: str) -> None:
    import numpy as np

    from expertstream.engine import Engine

    engine = Engine(ROOT / "models/granite", budget_mb=512)
    same = total = 0
    worst = 0.0
    for row in json.loads(Path(path).read_text()):
        cache = engine.new_cache()
        for token, expected in zip(row["tokens"], row["top1"], strict=True):
            logits = engine.forward([token], cache)
            same += int(logits.argmax()) == expected
            total += 1
        worst = max(worst, float(np.abs(logits[row["last"]] - np.array(row["last_values"])).max()))
    print(f"same most-likely token at {same}/{total} positions; "
          f"largest difference in the last top-50 logits {worst:.3f}")


if __name__ == "__main__":
    {"reference": reference, "check": check}[sys.argv[1]](sys.argv[2])

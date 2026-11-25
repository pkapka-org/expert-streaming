"""GraniteMoe forward pass with streamed experts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import kernel
from .store import ExpertStore, Key, load_hot


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.exp(x - x.max(-1, keepdims=True))
    return x / x.sum(-1, keepdims=True)


class KVCache:
    """Grows with the conversation. Allocating the full length up front would not be free: NumPy asks
    for 2 MB huge pages, so the first token alone would make most of the buffer resident."""

    STEP = 256

    def __init__(self, config: dict, limit: int):
        self.head = config["hidden_size"] // config["num_attention_heads"]
        self.kv_heads, self.length, self.limit = config["num_key_value_heads"], 0, limit
        empty = np.empty((self.kv_heads, 0, self.head), np.float32)
        self.k, self.v = ([empty] * config["num_hidden_layers"] for _ in range(2))

    def write(self, layer: int, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Append this layer's keys and values; return everything stored so far."""
        end = self.length + k.shape[1]
        for side, new in ((self.k, k), (self.v, v)):
            if side[layer].shape[1] < end:
                size = min(self.limit, -(-end // self.STEP) * self.STEP)
                grown = np.empty((self.kv_heads, size, self.head), np.float32)
                grown[:, : self.length] = side[layer][:, : self.length]
                side[layer] = grown
            side[layer][:, self.length: end] = new
        return self.k[layer][:, :end], self.v[layer][:, :end]


class Engine:
    def __init__(self, model: str | Path, budget_mb: int = 256, policy: str = "lfu",
                 prefetch: bool = False, threads: int = 4, direct: bool = True,
                 hot: list[Key] | None = None):
        model = Path(model)
        meta = json.loads((model / "meta.json").read_text())
        self.config = c = meta["config"]
        self.layers, self.top_k = c["num_hidden_layers"], c["num_experts_per_tok"]
        self.heads, self.kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
        self.head = c["hidden_size"] // self.heads
        raw = np.fromfile(model / "dense.bin", np.uint16)
        self.dense = {
            name: raw[d["offset"] // 2:][: int(np.prod(d["shape"]))].reshape(d["shape"])
            for name, d in meta["dense"].items()
        }
        # Small tensors are used every token, so keep them ready in float32.
        self.small = {n: kernel.to_f32(w) for n, w in self.dense.items() if "norm" in n or "router" in n}
        if hot is None and (model / "hot.json").exists():
            hot = load_hot(model / "hot.json")
        self.store = ExpertStore(model / "experts.bin", meta, budget_mb, policy, threads, hot, direct)
        self.threads = ThreadPoolExecutor(threads, thread_name_prefix="compute")
        self.prefetching = prefetch
        self.trace: list[tuple[int, tuple[int, ...]]] | None = None  # (layer, experts) when recording
        self.lookahead = [0, 0]  # experts predicted correctly, experts needed
        half = np.arange(0, self.head, 2, dtype=np.float32) / self.head
        self.inverse_frequency = 1.0 / (np.float32(c["rope_theta"]) ** half)

    def new_cache(self, limit: int = 4096) -> KVCache:
        return KVCache(self.config, limit)

    # -- pieces ---------------------------------------------------------------
    def _weight(self, layer: int, name: str) -> np.ndarray:
        return self.dense[f"model.layers.{layer}.{name}.weight"]

    def _norm(self, x: np.ndarray, name: str) -> np.ndarray:
        scale = np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(self.config["rms_norm_eps"]))
        return x / scale * self.small[f"model.{name}.weight"]

    def _linear(self, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(x) == 1:
            return kernel.matvec(w, x[0])[None]
        return x @ kernel.to_f32(w).T

    def _rope(self, x: np.ndarray, start: int) -> np.ndarray:
        angle = np.arange(start, start + x.shape[1], dtype=np.float32)[:, None] * self.inverse_frequency
        cos, sin = np.cos(angle), np.sin(angle)
        a, b = x[..., : self.head // 2], x[..., self.head // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], -1)

    def _attention(self, x: np.ndarray, layer: int, cache: KVCache) -> np.ndarray:
        n, start = len(x), cache.length
        parts = [self._weight(layer, f"self_attn.{p}_proj") for p in "qkv"]
        q, k, v = (self._linear(w, x) for w in parts)
        q = self._rope(q.reshape(n, self.heads, self.head).swapaxes(0, 1), start)
        k = self._rope(k.reshape(n, self.kv_heads, self.head).swapaxes(0, 1), start)
        k, v = cache.write(layer, k, v.reshape(n, self.kv_heads, self.head).swapaxes(0, 1))

        group = self.heads // self.kv_heads
        q = q.reshape(self.kv_heads, group * n, self.head)  # the heads of a group share k and v
        score = (q @ k.swapaxes(1, 2)) * np.float32(self.config["attention_multiplier"])
        if n > 1:
            mask = np.triu(np.full((n, start + n), -np.inf, np.float32), start + 1)
            score += np.tile(mask, (group, 1))
        out = (softmax(score) @ v).reshape(self.heads, n, self.head)
        return self._linear(self._weight(layer, "self_attn.o_proj"),
                            np.ascontiguousarray(out.swapaxes(0, 1).reshape(n, -1)))

    def _route(self, x: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray]:
        logits = x @ self.small[f"model.layers.{layer}.block_sparse_moe.router.layer.weight"].T
        chosen = np.argpartition(-logits, self.top_k - 1, axis=-1)[:, : self.top_k]
        return chosen, softmax(np.take_along_axis(logits, chosen, -1))

    def _expert(self, x: np.ndarray, win: np.ndarray, wout: np.ndarray) -> np.ndarray:
        if len(x) <= 4:
            return np.stack([kernel.expert_apply(win, wout, row) for row in x])
        hidden = x @ kernel.to_f32(win).T
        a, gate = np.split(hidden, 2, axis=-1)
        # Below -80 SiLU is zero in float32; clipping only avoids an overflow warning.
        return (a / (1.0 + np.exp(-np.maximum(a, -80))) * gate) @ kernel.to_f32(wout).T

    def _moe(self, x: np.ndarray, layer: int, guess: list[int]) -> np.ndarray:
        chosen, weight = self._route(x, layer)
        needed = np.unique(chosen)
        if self.trace is not None:
            self.trace.append((layer, tuple(int(e) for e in needed)))
        out = np.zeros_like(x)
        # A long prompt needs nearly every expert, so take them a cache-sized group at a time.
        size = self.store.minimum - 4 if len(x) > 1 else self.top_k
        for i in range(0, len(needed), size):
            with self.store.use(layer, needed[i: i + size].tolist()) as ready:
                if len(x) == 1:
                    # This layer's experts are held by now, so a wrong guess cannot evict them.
                    self.store.prefetch(layer + 1, guess, spare=self.top_k)
                    jobs = [(e, self.threads.submit(kernel.expert_apply, win, wout, x[0]))
                            for e, win, wout in ready]
                    parts = {e: job.result() for e, job in jobs}
                    for slot, e in enumerate(chosen[0]):  # fixed order: results do not depend on timing
                        out[0] += weight[0, slot] * parts[int(e)]
                else:
                    for e, win, wout in ready:
                        rows, slots = np.nonzero(chosen == e)
                        out[rows] += weight[rows, slots][:, None] * self._expert(x[rows], win, wout)
        return out

    def _look_ahead(self, x: np.ndarray, layer: int) -> list[int]:
        """Guess the next layer's experts from the hidden state before this layer's experts ran."""
        if layer + 1 >= self.layers or len(x) != 1:
            return []
        guess, _ = self._route(self._norm(x, f"layers.{layer + 1}.post_attention_layernorm"), layer + 1)
        return guess[0].tolist()

    # -- running ----------------------------------------------------------------
    def forward(self, tokens: list[int], cache: KVCache) -> np.ndarray:
        """Add tokens to the cache and return the logits after the last one."""
        if not tokens:
            raise ValueError("no tokens")
        if cache.length + len(tokens) > cache.limit:
            raise ValueError(f"the conversation is longer than {cache.limit} tokens")
        c, residual = self.config, np.float32(self.config["residual_multiplier"])
        embedding = self.dense["model.embed_tokens.weight"]
        x = kernel.to_f32(embedding[np.asarray(tokens)]) * np.float32(c["embedding_multiplier"])
        guess: list[int] = []
        for layer in range(self.layers):
            x = x + residual * self._attention(self._norm(x, f"layers.{layer}.input_layernorm"), layer, cache)
            normed = self._norm(x, f"layers.{layer}.post_attention_layernorm")
            if guess:
                actual = set(self._route(normed, layer)[0][0].tolist())
                self.lookahead[0] += len(actual.intersection(guess))
                self.lookahead[1] += len(actual)
            guess = self._look_ahead(x, layer)
            x = x + residual * self._moe(normed, layer, guess if self.prefetching else [])
        cache.length += len(tokens)
        last = self._norm(x[-1:], "norm")[0]
        blocks = np.array_split(embedding, 4)  # the output layer reuses the embedding
        logits = np.concatenate(list(self.threads.map(lambda w: kernel.matvec(w, last), blocks)))
        return logits / np.float32(c["logits_scaling"])

    def prefill(self, tokens: list[int], cache: KVCache, chunk: int = 256) -> np.ndarray:
        for i in range(0, len(tokens), chunk):
            logits = self.forward(tokens[i: i + chunk], cache)
        return logits

    def generate(self, tokens: list[int], limit: int = 256, temperature: float = 0.0,
                 seed: int = 0, context: int = 4096):
        """Yield new token ids until the end token or the limit."""
        random = np.random.default_rng(seed)
        cache = self.new_cache(context)
        logits = self.prefill(tokens, cache)
        for i in range(limit):
            if temperature > 0:
                token = int(random.choice(len(logits), p=softmax(logits.astype(np.float64) / temperature)))
            else:
                token = int(logits.argmax())
            if token == self.config["eos_token_id"] or cache.length >= cache.limit:
                return
            yield token
            if i + 1 < limit:
                logits = self.forward([token], cache)

    def close(self) -> None:
        self.threads.shutdown()
        self.store.close()

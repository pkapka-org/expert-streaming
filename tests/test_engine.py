import json
from pathlib import Path

import numpy as np
import pytest

from expertstream import kernel
from expertstream.convert import convert
from expertstream.engine import Engine

FIXTURE = Path(__file__).parent / "fixture"
REFERENCE = json.loads((FIXTURE / "reference.json").read_text())
TOKENS, LOGITS = REFERENCE["tokens"], np.array(REFERENCE["logits"])


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    convert(FIXTURE, out)
    return out


def small_budget(model):
    """Just above the smallest allowed cache: 7 of 24 experts."""
    return 7 * json.loads((model / "meta.json").read_text())["stride"] / 1e6


def make(model, **kwargs):
    return Engine(model, budget_mb=kwargs.pop("budget_mb", small_budget(model)), **kwargs)


def test_prompt_matches_transformers(model):
    engine = make(model)
    logits = engine.forward(TOKENS, engine.new_cache())
    np.testing.assert_allclose(logits, LOGITS[-1], atol=1e-3)


def test_token_by_token_matches_transformers_at_every_position(model):
    engine = make(model)
    cache = engine.new_cache()
    for i, token in enumerate(TOKENS):
        np.testing.assert_allclose(engine.forward([token], cache), LOGITS[i], atol=1e-3)


def test_chunked_prompt_matches(model):
    engine = make(model)
    np.testing.assert_allclose(engine.prefill(TOKENS, engine.new_cache(), chunk=7), LOGITS[-1], atol=1e-3)


def test_greedy_text_matches_transformers(model):
    assert list(make(model).generate(TOKENS, limit=12)) == REFERENCE["greedy"]


@pytest.mark.parametrize("policy", ["lru", "layer", "lfu", "pinned"])
@pytest.mark.parametrize("prefetch", [True, False])
def test_caching_never_changes_the_output(model, policy, prefetch):
    engine = make(model, policy=policy, prefetch=prefetch, hot=[(0, 1), (1, 2), (2, 3)])
    cache, whole = engine.new_cache(), make(model, budget_mb=100)
    other = whole.new_cache()
    for token in TOKENS:
        assert np.array_equal(engine.forward([token], cache), whole.forward([token], other))


def test_memory_stays_inside_the_budget(model):
    engine = make(model)
    engine.forward(TOKENS, engine.new_cache())
    assert engine.store.allocated <= engine.store.capacity == 7
    assert len(engine.store.slots) <= 7
    assert engine.store.busy == {} and engine.store.loading == {}


def test_everything_in_memory_reads_each_expert_once(model):
    engine = make(model, budget_mb=100)
    cache = engine.new_cache()
    for token in TOKENS:
        engine.forward([token], cache)
    assert engine.store.stats["bytes"] <= 24 * engine.store.stride


def test_budget_too_small_is_refused(model):
    with pytest.raises(ValueError, match="at least"):
        Engine(model, budget_mb=0)


def test_context_limit(model):
    engine = make(model)
    with pytest.raises(ValueError, match="longer than"):
        engine.forward(TOKENS, engine.new_cache(limit=10))


def test_read_error_is_raised_and_leaves_the_cache_usable(model, monkeypatch):
    engine = make(model, prefetch=False)
    real, calls = engine.store._read.__globals__["os"].preadv, []

    def failing(*args):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk error")
        return real(*args)

    monkeypatch.setattr("expertstream.store.os.preadv", failing)
    with pytest.raises(OSError, match="disk error"):
        engine.forward([5], engine.new_cache())
    assert engine.store.busy == {} and engine.store.loading == {}
    cache = engine.new_cache()
    for i, token in enumerate(TOKENS):
        np.testing.assert_allclose(engine.forward([token], cache), LOGITS[i], atol=1e-3)


def test_kernels_match_numpy():
    random = np.random.default_rng(0)
    bits = random.normal(size=(37, 53)).astype(np.float32).view(np.uint32) >> 16
    w = bits.astype(np.uint16)
    full = (w.astype(np.uint32) << 16).view(np.float32)
    x = random.normal(size=53).astype(np.float32)
    assert np.array_equal(kernel.to_f32(w), full)
    np.testing.assert_allclose(kernel.matvec(w, x), full @ x, rtol=1e-4, atol=1e-5)


def test_pinned_policy_gives_way_when_the_cache_is_tiny():
    from expertstream.policies import Pinned

    policy = Pinned(4, layers=2, hot=[(0, 0), (0, 1), (0, 2)])
    for key in [(0, 0), (0, 1), (0, 2), (1, 5)]:
        policy.add(key)
    assert policy.victim((1, 6), busy={(1, 5)}) in {(0, 0), (0, 1), (0, 2)}


@pytest.mark.parametrize("policy", ["lru", "layer", "lfu", "pinned"])
def test_smallest_cache_works_with_every_policy(model, policy):
    hot = [(layer, e) for layer in range(3) for e in range(8)]
    stride = json.loads((model / "meta.json").read_text())["stride"]
    engine = make(model, policy=policy, hot=hot, budget_mb=6 * stride / 1e6)
    cache = engine.new_cache()
    for i, token in enumerate(TOKENS):
        np.testing.assert_allclose(engine.forward([token], cache), LOGITS[i], atol=1e-3)


def test_conversation_buffer_grows_across_steps(model, monkeypatch):
    monkeypatch.setattr("expertstream.engine.KVCache.STEP", 8)
    engine = make(model)
    cache = engine.new_cache(limit=41)
    for i, token in enumerate(TOKENS):
        np.testing.assert_allclose(engine.forward([token], cache), LOGITS[i], atol=1e-3)
    assert cache.k[0].shape[1] == 40

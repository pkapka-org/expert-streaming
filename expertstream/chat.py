"""Tokenizer and Granite's chat format."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

ROLES = {"system", "user", "assistant"}


class Chat:
    def __init__(self, model: str | Path):
        self.tokenizer = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))
        self.special = {t: self.tokenizer.token_to_id(t)
                        for t in ("<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>")}

    def _plain(self, text: str) -> list[int]:
        """Encode text so that special tokens written inside it stay ordinary text."""
        for token in self.special:
            text = text.replace(token, token.replace("|", "​|"))
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode(self, messages: list[dict]) -> list[int]:
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}, *messages]
        start, end, stop = self.special.values()
        ids: list[int] = []
        for m in messages:
            if m["role"] not in ROLES:
                raise ValueError(f"unknown role: {m['role']}")
            ids += [start, *self._plain(m["role"]), end, *self._plain(m["content"]), stop, *self._plain("\n")]
        return [*ids, start, *self._plain("assistant"), end]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

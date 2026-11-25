"""Split a GraniteMoe checkpoint into always-loaded weights and an expert file.

    python -m expertstream.convert models/src models/granite

experts.bin holds one fixed-size, 4096-byte-aligned block per expert, so each expert is one
direct read. No machine-learning library is needed: safetensors is a JSON header plus raw bytes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ALIGN = 4096


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def convert(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    if config["model_type"] != "granitemoe":
        raise SystemExit(f"unsupported model type: {config['model_type']}")
    header, base = read_header(src / "model.safetensors")
    layers, experts = config["num_hidden_layers"], config["num_local_experts"]

    def span(name: str) -> tuple[int, int]:
        begin, end = header[name]["data_offsets"]
        if header[name]["dtype"] != "BF16":
            raise SystemExit(f"{name}: expected BF16 weights")
        return base + begin, end - begin

    expert_names = {f"model.layers.{n}.block_sparse_moe.{part}_linear.weight"
                    for n in range(layers) for part in ("input", "output")}
    first = "model.layers.0.block_sparse_moe."
    used = (span(first + "input_linear.weight")[1] + span(first + "output_linear.weight")[1]) // experts
    stride = -(-used // ALIGN) * ALIGN

    with (src / "model.safetensors").open("rb") as f, (dst / "experts.bin").open("wb") as out:
        for layer in range(layers):
            parts = [span(f"model.layers.{layer}.block_sparse_moe.{p}_linear.weight")
                     for p in ("input", "output")]
            for e in range(experts):
                written = 0
                for start, total in parts:
                    size = total // experts
                    f.seek(start + e * size)
                    written += out.write(f.read(size))
                out.write(b"\0" * (stride - written))

    dense, position = {}, 0
    with (src / "model.safetensors").open("rb") as f, (dst / "dense.bin").open("wb") as out:
        for name, info in header.items():
            if name in expert_names:
                continue
            start, size = span(name)
            f.seek(start)
            out.write(f.read(size))
            dense[name] = {"offset": position, "shape": info["shape"]}
            position += size

    meta = {"config": config, "stride": stride, "dense": dense}
    (dst / "meta.json").write_text(json.dumps(meta))
    if (src / "tokenizer.json").exists():
        shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    return meta


if __name__ == "__main__":
    result = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    for name in ("experts.bin", "dense.bin"):
        print(f"{name}: {(Path(sys.argv[2]) / name).stat().st_size / 1e6:.0f} MB")

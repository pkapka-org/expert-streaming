"""Make tests/fixture: a tiny random GraniteMoe and the logits Hugging Face Transformers gives for it.

Needs torch and transformers, which the engine itself does not use.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixture"
torch.manual_seed(1)
config = dict(
    model_type="granitemoe", vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, num_local_experts=8, num_experts_per_tok=2,
    attention_multiplier=0.0625, embedding_multiplier=12.0, residual_multiplier=0.22, logits_scaling=6.0,
    rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True, eos_token_id=0, bos_token_id=0,
    pad_token_id=0,
)
model = GraniteMoeForCausalLM(GraniteMoeConfig(**{k: v for k, v in config.items() if k != "model_type"}))
with torch.no_grad():
    for p in model.parameters():
        p.mul_(8)  # the default initial weights are too small to make the experts matter
    # Round the weights to what bf16 can hold, but compute in float32. Calling model.to(bfloat16)
    # would also round the rotary position buffer, which is not part of the checkpoint.
    for p in model.parameters():
        p.copy_(p.to(torch.bfloat16).to(torch.float32))
model.eval()
NAMES = {  # Transformers 5 names -> names in the published checkpoint
    "experts.gate_up_proj": "input_linear.weight", "experts.down_proj": "output_linear.weight",
    "router.weight": "router.layer.weight",
}


def published(name: str) -> str:
    for new, old in NAMES.items():
        name = name.replace("block_sparse_moe." + new, "block_sparse_moe." + old)
    return name


weights = {published(k): v.to(torch.bfloat16).contiguous().clone() for k, v in model.state_dict().items()
           if k != "lm_head.weight"}
save_file(weights, OUT / "model.safetensors")
(OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
tokens = torch.randint(1, 128, (40,))
with torch.no_grad():
    logits = model(tokens[None]).logits[0]
    greedy = model.generate(tokens[None], max_new_tokens=12, do_sample=False, eos_token_id=None)[0, 40:]
(OUT / "reference.json").write_text(json.dumps({
    "tokens": tokens.tolist(), "logits": [[round(x, 5) for x in row] for row in logits.tolist()],
    "greedy": greedy.tolist()}) + "\n")

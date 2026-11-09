
import json, torch
from transformers import AutoConfig, AutoModelForCausalLM
cands = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
out = {}
for c in cands:
    try:
        cfg = AutoConfig.from_pretrained(c)
        with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg)
        n = sum(p.numel() for p in m.parameters())
        L, H = cfg.num_hidden_layers, cfg.num_attention_heads; Hk = getattr(cfg, "num_key_value_heads", H); D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
        lic = "Apache-2.0" if "Qwen2.5-3B" not in c else "Qwen-research (non-commercial)"
        out[c] = {"params_B": n / 1e9, "layers": L, "hidden": cfg.hidden_size, "heads": H, "kv_heads": Hk, "head_dim": D, "vocab": cfg.vocab_size, "max_positions": cfg.max_position_embeddings,
                  "fp16_weights_GB": n * 2 / 2**30, "kv_bytes_per_token_fp16": 2 * L * Hk * D * 2, "tied_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)), "license_note": lic}
        print(c, out[c])
    except Exception as ex:
        out[c] = {"error": str(ex)[:150]}; print(c, "ERR", str(ex)[:100])
json.dump(out, open("/content/work/model_facts.json", "w"), indent=1)

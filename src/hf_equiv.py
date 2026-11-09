
import sys, json, torch
sys.path.insert(0, "/content/work")
from common import *
from minillm import *
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(MID)
b = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).cuda()
T_hf = PeftModel.from_pretrained(b, sys.argv[1]).merge_and_unload().eval(); target = MiniQwen(T_hf, "sdpa")
d = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float16).cuda(); D_hf = PeftModel.from_pretrained(d, sys.argv[2]).merge_and_unload().eval()
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
gc = GenerationConfig(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
out = {"hf_default_generation_config": {k: v for k, v in AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16).generation_config.to_dict().items() if k in ("repetition_penalty", "temperature", "top_p", "top_k", "do_sample")}}
print(out, flush=True)
test = load_split("test")
P = [tok(build_prompt(tok, e["note"], e["criterion_type"], e["criterion_text"]), add_special_tokens=False).input_ids for e in test]
long_ids = [t for p in P[:12] for t in p]
rows = []
for L in [128, 512, 1024]:
    for N in [16, 64, 128]:
        ids = torch.tensor([long_ids[:L]], device="cuda"); a = greedy_cached(target, ids, N)
        h = T_hf.generate(ids, generation_config=gc, max_new_tokens=N, eos_token_id=None)[0, L:].tolist()
        rows.append({"in_len": L, "out_len": N, "cached_eq_hf_pure_greedy": a == h}); print(rows[-1], flush=True)
out["equivalence_hf_penalty_off"] = rows
same = 0; n = len(P)
for i in range(n):
    ids = torch.tensor([P[i]], device="cuda"); a = greedy_cached(target, ids, 64, EOS)
    h = T_hf.generate(ids, generation_config=gc, max_new_tokens=64, eos_token_id=list(EOS))[0, ids.shape[1]:].tolist()
    h = h[:[j for j, t in enumerate(h) if t in EOS][0] + 1] if any(t in EOS for t in h) else h
    same += int(a == h)
out["test_prompts_engine_vs_hf_pure_greedy"] = {"n": n, "identical": same}; print(out["test_prompts_engine_vs_hf_pure_greedy"], flush=True)
ok = 0
for i in list(range(0, n, 5))[:20]:
    ids = torch.tensor([P[i]], device="cuda")
    a = T_hf.generate(ids, generation_config=gc, assistant_model=D_hf, max_new_tokens=64, eos_token_id=list(EOS))[0, ids.shape[1]:].tolist()
    a = a[:[j for j, t in enumerate(a) if t in EOS][0] + 1] if any(t in EOS for t in a) else a
    ok += int(a == greedy_cached(target, ids, 64, EOS))
out["hf_assisted_vs_engine"] = {"n": 20, "identical": ok}; print(out["hf_assisted_vs_engine"], flush=True)
json.dump(out, open("/content/work/hf_equiv.json", "w"), indent=1); print("HFEQ DONE", flush=True)

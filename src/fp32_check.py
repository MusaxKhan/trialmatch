
import sys, json, torch
sys.path.insert(0, "/content/work")
from common import *
from minillm import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(MID)
def load(mid, ad):
    b = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).cuda()
    return PeftModel.from_pretrained(b, ad).merge_and_unload().eval()
T_hf, D_hf = load(MID, sys.argv[1]), load(DRAFT, sys.argv[2])
target, draft = MiniQwen(T_hf, "sdpa"), MiniQwen(D_hf, "sdpa")
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
test = load_split("test")
P = [tok(build_prompt(tok, e["note"], e["criterion_type"], e["criterion_text"]), add_special_tokens=False).input_ids for e in test]
long_ids = [t for p in P[:12] for t in p]
out = {}
ids = torch.tensor([P[0]], device="cuda")
lg = target.forward(ids, target.new_cache(ids.shape[1])); hf = T_hf(ids).logits
out["forward_vs_hf_fp32"] = {"max_abs_logit_diff": (lg - hf).abs().max().item(), "argmax_agree_all_positions": bool((lg.argmax(-1) == hf.argmax(-1)).all().item()), "T": ids.shape[1]}
print(out["forward_vs_hf_fp32"], flush=True)
rows = []
for L, N in [(128, 64), (512, 64), (1024, 64)]:
    ids = torch.tensor([long_ids[:L]], device="cuda")
    a = greedy_nocache(target, ids, N); b = greedy_cached(target, ids, N)
    h = T_hf.generate(ids, max_new_tokens=N, do_sample=False, eos_token_id=None, pad_token_id=tok.pad_token_id)[0, L:].tolist()
    rows.append({"in_len": L, "out_len": N, "cached_eq_nocache": a == b, "cached_eq_hf": b == h}); print(rows[-1], flush=True)
out["equivalence_fp32"] = rows
sub = list(range(0, len(P), 5))[:20]; ok = 0; acc = prop = 0
for i in sub:
    ids = torch.tensor([P[i]], device="cuda"); c = greedy_cached(target, ids, 64, EOS); st = {}; s = speculative(target, draft, ids, 64, 4, EOS, stats=st)
    ok += int(s == c); acc += st["accepted"]; prop += st["proposed"]
out["speculative_fp32"] = {"n": len(sub), "token_identical": ok, "acceptance": acc / max(prop, 1)}
print(out["speculative_fp32"], flush=True)
json.dump(out, open("/content/work/fp32_check.json", "w"), indent=1); print("FP32 DONE", flush=True)

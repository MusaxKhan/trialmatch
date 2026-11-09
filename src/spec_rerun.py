
import sys, json, time, statistics, torch
sys.path.insert(0, "/content/work")
from common import *
from minillm import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
adapter_t, adapter_d = sys.argv[1], sys.argv[2]
OUT = json.load(open("/content/work/decode_results.json"))
tok = AutoTokenizer.from_pretrained(MID)
def load(mid, ad):
    b = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).cuda()
    m = PeftModel.from_pretrained(b, ad).merge_and_unload(); m.eval(); return m
T_hf, D_hf = load(MID, adapter_t), load(DRAFT, adapter_d)
target, draft = MiniQwen(T_hf, "sdpa"), MiniQwen(D_hf, "sdpa")
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
test = load_split("test")
P = [tok(build_prompt(tok, e["note"], e["criterion_type"], e["criterion_text"]), add_special_tokens=False).input_ids for e in test]
long_ids = [t for p in P[:12] for t in p]
def ids_of(L): return torch.tensor([long_ids[:L]], device="cuda")
def save(): json.dump(OUT, open("/content/work/decode_results.json", "w"), indent=1)
def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return -1 if len(a) == len(b) else min(len(a), len(b))

# 5. speculative decoding: equivalence + latency
sub = list(range(0, len(P), 5))[:40]
spec = []
for K in [2, 4, 6]:
    eqn = 0; Ts, Tc, acc, prop, rounds, ntok = 0.0, 0.0, 0, 0, 0, 0; mism = []
    for i in sub:
        ids = torch.tensor([P[i]], device="cuda")
        st = {}; c = greedy_cached(target, ids, 64, EOS, stats=st); Tc += st["total"]
        st = {}; s = speculative(target, draft, ids, 64, K, EOS, stats=st); Ts += st["total"]
        acc += st["accepted"]; prop += st["proposed"]; rounds += st["rounds"]; ntok += st["n"]
        if s == c: eqn += 1
        else: mism.append({"i": i, "first_mismatch": first_mismatch(s, c)})
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b0 = torch.cuda.memory_allocated()
    speculative(target, draft, torch.tensor([P[sub[0]]], device="cuda"), 64, K, EOS)
    pk = (torch.cuda.max_memory_allocated() - b0) / 2**20
    r = {"K": K, "n_prompts": len(sub), "token_identical": eqn, "mismatches": mism, "acceptance_rate": acc / max(prop, 1), "tokens_per_round": ntok / max(rounds, 1), "latency_spec_s": Ts / len(sub), "latency_cached_s": Tc / len(sub), "speedup": Tc / Ts, "tok_per_s_spec": ntok / Ts, "peak_extra_MB_spec": pk}
    spec.append(r); print("5 spec", r, flush=True)
OUT["speculative"] = spec; save()
try:
    ok = 0
    for i in sub[:8]:
        ids = torch.tensor([P[i]], device="cuda")
        a = T_hf.generate(ids, assistant_model=D_hf, max_new_tokens=64, do_sample=False, eos_token_id=list(EOS), pad_token_id=tok.pad_token_id)[0, ids.shape[1]:].tolist()
        a = a[:[j for j, t in enumerate(a) if t in EOS][0] + 1] if any(t in EOS for t in a) else a
        ok += int(a == greedy_cached(target, ids, 64, EOS))
    OUT["hf_assisted_reference_equal"] = {"n": 8, "equal": ok}; print("5b HF assisted equal:", ok, "/ 8", flush=True)
except Exception as ex:
    OUT["hf_assisted_reference_equal"] = str(ex)[:200]
save(); print("DECODE TESTS DONE", flush=True)

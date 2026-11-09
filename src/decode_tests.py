
import sys, json, time, statistics, torch
sys.path.insert(0, "/content/work")
from common import *
from minillm import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
adapter_t, adapter_d = sys.argv[1], sys.argv[2]
OUT = {}
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

# 0. my forward vs HF forward
ids = torch.tensor([P[0]], device="cuda")
lg = target.forward(ids, target.new_cache(ids.shape[1])).float(); hf = T_hf(ids).logits.float()
OUT["forward_vs_hf"] = {"max_abs_logit_diff": (lg - hf).abs().max().item(), "argmax_agree_all_positions": bool((lg.argmax(-1) == hf.argmax(-1)).all().item()), "T": ids.shape[1]}
print("0 forward vs HF:", OUT["forward_vs_hf"], flush=True)

# 1. cached vs no-cache equivalence (forced length, no EOS) + HF generate reference
eq = []
for L in [128, 512, 1024]:
    for N in [16, 64, 128]:
        ids = ids_of(L)
        a = greedy_nocache(target, ids, N); b = greedy_cached(target, ids, N)
        h1 = T_hf.generate(ids, max_new_tokens=N, do_sample=False, eos_token_id=None, pad_token_id=tok.pad_token_id, use_cache=True)[0, L:].tolist()
        eq.append({"in_len": L, "out_len": N, "cached_eq_nocache": a == b, "first_mismatch_cached_vs_nocache": first_mismatch(a, b), "cached_eq_hf": b == h1, "first_mismatch_cached_vs_hf": first_mismatch(b, h1)})
        print("1 eq", eq[-1], flush=True)
OUT["equivalence_cache"] = eq; save()

# 2. real test prompts: MiniQwen cached engine vs HF generate used for the reported metrics
tr = json.load(open("/content/work/q15/test_results.json"))["tuned_test"]["recs"] if len(sys.argv) > 3 else None
real = []; t0 = time.time(); same = 0; n_tok = 0; texts = []
for i in range(len(P)):
    o = greedy_cached(target, torch.tensor([P[i]], device="cuda"), 64, EOS); n_tok += len(o)
    txt = tok.decode([t for t in o if t not in EOS]); texts.append(txt)
    if tr is not None and txt.strip() == tr[i]["raw"].strip(): same += 1
OUT["engine_vs_hf_generate_test"] = {"n": len(P), "identical_raw_output": same if tr is not None else None, "tokens": n_tok, "seconds": time.time() - t0}
print("2 engine on test:", OUT["engine_vs_hf_generate_test"], flush=True); save()

# 3. tiled attention inside the full model
tm = MiniQwen(T_hf, "tiled"); res = []
for i in range(4):
    ids = torch.tensor([P[i]], device="cuda")
    a = greedy_cached(target, ids, 24, EOS); b = greedy_cached(tm, ids, 24, EOS)
    res.append({"prompt": i, "equal": a == b, "first_mismatch": first_mismatch(a, b)})
OUT["tiled_in_model"] = res; print("3 tiled in model:", res, flush=True); save()

# 4. cache benchmark
def run_cfg(fn, reps=3, warm=1):
    for _ in range(warm): fn({})
    torch.cuda.synchronize(); torch.cuda.empty_cache(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    S = []
    for _ in range(reps):
        st = {}; fn(st); S.append(st)
    med = {k: statistics.median(s[k] for s in S) for k in S[0]}
    med["peak_extra_MB"] = (torch.cuda.max_memory_allocated() - base) / 2**20
    med["tok_per_s"] = med["n"] / med["total"]
    med["decode_tok_per_s"] = (med["n"] - 1) / max(med["total"] - med["prefill"], 1e-9)
    return med
bench = []
for L, N in [(128, 32), (128, 128), (512, 32), (512, 128), (1024, 32), (1024, 128), (2048, 32)]:
    ids = ids_of(L)
    nc = run_cfg(lambda st: greedy_nocache(target, ids, N, stats=st), reps=2)
    ca = run_cfg(lambda st: greedy_cached(target, ids, N, stats=st), reps=3)
    row = {"in_len": L, "out_len": N, "nocache": nc, "cached": ca, "speedup_total": nc["total"] / ca["total"]}
    bench.append(row); print("4", L, N, "nocache %.2fs cached %.2fs speedup %.1fx; cached prefill %.3fs decode %.1f tok/s peakMB %.0f vs %.0f" % (nc["total"], ca["total"], row["speedup_total"], ca["prefill"], ca["decode_tok_per_s"], ca["peak_extra_MB"], nc["peak_extra_MB"]), flush=True)
OUT["bench_cache"] = bench; OUT["kv_bytes_per_token"] = {"target_1.5B": target.kv_bytes_per_token(), "draft_0.5B": draft.kv_bytes_per_token()}; save()

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

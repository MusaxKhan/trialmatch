
import sys, json, time, statistics, torch
sys.path.insert(0, "/content/work")
from common import *
from minillm import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
tok = AutoTokenizer.from_pretrained(MID)
def load(mid, ad):
    b = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).cuda()
    return PeftModel.from_pretrained(b, ad).merge_and_unload().eval()
target, draft = MiniQwen(load(MID, sys.argv[1]), "sdpa"), MiniQwen(load(DRAFT, sys.argv[2]), "sdpa")
EOS = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
test = load_split("test"); out = {}
# 1. per-token decode latency (batch 1, 512-token context): target vs draft  -> cost ratio c
P = [tok(build_prompt(tok, e["note"], e["criterion_type"], e["criterion_text"]), add_special_tokens=False).input_ids for e in test]
ids = torch.tensor([[t for p in P[:3] for t in p][:512]], device="cuda")
def step_ms(m):
    c = m.new_cache(700); m.forward(ids, c, last_only=True); x = ids[:, -1:]
    for _ in range(5): m.forward(x, c, last_only=True)
    torch.cuda.synchronize(); ts = []
    for _ in range(40):
        t = time.perf_counter(); m.forward(x, c, last_only=True); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)
tm, dm = step_ms(target), step_ms(draft)
out["decode_step_ms"] = {"target_1.5B": tm, "draft_0.5B": dm, "cost_ratio_c": dm / tm}
print(out["decode_step_ms"], flush=True)
# 2. 4-token verification pass vs 1-token pass of the target (cost of verifying K+1 tokens)
c = target.new_cache(700); target.forward(ids, c, last_only=True); xs = ids[:, -5:]
ts = []
for _ in range(25):
    c.crop(512); torch.cuda.synchronize(); t = time.perf_counter(); target.forward(xs, c); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
out["verify_5tokens_ms"] = statistics.median(ts); print("verify 5 tokens ms", out["verify_5tokens_ms"], flush=True)
# 3. full held-out test through OUR engine (fp16, batch 1, KV cache) -> metrics
recs = []
for i, e in enumerate(test):
    o = greedy_cached(target, torch.tensor([P[i]], device="cuda"), 64, EOS)
    txt = tok.decode([t for t in o if t not in EOS]); lab, ev, ok = parse_output(txt, e["criterion_type"])
    recs.append({"annotation_id": e["annotation_id"], "patient_id": e["patient_id"], "type": e["criterion_type"], "gold": e["expert_eligibility"], "pred": lab, "gold_ev": sorted(e["expert_sentences"]), "pred_ev": ev, "valid": ok, "raw": txt})
m = metrics(recs); json.dump({"metrics": m, "recs": recs}, open("/content/work/engine_test_results.json", "w"))
out["engine_test"] = {k: (round(v["accuracy"], 3), round(v["macro_f1"], 3), round(v["evidence"]["micro_F1"], 3), v["invalid_outputs"]) for k, v in m.items()}
print("engine test", out["engine_test"], flush=True)
json.dump(out, open("/content/work/extras.json", "w"), indent=1); print("EXTRA DONE", flush=True)

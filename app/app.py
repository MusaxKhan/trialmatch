
"""TrialMatch screening app (educational prototype, NOT a clinical enrolment tool).
Run:  uvicorn app:app --host 127.0.0.1 --port 8000     then open http://127.0.0.1:8000
"""
import os, re, sys, json, time, math, hashlib, threading
import torch
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import LABELS, build_prompt, parse_output, MID, DRAFT
from minillm import MiniQwen, greedy_cached, greedy_nocache, speculative, ref_attention, tiled_attention, sdpa_attention

ADAPTER_T = os.environ.get("ADAPTER_T", os.path.join(HERE, "adapters", "target"))
ADAPTER_D = os.environ.get("ADAPTER_D", os.path.join(HERE, "adapters", "draft"))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32
MAXNEW = 64
S = {"ready": False, "error": None, "loading": "starting", "info": {}}
LOCK = threading.Lock()
app = FastAPI(title="TrialMatch screening prototype")

def _load():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        S["loading"] = "tokenizer"; tok = AutoTokenizer.from_pretrained(MID)
        def ld(mid, ad, name):
            S["loading"] = "loading " + name
            m = AutoModelForCausalLM.from_pretrained(mid, dtype=DT).to(DEV)
            tuned = os.path.isdir(ad) and os.environ.get("TRIALMATCH_BASE") != "1"
            if tuned: m = PeftModel.from_pretrained(m, ad).merge_and_unload()
            return m.eval(), tuned
        t, tt = ld(MID, ADAPTER_T, "target"); d, dt = ld(DRAFT, ADAPTER_D, "draft")
        S.update(tok=tok, target=MiniQwen(t, "sdpa"), draft=MiniQwen(d, "sdpa"),
                 eos={tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}, ready=True, loading="ready")
        h = hashlib.sha256(open(os.path.join(ADAPTER_T, "adapter_model.safetensors"), "rb").read()).hexdigest()[:16] if tt else "none (base weights)"
        S["info"] = {"target": MID, "draft": DRAFT, "target_finetuned": tt, "draft_finetuned": dt, "adapter_sha256_16": h, "device": DEV, "dtype": str(DT).split(".")[1]}
    except Exception as ex:
        S["error"] = repr(ex); S["loading"] = "failed"
threading.Thread(target=_load, daemon=True).start()

# ---------- input handling ----------
def split_sentences(raw):
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines and all(re.match(r"^\d+\.\s", l) for l in lines[:min(2, len(lines))]):
        return [re.sub(r"^\d+\.\s*", "", l) for l in lines]
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", " ".join(lines))
    return [p.strip() for p in parts if p.strip()]
class Crit(BaseModel):
    type: str
    text: str
class Req(BaseModel):
    note: str
    criteria: List[Crit]
    mode: str = "cached"
    K: int = 4
def validate(r):
    if len(r.note.strip()) < 20: raise HTTPException(422, "Patient note is empty or too short (min 20 characters).")
    if len(r.note) > 8000: raise HTTPException(422, "Patient note too long (max 8000 characters); the model was trained on notes under ~1500 characters.")
    if not (1 <= len(r.criteria) <= 30): raise HTTPException(422, "Provide between 1 and 30 criteria.")
    for i, c in enumerate(r.criteria):
        if c.type not in ("inclusion", "exclusion"): raise HTTPException(422, "Criterion %d: type must be inclusion or exclusion." % (i + 1))
        if not (5 <= len(c.text.strip()) <= 1000): raise HTTPException(422, "Criterion %d: text must be 5-1000 characters." % (i + 1))
    if r.mode not in ("cached", "nocache", "speculative"): raise HTTPException(422, "Unknown inference mode.")
    if not (1 <= r.K <= 8): raise HTTPException(422, "K must be between 1 and 8.")
    if not S["ready"]: raise HTTPException(503, "Model not ready: " + (S["error"] or S["loading"]))

def run_one(note_numbered, sents, ctype, ctext, mode, K):
    tok = S["tok"]; ids = torch.tensor([tok(build_prompt(tok, note_numbered, ctype, ctext), add_special_tokens=False).input_ids], device=DEV)
    st = {}
    if mode == "nocache": out = greedy_nocache(S["target"], ids, MAXNEW, S["eos"], st)
    elif mode == "speculative": out = speculative(S["target"], S["draft"], ids, MAXNEW, K, S["eos"], st)
    else: out = greedy_cached(S["target"], ids, MAXNEW, S["eos"], st)
    raw = tok.decode([t for t in out if t not in S["eos"]])
    label, ev, ok = parse_output(raw, ctype)
    flags = []
    bad = [e for e in ev if e >= len(sents)]
    if bad: flags.append("cited sentence id(s) %s do not exist in the note" % bad); ev = [e for e in ev if e < len(sents)]
    if not ok: flags.append("model output could not be parsed into a valid label")
    definitive = label in ("included", "not included", "excluded", "not excluded")
    if definitive and not ev: flags.append("unsupported: definitive label without any cited evidence")
    ev_txt = [{"id": e, "text": sents[e]} for e in ev]
    if not ok: expl = "No valid prediction was produced for this criterion."
    elif label == "not applicable": expl = "The model judged the criterion not applicable to this patient."
    elif label == "not enough information" and not ev: expl = "The note does not contain information that supports or contradicts this criterion; absence of evidence is not treated as eligibility or ineligibility."
    else: expl = "Label '%s' is based on the cited sentence(s) only: %s" % (label, " | ".join("[%d] %s" % (e["id"], e["text"]) for e in ev_txt) or "none cited")
    return {"criterion_type": ctype, "criterion": ctext, "label": label, "evidence_ids": ev, "evidence": ev_txt, "explanation": expl, "flags": flags, "raw_output": raw, "tokens": out, "n_tokens": len(out), "seconds": st.get("total"), "stats": {k: v for k, v in st.items() if k in ("acceptance", "tokens_per_round", "proposed", "accepted", "rounds", "prefill")}}

def verdict(rs):
    inc = [r["label"] for r in rs if r["criterion_type"] == "inclusion"]; exc = [r["label"] for r in rs if r["criterion_type"] == "exclusion"]
    why = []
    if any(l == "INVALID" for l in inc + exc): return "Incomplete: some criteria could not be assessed", ["invalid model output for at least one criterion"]
    if "excluded" in exc: return "Likely NOT eligible (an exclusion criterion is met)", ["rule R1: any exclusion criterion judged 'excluded'"]
    if "not included" in inc: return "Likely NOT eligible (an inclusion criterion is not met)", ["rule R2: any inclusion criterion judged 'not included'"]
    if "not enough information" in inc + exc: return "Undetermined: more information needed", ["rule R3: no criterion violated, but at least one has insufficient information"]
    return "Preliminarily eligible (all assessed criteria satisfied)", ["rule R4: all applicable inclusion criteria 'included' and exclusion criteria 'not excluded'"]

@app.get("/api/status")
def status(): return {"ready": S["ready"], "loading": S["loading"], "error": S["error"], **S["info"]}

@app.post("/api/screen")
def screen(r: Req):
    validate(r); sents = split_sentences(r.note)
    if not sents: raise HTTPException(422, "Could not split the note into sentences.")
    numbered = "\n".join("%d. %s" % (i, s) for i, s in enumerate(sents))
    t0 = time.time()
    with LOCK: rs = [run_one(numbered, sents, c.type, c.text.strip(), r.mode, r.K) for c in r.criteria]
    v, why = verdict(rs)
    return {"mode": r.mode, "K": r.K, "sentences": sents, "results": rs, "overall": v, "overall_rules": why, "seconds": time.time() - t0,
            "provenance": {**S["info"], "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "max_new_tokens": MAXNEW},
            "disclaimer": "Educational screening prototype. Not a clinical enrolment decision."}

@app.post("/api/compare")
def compare(r: Req):
    validate(r); sents = split_sentences(r.note)
    numbered = "\n".join("%d. %s" % (i, s) for i, s in enumerate(sents)); crits = r.criteria[:6]
    out = {}; ref_tokens = None
    with LOCK:
        for mode in ["nocache", "cached", "speculative"]:
            if DEV == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b0 = torch.cuda.memory_allocated()
            t0 = time.time(); rs = [run_one(numbered, sents, c.type, c.text.strip(), mode, r.K) for c in crits]
            if DEV == "cuda": torch.cuda.synchronize()
            dt = time.time() - t0; toks = [x["tokens"] for x in rs]; n = sum(len(t) for t in toks)
            if mode == "cached": ref_tokens = toks
            out[mode] = {"seconds": dt, "tokens": n, "tok_per_s": n / dt, "peak_extra_MB": (torch.cuda.max_memory_allocated() - b0) / 2**20 if DEV == "cuda" else None, "labels": [x["label"] for x in rs], "_toks": toks,
                         "acceptance": (sum(x["stats"].get("accepted", 0) for x in rs) / max(sum(x["stats"].get("proposed", 0) for x in rs), 1)) if mode == "speculative" else None}
    for m in out: out[m]["tokens_identical_to_cached"] = out[m].pop("_toks") == ref_tokens
    return {"criteria_evaluated": len(crits), "modes": out, "device": DEV}

class AttnReq(BaseModel):
    T: int = 512
    heads: int = 8
    dim: int = 64
    causal: bool = True
    block: int = 64
@app.post("/api/attention")
def attention(a: AttnReq):
    if not (8 <= a.T <= 4096 and 1 <= a.heads <= 32 and a.dim in (32, 64, 128) and 8 <= a.block <= 512): raise HTTPException(422, "T in [8,4096], heads in [1,32], dim in {32,64,128}, block in [8,512].")
    g = torch.Generator(device=DEV).manual_seed(0)
    q, k, v = [torch.randn(1, a.heads, a.T, a.dim, device=DEV, generator=g).to(DT) for _ in range(3)]
    s = (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(a.dim)
    if a.causal: s = s.masked_fill(~torch.ones(a.T, a.T, dtype=torch.bool, device=DEV).tril(), float("-inf"))
    gt = torch.softmax(s, -1) @ v.double(); res = {}
    with LOCK:
        for name, fn in [("reference", lambda: ref_attention(q, k, v, a.causal)), ("tiled_online_softmax", lambda: tiled_attention(q, k, v, a.causal, a.block, a.block)), ("pytorch_sdpa", lambda: sdpa_attention(q, k, v, a.causal))]:
            fn()
            if DEV == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b0 = torch.cuda.memory_allocated()
            t0 = time.perf_counter(); o = fn()
            if DEV == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0; d = (o.double() - gt).abs()
            res[name] = {"ms": dt * 1000, "max_abs_err_vs_fp64": d.max().item(), "mean_abs_err": d.mean().item(), "peak_extra_MB": (torch.cuda.max_memory_allocated() - b0) / 2**20 if DEV == "cuda" else None}
    return {"config": a.dict(), "dtype": str(DT).split(".")[1], "device": DEV, "results": res}

@app.get("/api/results")
def results():
    out = {}
    for k, f in [("decode", "decode_results.json"), ("attn_numeric", "attn_numeric.json"), ("attn_perf", "attn_perf.json"), ("metrics", "metrics_summary.json"), ("training", "train_log.json")]:
        p = os.path.join(RESULTS, f)
        out[k] = json.load(open(p)) if os.path.exists(p) else None
    return out

@app.get("/", response_class=HTMLResponse)
def index(): return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()

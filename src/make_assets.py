
import json, os, sys, shutil, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = "/content/work"; A = W + "/report/assets"; os.makedirs(A, exist_ok=True)
sel, alt = sys.argv[1], sys.argv[2]
L = lambda p: json.load(open(p))
tl = L("%s/%s/train_log.json" % (W, sel)); tla = L("%s/%s/train_log.json" % (W, alt)) if os.path.exists("%s/%s/train_log.json" % (W, alt)) else None
res = L("%s/%s/test_results.json" % (W, sel)); dec = L(W + "/decode_results.json"); an = L(W + "/attn_numeric.json"); ap = L(W + "/attn_perf.json"); facts = L(W + "/model_facts.json")
vb = L("%s/%s/val_base.json" % (W, sel)) if os.path.exists("%s/%s/val_base.json" % (W, sel)) else None
plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True})
def esc(s): return str(s).replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
def save(n): plt.tight_layout(); plt.savefig("%s/%s.pdf" % (A, n)); plt.savefig("%s/%s.png" % (A, n), dpi=200); plt.close()
def tab(name, s): open("%s/%s.tex" % (A, name), "w").write(s)
f3 = lambda x: "%.3f" % x
# ---- training curves
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1))
st = [r["step"] for r in tl["train_log"]]; lo = [r["loss"] for r in tl["train_log"]]; k = 5
ax[0].plot(st, lo, alpha=.35, label="per step"); ax[0].plot(st[k - 1:], np.convolve(lo, np.ones(k) / k, mode="valid"), label="moving avg (5)"); ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("train loss (completion tokens)"); ax[0].set_yscale("log"); ax[0].legend()
for t, nm in [(tl, "v1 (balance power 0.5)"), (tla, "v2 (balance power 1.0)")]:
    if t is None: continue
    e = t["epoch_log"]; ax[1].plot([x["epoch"] for x in e], [x["val_macro_f1"] for x in e], "o-", label=nm + " macro-F1"); ax[1].plot([x["epoch"] for x in e], [x["val_acc"] for x in e], "s--", alpha=.6, label=nm + " acc")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation"); ax[1].legend(fontsize=5.5); save("fig_train")
# ---- confusion matrices (tuned, test)
tm = res["tuned_test"]["metrics"]; fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
ab = {"included": "incl", "not included": "not incl", "excluded": "excl", "not excluded": "not excl", "not enough information": "NEI", "not applicable": "N/A", "INVALID": "inval"}
for a_, nm in zip(ax, ["inclusion", "exclusion"]):
    c = np.array(tm[nm]["confusion"]); cl = tm[nm]["confusion_labels"]; a_.imshow(c, cmap="Blues")
    for i in range(c.shape[0]):
        for j in range(c.shape[1]): a_.text(j, i, int(c[i, j]), ha="center", va="center", fontsize=6, color="white" if c[i, j] > c.max() / 2 else "black")
    a_.set_xticks(range(len(cl))); a_.set_xticklabels([ab[x] for x in cl], rotation=45, ha="right"); a_.set_yticks(range(len(cl))); a_.set_yticklabels([ab[x] for x in cl]); a_.set_title(nm + " (rows=gold, cols=pred)", fontsize=7); a_.grid(False)
save("fig_conf")
# ---- main results table
rows = [("Majority label per type + sentence 0", res["majority_baseline"]["metrics"]), ("Qwen2.5-1.5B zero-shot (before)", res["base_test"]["metrics"]), ("Qwen2.5-1.5B + LoRA (after)", res["tuned_test"]["metrics"])]
s = "\\begin{tabular}{lcccc}\\toprule System & Acc & Macro-F1 & Evid. $\\mu$-F1 & Invalid\\\\\\midrule\n"
for nm, m in rows:
    o = m["overall"]; s += "%s & %s & %s & %s & %d\\\\\n" % (esc(nm), f3(o["accuracy"]), f3(o["macro_f1"]), f3(o["evidence"]["micro_F1"]), o["invalid_outputs"])
tab("tab_main", s + "\\bottomrule\\end{tabular}")
s = "\\begin{tabular}{llcccc}\\toprule System & Subset & Acc & Macro-F1 & Evid. P/R/F1 & n\\\\\\midrule\n"
for nm, m in rows[1:]:
    for sub in ["inclusion", "exclusion"]:
        o = m[sub]; e = o["evidence"]; s += "%s & %s & %s & %s & %.2f/%.2f/%.2f & %d\\\\\n" % (esc(nm.split(" (")[0] + (" (before)" if "before" in nm else " (after)")), sub, f3(o["accuracy"]), f3(o["macro_f1"]), e["micro_P"], e["micro_R"], e["micro_F1"], o["n"])
tab("tab_split", s + "\\bottomrule\\end{tabular}")
s = "\\begin{tabular}{llcccc}\\toprule Subset & Class & P & R & F1 & Support\\\\\\midrule\n"
for sub in ["inclusion", "exclusion"]:
    for c, v in tm[sub]["per_class"].items(): s += "%s & %s & %s & %s & %s & %d\\\\\n" % (sub, esc(c), f3(v["P"]), f3(v["R"]), f3(v["F1"]), v["support"])
tab("tab_perclass", s + "\\bottomrule\\end{tabular}")

import ast
dist = L(W + "/label_dist.json")
pairs = [("inclusion", "included"), ("inclusion", "not included"), ("inclusion", "not enough information"), ("inclusion", "not applicable"), ("exclusion", "excluded"), ("exclusion", "not excluded"), ("exclusion", "not enough information"), ("exclusion", "not applicable")]
s = "\\begin{tabular}{llrrr}\\toprule Type & Label & Train & Val & Test\\\\\\midrule\n"
s += "\\multicolumn{2}{l}{Annotations (patients)} & %d (%d) & %d (%d) & %d (%d)\\\\\n" % tuple(x for sp in ["train", "val", "test"] for x in (dist[sp]["n"], dist[sp]["patients"]))
for t, lb in pairs:
    cnt = [sum(v for k, v in dist[sp]["by_label"].items() if ast.literal_eval(k) == (t, lb)) for sp in ["train", "val", "test"]]
    s += "%s & %s & %d & %d & %d\\\\\n" % (t, esc(lb.replace("not enough information", "not enough info.")), *cnt)
tab("tab_dist", s + "\\bottomrule\\end{tabular}")
# ---- model facts
s = "\\begin{tabular}{lccccccc}\\toprule Model & Params & Layers & Q/KV heads & Ctx & fp16 GB & KV B/tok & License\\\\\\midrule\n"
for c, v in facts.items():
    if "error" in v: continue
    s += "%s & %.2fB & %d & %d/%d & %dk & %.1f & %s & %s\\\\\n" % (esc(c.split("/")[1].replace("-Instruct", "").replace("-Chat-v1.0", "")), v["params_B"], v["layers"], v["heads"], v["kv_heads"], v["max_positions"] // 1024, v["fp16_weights_GB"], format(v["kv_bytes_per_token_fp16"], ","), esc(v["license_note"].split(" ")[0]))
tab("tab_models", s + "\\bottomrule\\end{tabular}")
# ---- equivalence + cache bench
s = "\\begin{tabular}{rrcc}\\toprule Input & Output & cached = no-cache & cached = HF generate\\\\\\midrule\n"
for r in dec["equivalence_cache"]: s += "%d & %d & %s & %s\\\\\n" % (r["in_len"], r["out_len"], "yes" if r["cached_eq_nocache"] else "NO (@%d)" % r["first_mismatch_cached_vs_nocache"], "yes" if r["cached_eq_hf"] else "NO (@%d)" % r["first_mismatch_cached_vs_hf"])
tab("tab_equiv", s + "\\bottomrule\\end{tabular}")
s = "\\begin{tabular}{rrrrrrrr}\\toprule In & Out & No-cache s & Cached s & Speedup & Prefill s & Decode tok/s & Peak MB nc/c\\\\\\midrule\n"
for b in dec["bench_cache"]: s += "%d & %d & %.2f & %.2f & %.1f$\\times$ & %.3f & %.1f & %.0f/%.0f\\\\\n" % (b["in_len"], b["out_len"], b["nocache"]["total"], b["cached"]["total"], b["speedup_total"], b["cached"]["prefill"], b["cached"]["decode_tok_per_s"], b["nocache"]["peak_extra_MB"], b["cached"]["peak_extra_MB"])
tab("tab_cache", s + "\\bottomrule\\end{tabular}")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1)); lab = ["%d/%d" % (b["in_len"], b["out_len"]) for b in dec["bench_cache"]]; x = np.arange(len(lab))
ax[0].bar(x - .2, [b["nocache"]["total"] for b in dec["bench_cache"]], .4, label="no cache"); ax[0].bar(x + .2, [b["cached"]["total"] for b in dec["bench_cache"]], .4, label="KV cache"); ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=30); ax[0].set_ylabel("total time (s)"); ax[0].set_xlabel("input/output tokens"); ax[0].legend()
ax[1].plot([b["in_len"] + b["out_len"] for b in dec["bench_cache"]], [b["cached"]["kv_bytes"] / 2**20 if "kv_bytes" in b["cached"] else 0 for b in dec["bench_cache"]], "o-"); ax[1].set_xlabel("context length (tokens)"); ax[1].set_ylabel("KV cache in use (MB)"); save("fig_cache")
# ---- attention
s = "\\begin{tabular}{lcccc}\\toprule dtype & ref & tiled & tiled(128) & SDPA\\\\\\midrule\n"
for dt in ["float32", "float16"]:
    sub = [r for r in an if r["dtype"] == dt]; s += "%s max err & %s\\\\\n" % (dt, " & ".join("%.1e" % max(r[n + "_max"] for r in sub) for n in ["ref", "tiled", "tiled128", "sdpa"]))
    s += "%s mean err & %s\\\\\n" % (dt, " & ".join("%.1e" % np.mean([r[n + "_mean"] for r in sub]) for n in ["ref", "tiled", "tiled128", "sdpa"]))
tab("tab_attn_num", s + "\\bottomrule\\end{tabular}")
s = "\\begin{tabular}{rrrrrrr}\\toprule T & ref ms & tiled ms & SDPA ms & ref MB & tiled MB & SDPA MB\\\\\\midrule\n"
for p in ap: s += "%d & %.2f & %.2f & %.2f & %.0f & %.0f & %.0f\\\\\n" % (p["T"], p["ref_ms"], p["tiled_ms"], p["sdpa_ms"], p["ref_peakMB"], p["tiled_peakMB"], p["sdpa_peakMB"])
tab("tab_attn_perf", s + "\\bottomrule\\end{tabular}")
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1)); T = [p["T"] for p in ap]
for n in ["ref", "tiled", "sdpa"]: ax[0].plot(T, [p[n + "_ms"] for p in ap], "o-", label=n); ax[1].plot(T, [p[n + "_peakMB"] for p in ap], "o-", label=n)
for a_, yl in zip(ax, ["time (ms)", "peak extra memory (MB)"]): a_.set_xscale("log", base=2); a_.set_yscale("log"); a_.set_xlabel("sequence length T"); a_.set_ylabel(yl); a_.legend()
save("fig_attn")
# ---- speculative
s = "\\begin{tabular}{rcccccc}\\toprule K & Token-identical & Accept. & Tok/round & Spec s & Cached s & Speedup\\\\\\midrule\n"
for r in dec["speculative"]: s += "%d & %d/%d & %.2f & %.2f & %.3f & %.3f & %.2f$\\times$\\\\\n" % (r["K"], r["token_identical"], r["n_prompts"], r["acceptance_rate"], r["tokens_per_round"], r["latency_spec_s"], r["latency_cached_s"], r["speedup"])
tab("tab_spec", s + "\\bottomrule\\end{tabular}")
fig, ax = plt.subplots(figsize=(3.4, 2.1)); K = [r["K"] for r in dec["speculative"]]
ax.bar(np.arange(len(K)) - .2, [r["speedup"] for r in dec["speculative"]], .4, label="speedup vs cached"); ax.bar(np.arange(len(K)) + .2, [r["acceptance_rate"] for r in dec["speculative"]], .4, label="draft acceptance"); ax.axhline(1, color="k", lw=.6); ax.set_xticks(range(len(K))); ax.set_xticklabels(["K=%d" % k for k in K]); ax.legend(); save("fig_spec")
# ---- error analysis (>= 8 distinct errors)
recs = res["tuned_test"]["recs"]; test = {e["annotation_id"]: e for e in json.load(open(W + "/test.json"))}
errs = [r for r in recs if r["gold"] != r["pred"]]; seen = set(); pick = []
def kind(r):
    if not r["valid"]: return "invalid"
    if r["pred"] in ("included", "excluded", "not included", "not excluded") and not r["pred_ev"]: return "unsupported"
    return "%s->%s" % (r["gold"], r["pred"])
for r in sorted(errs, key=lambda r: (kind(r) in seen, r["type"])):
    k = kind(r)
    if k not in seen: seen.add(k); pick.append(r)
pick = pick[:12]; ea = []
s = "\\begin{tabular}{p{0.4cm}p{6.5cm}p{2.3cm}p{2.3cm}p{1.4cm}p{1.4cm}}\\toprule \\# & Criterion (truncated) & Gold & Pred & Gold ev. & Pred ev.\\\\\\midrule\n"
for i, r in enumerate(pick):
    e = test[r["annotation_id"]]; ea.append({"annotation_id": r["annotation_id"], "type": r["type"], "criterion": e["criterion_text"], "gold": r["gold"], "pred": r["pred"], "gold_ev": r["gold_ev"], "pred_ev": r["pred_ev"], "note": e["note"], "raw": r["raw"], "kind": kind(r)})
    s += "%d & %s & %s & %s & %s & %s\\\\\n" % (i + 1, esc(e["criterion_text"][:70] + ("..." if len(e["criterion_text"]) > 70 else "")), esc(r["gold"].replace("not enough information", "NEI")), esc(r["pred"].replace("not enough information", "NEI")), esc(r["gold_ev"]), esc(r["pred_ev"]))
tab("tab_errors", s + "\\bottomrule\\end{tabular}"); json.dump(ea, open(W + "/error_analysis.json", "w"), indent=1)
print("errors total", len(errs), "distinct kinds picked", len(pick), [kind(r) for r in pick])
# ---- summary for the app + results folder
ms = {nm: {"accuracy": m["overall"]["accuracy"], "macro_f1": m["overall"]["macro_f1"], "evidence_micro_f1": m["overall"]["evidence"]["micro_F1"]} for nm, m in rows}
os.makedirs(W + "/results", exist_ok=True); json.dump(ms, open(W + "/results/metrics_summary.json", "w"), indent=1)
for f in ["decode_results.json", "attn_numeric.json", "attn_perf.json", "model_facts.json", "error_analysis.json", "split_manifest.json"]: shutil.copy("%s/%s" % (W, f), "%s/results/%s" % (W, f))
shutil.copy("%s/%s/train_log.json" % (W, sel), W + "/results/train_log.json"); shutil.copy("%s/%s/test_results.json" % (W, sel), W + "/results/test_results.json")
if vb: shutil.copy("%s/%s/val_base.json" % (W, sel), W + "/results/val_base.json")
shutil.copy("%s/%s/val_tuned.json" % (W, sel), W + "/results/val_tuned.json")
print("assets written", sorted(os.listdir(A)))

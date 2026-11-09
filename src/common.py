import json, re, torch, numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

MID = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
LABELS = {"inclusion": ["included", "not included", "not enough information", "not applicable"],
          "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"]}
SYS = ("You are a clinical trial eligibility screener. Judge ONE criterion against the patient note using only the note. "
       "Answer with JSON: {\"label\": <label>, \"evidence\": [<supporting sentence ids>]}. "
       "Inclusion labels: included, not included, not enough information, not applicable. "
       "Exclusion labels: excluded, not excluded, not enough information, not applicable.")

def build_prompt(tok, note, ctype, ctext):
    user = "Patient note (numbered sentences):\n%s\n\nCriterion type: %s\nCriterion: %s\n\nReturn the JSON." % (note, ctype, ctext)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def build_target(label, ev):
    return json.dumps({"label": label, "evidence": sorted(int(e) for e in ev)}) + "<|im_end|>"

def parse_output(txt, ctype):
    """returns (label or 'INVALID', evidence list, valid flag)"""
    m = re.search(r'"label"\s*:\s*"([^"]+)"', txt)
    lab = m.group(1).strip().lower() if m else None
    ev = []
    m2 = re.search(r'"evidence"\s*:\s*\[([^\]]*)\]', txt)
    if m2:
        ev = [int(x) for x in re.findall(r"\d+", m2.group(1))]
    if lab not in LABELS[ctype]:
        return "INVALID", ev, False
    return lab, sorted(set(ev)), True

@torch.no_grad()
def generate_batch(model, tok, prompts, max_new=48, bs=16):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for j, t in zip(idx, txt):
            res[j] = t
    return res

def predict(model, tok, examples, bs=16):
    prompts = [build_prompt(tok, e["note"], e["criterion_type"], e["criterion_text"]) for e in examples]
    raw = generate_batch(model, tok, prompts, bs=bs)
    recs = []
    for e, r in zip(examples, raw):
        lab, ev, ok = parse_output(r, e["criterion_type"])
        recs.append({"annotation_id": e["annotation_id"], "patient_id": e["patient_id"], "type": e["criterion_type"],
                     "gold": e["expert_eligibility"], "pred": lab, "gold_ev": sorted(e["expert_sentences"]), "pred_ev": ev,
                     "valid": ok, "raw": r})
    return recs

def evidence_scores(recs):
    tp = fp = fn = 0
    per = []
    for r in recs:
        g, p = set(r["gold_ev"]), set(r["pred_ev"])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        if not g and not p:
            per.append(1.0)
        elif not g or not p:
            per.append(0.0)
        else:
            pr, rc = len(g & p) / len(p), len(g & p) / len(g)
            per.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"micro_P": P, "micro_R": R, "micro_F1": F, "macro_F1": float(np.mean(per)) if per else 0.0}

def metrics(recs):
    out = {}
    for name in ["overall", "inclusion", "exclusion"]:
        sub = recs if name == "overall" else [r for r in recs if r["type"] == name]
        g = [r["gold"] for r in sub]; p = [r["pred"] for r in sub]
        labs = sorted(set(g))
        pr, rc, f1, sup = precision_recall_fscore_support(g, p, labels=labs, zero_division=0)
        cl = labs + ["INVALID"]
        out[name] = {"n": len(sub), "accuracy": float(np.mean([a == b for a, b in zip(g, p)])),
                     "macro_f1": float(f1.mean()),
                     "per_class": {l: {"P": float(a), "R": float(b), "F1": float(c), "support": int(d)} for l, a, b, c, d in zip(labs, pr, rc, f1, sup)},
                     "confusion_labels": cl, "confusion": confusion_matrix(g, p, labels=cl).tolist(),
                     "evidence": evidence_scores(sub), "invalid_outputs": int(sum(not r["valid"] for r in sub))}
    return out

def load_split(name):
    return json.load(open("/content/work/%s.json" % name))


import sys, time, json, random, math, collections, argparse, os, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/content/work")
from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MID); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--bs", type=int, default=2); ap.add_argument("--acc", type=int, default=8)
ap.add_argument("--balance_pow", type=float, default=0.5); ap.add_argument("--base_eval", type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda()
train, val = load_split("train"), load_split("val")
if a.base_eval:
    t0 = time.time(); br = predict(base, tok, val); bm = metrics(br)
    json.dump({"metrics": bm, "recs": br}, open(a.out + "/val_base.json", "w"))
    print("BASE val (acc, macroF1, evidF1, invalid):", {k: (round(v["accuracy"], 3), round(v["macro_f1"], 3), round(v["evidence"]["micro_F1"], 3), v["invalid_outputs"]) for k, v in bm.items()}, round(time.time() - t0), "s", flush=True)
MAXLEN = 1024
def enc(e):
    p = tok(build_prompt(tok, e["note"], e["criterion_type"], e["criterion_text"]), add_special_tokens=False).input_ids
    t = tok(build_target(e["expert_eligibility"], e["expert_sentences"]), add_special_tokens=False).input_ids
    tr = len(p) + len(t) > MAXLEN
    if tr: p = p[len(p) + len(t) - MAXLEN:]
    return p, t, tr
data = [enc(e) for e in train]
lens = [len(p) + len(t) for p, t, _ in data]
print("train seq tokens max/mean:", max(lens), int(np.mean(lens)), "truncated examples:", sum(x[2] for x in data), "target tokens mean:", np.mean([len(t) for _, t, _ in data]), flush=True)
base.gradient_checkpointing_enable(); base.enable_input_require_grads()
cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(base, cfg)
params = [p for p in model.parameters() if p.requires_grad]
print("trainable params:", sum(p.numel() for p in params), "dtype:", params[0].dtype, flush=True)
def collate(ix):
    L = max(len(data[i][0]) + len(data[i][1]) for i in ix)
    ids = torch.full((len(ix), L), tok.pad_token_id); lab = torch.full((len(ix), L), -100); att = torch.zeros((len(ix), L), dtype=torch.long)
    for r, i in enumerate(ix):
        p, t, _ = data[i]; n = len(p) + len(t)
        ids[r, :n] = torch.tensor(p + t); lab[r, len(p):n] = torch.tensor(t); att[r, :n] = 1
    return ids.cuda(), lab.cuda(), att.cuda()
BS, ACC, EP, LR = a.bs, a.acc, a.epochs, a.lr
cnt = collections.Counter(e["expert_eligibility"] for e in train)
w = torch.tensor([(1.0 / cnt[e["expert_eligibility"]]) ** a.balance_pow for e in train])
spe = math.ceil(len(train) / (BS * ACC)); total = spe * EP
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
sched = lambda s: min(1.0, (s + 1) / 5) * 0.5 * (1 + math.cos(math.pi * s / total))
scaler = torch.amp.GradScaler("cuda")
log, ep_log, best, step = [], [], -1, 0
for ep in range(EP):
    model.train(); t1 = time.time()
    idx = torch.multinomial(w, len(train), replacement=True).tolist()
    for b0 in range(0, len(idx), BS * ACC):
        chunk = idx[b0:b0 + BS * ACC]; tl = 0.0
        ntok = sum(len(data[i][1]) for i in chunk)
        for m0 in range(0, len(chunk), BS):
            ids, lab, att = collate(chunk[m0:m0 + BS])
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / ntok
            scaler.scale(loss).backward(); tl += loss.item()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups: g["lr"] = LR * sched(step)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
        log.append({"step": step, "epoch": ep + 1, "loss": tl, "lr": LR * sched(step - 1)})
        if step % 10 == 0: print("step", step, "/", total, "loss", round(tl, 4), round(time.time() - t1), "s", flush=True)
    model.eval(); recs = predict(model, tok, val); m = metrics(recs)
    ep_log.append({"epoch": ep + 1, "val_acc": m["overall"]["accuracy"], "val_macro_f1": m["overall"]["macro_f1"], "val_evid_f1": m["overall"]["evidence"]["micro_F1"], "invalid": m["overall"]["invalid_outputs"]})
    print("EPOCH", ep + 1, ep_log[-1], flush=True)
    json.dump({"train_log": log, "epoch_log": ep_log, "args": vars(a)}, open(a.out + "/train_log.json", "w"))
    if m["overall"]["macro_f1"] > best:
        best = m["overall"]["macro_f1"]; model.save_pretrained(a.out + "/adapter_best")
        json.dump({"metrics": m, "recs": recs}, open(a.out + "/val_tuned.json", "w")); print("  saved best epoch", ep + 1, flush=True)
print("DONE best val macro-F1", best, flush=True)

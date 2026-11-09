
import sys, json, collections, torch
sys.path.insert(0, "/content/work")
from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
tag, adapter = sys.argv[1], sys.argv[2]
model_id = sys.argv[3] if len(sys.argv) > 3 else MID
tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).cuda()
train, test = load_split("train"), load_split("test")
res = {}
# trivial baselines (no model): majority label per criterion type from TRAIN; evidence = sentence 0
maj = {t: collections.Counter(e["expert_eligibility"] for e in train if e["criterion_type"] == t).most_common(1)[0][0] for t in ["inclusion", "exclusion"]}
brec = [{"type": e["criterion_type"], "gold": e["expert_eligibility"], "pred": maj[e["criterion_type"]], "gold_ev": sorted(e["expert_sentences"]), "pred_ev": [0], "valid": True} for e in test]
res["majority_baseline"] = {"metrics": metrics(brec), "majority": maj}
br = predict(base, tok, test); res["base_test"] = {"metrics": metrics(br), "recs": br}
m = PeftModel.from_pretrained(base, adapter); m.eval()
tr = predict(m, tok, test); res["tuned_test"] = {"metrics": metrics(tr), "recs": tr}
json.dump(res, open("/content/work/%s/test_results.json" % tag, "w"))
for k in ["majority_baseline", "base_test", "tuned_test"]:
    mm = res[k]["metrics"]
    print(k, {n: (round(v["accuracy"], 3), round(v["macro_f1"], 3), round(v["evidence"]["micro_F1"], 3), v["invalid_outputs"]) for n, v in mm.items()})

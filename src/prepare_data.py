"""Data preparation: prescribed patient-grouped split of ncbi/TrialGPT-Criterion-Annotations.

Sorts the unique patient IDs, shuffles them with numpy default_rng(42), assigns the first 37 patients to
train, the next 8 to validation and the last 8 to test, keeps every annotation of a patient in exactly one
split, drops all GPT-4 produced fields, normalises the expert sentence IDs to integer lists, asserts that
the splits are disjoint and writes the split manifest (patient IDs and counts).

    python prepare_data.py            # writes /content/work/{train,val,test}.json and split_manifest.json
"""
import collections
import json
import os
import re

import numpy as np
from datasets import load_dataset

WORK = os.environ.get("TM_WORK", "/content/work")
KEEP = ["annotation_id", "patient_id", "note", "trial_id", "trial_title", "criterion_type", "criterion_text",
        "expert_sentences", "expert_eligibility"]            # gpt4_* fields and the dataset's own 'training' flag are NOT kept


def main():
    os.makedirs(WORK, exist_ok=True)
    rows = [dict(r) for r in load_dataset("ncbi/TrialGPT-Criterion-Annotations")["train"]]
    pids = sorted({r["patient_id"] for r in rows})
    perm = list(np.random.default_rng(42).permutation(pids))
    tr, va, te = perm[:37], perm[37:45], perm[45:]
    assert (len(pids), len(tr), len(va), len(te)) == (53, 37, 8, 8)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "patient leakage"
    split_of = {p: "train" for p in tr}
    split_of.update({p: "val" for p in va})
    split_of.update({p: "test" for p in te})
    splits = {}
    for s in ["train", "val", "test"]:
        recs = []
        for r in rows:
            if split_of[r["patient_id"]] != s:
                continue
            e = {k: r[k] for k in KEEP}
            e["expert_sentences"] = [int(x) for x in re.findall(r"\d+", str(e["expert_sentences"]))]
            recs.append(e)
        splits[s] = recs
        json.dump(recs, open(f"{WORK}/{s}.json", "w"))
    man = {"seed": 42, "patients": {"train": sorted(tr), "val": sorted(va), "test": sorted(te)},
           "counts": {s: len(v) for s, v in splits.items()}}
    json.dump(man, open(f"{WORK}/split_manifest.json", "w"), indent=1)
    # every patient appears in exactly one split
    seen = collections.Counter(e["patient_id"] for s in splits.values() for e in s)
    assert all(split_of[p] == s for s, v in splits.items() for p in {e["patient_id"] for e in v})
    print("annotations per split:", man["counts"], "| patients:", len(seen))


if __name__ == "__main__":
    main()

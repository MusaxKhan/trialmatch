# 5–7 minute demo video script

Record the browser app (screen + voice). Suggested timing:

| Time | What to show | What to say |
|---|---|---|
| 0:00–0:30 | Title slide / repo README | Task: criterion-level trial screening with evidence; open-weight Qwen2.5-1.5B + LoRA; own KV-cache, tiled attention, speculative decoding; educational prototype, not for clinical decisions. |
| 0:30–1:15 | Repo tree: `src/`, `app/`, `results/`, `report/` | Fixed split 37/8/8 patients (`default_rng(42)`), manifest in `results/split_manifest.json`, GPT-4 fields dropped, adapters hosted separately. |
| 1:15–3:00 | App → **Screening** tab. *Load synthetic sample*, *Run screening* | Header shows model + device. Each criterion: label, cited sentence IDs resolved to text, explanation built only from cited sentences, flags for unsupported/invalid output. Overall outcome from rules R1–R4 (About tab) — an aggregation, not ground truth. Show *not enough information* case: absence of evidence is not eligibility. |
| 3:00–3:40 | Robustness | Clear the note → red validation error. Add a criterion with `exclusion` and 2 characters of text → error. Show app keeps working. Paste your own **unseen** note + criteria and run. |
| 3:40–4:10 | **Download report (HTML)**; open it | Provenance (model, adapter hash, mode, timestamp) and disclaimer are in the export. |
| 4:10–5:30 | **Inference lab** → *Run comparison* | No-cache vs KV-cache vs speculative on the same criteria: time, tokens/s, GPU memory, identical-token check. Explain: cache removes recomputation (prefill once, 1 token/step, 28 KiB/token); speculative = 0.5B draft proposes K tokens, 1.5B verifies, rollback via cache `crop`; identical output but not always faster (draft cost ratio ≈ 1.0 on this GPU: 43 ms per token for both models). |
| 5:30–6:15 | **Attention lab** (T=2048, tile 64) | Reference vs tiled online-softmax vs PyTorch SDPA: error vs fp64 ≈ 1e-5 in fp32, tiled uses ~45× less memory at T=4096 but is ~5× slower than the dense reference in pure PyTorch — fused SDPA is fastest. |
| 6:15–6:50 | **Experiments** tab | Held-out test table (before/after tuning, baselines), cache benchmark, speculative table. |
| 6:50–7:00 | Close | Limitations: 8 test patients, rare classes fragile, one GPU, one run per config. |

Tips: have the app already loaded (first start downloads ~4 GB), use *Load synthetic sample* so no real patient text
is shown, and keep a prepared second note/criteria pair for the "unseen case" part of the live evaluation.

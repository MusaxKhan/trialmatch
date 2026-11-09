# TrialMatch — criterion-level clinical-trial screening (GenAI Assignment 2)

> **Educational screening prototype — not a clinical enrolment decision.**

Given a patient note and one trial criterion (inclusion or exclusion) the system returns an
expert-style eligibility label (`included / not included / excluded / not excluded / not enough
information / not applicable`) plus the IDs of the supporting patient sentences.

What is implemented here (everything except the pretrained weights/tokenizers is our own code):

| Task | Where | Notes |
|---|---|---|
| Data, fixed split (37/8/8 patients, `default_rng(42)`) | `src/prepare_data.py`, `results/split_manifest.json` | GPT-4 fields dropped, leakage assert |
| LoRA fine-tuning (Qwen2.5-1.5B target, 0.5B draft) | `src/train.py` | completion-only loss, class-balanced sampling |
| Metrics (acc, macro-F1, per-class, confusion, evidence P/R/F1) | `src/common.py`, `src/eval_test.py` | |
| Hand-written Qwen2 forward pass + **KV cache** | `src/minillm.py` (`MiniQwen`, `KVCache`) | explicit positions, causal offset mask, cache growth |
| **Reference + tiled online-softmax attention** | `src/minillm.py` | `ref_attention`, `tiled_attention` |
| **Greedy speculative decoding** with cache rollback | `src/minillm.py` (`speculative`) | |
| Correctness tests and benchmarks | `src/decode_tests.py`, `src/attn_tests.py` | outputs in `results/` |
| Browser application | `app/app.py`, `app/static/index.html` | FastAPI, no notebook needed |
| IEEE paper (LaTeX) | `report/main.tex` | |

## Quick start (local machine, browser app)

1. Python 3.10+; an NVIDIA GPU is recommended (the code has a CPU fp32 fallback, but it was not tested on a CPU-only machine; the demo and all benchmarks ran on a Colab T4).
2. `pip install -r requirements.txt`
3. Download the adapters (see **Model weights** below) and unpack so that you have
   `adapters/target/adapter_model.safetensors` and `adapters/draft/adapter_model.safetensors`.
4. Start the server:

   ```bash
   cd app
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```
   (Windows: double-click `run_app.bat`.) Open <http://127.0.0.1:8000>. The first start downloads the
   base Qwen2.5 weights (~4 GB) from Hugging Face; the header shows `connecting…`/`loading` until ready.
5. In the page: *Load synthetic sample* → *Run screening*. Tabs: **Screening**, **Inference lab**
   (live no-cache vs KV-cache vs speculative comparison), **Attention lab**, **Experiments**
   (saved benchmark tables), **About & rules**.

Environment variables: `ADAPTER_T`, `ADAPTER_D` (adapter folders), `RESULTS_DIR`,
`TRIALMATCH_BASE=1` (serve the *base* models without LoRA, for before/after demos).

Invalid input (empty note, bad criteria, too-long text, unknown mode) returns a readable error in the
page instead of crashing; the model is loaded once at start-up and `/api/status` reports progress.

## Model weights

Fine-tuned LoRA adapters (≈74 MB target, ≈35 MB draft), hosted as a GitHub release asset:
**https://github.com/MusaxKhan/trialmatch/releases/download/v1.0/adapters.zip**

```bash
python scripts/download_adapters.py https://github.com/MusaxKhan/trialmatch/releases/download/v1.0/adapters.zip
```
(unpacks into `adapters/target` and `adapters/draft`). Base weights
(`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`, Apache-2.0) are pulled from Hugging Face.

## Reproducing the experiments (Google Colab T4)

The scripts use `/content/work` as their working directory (the Colab convention). On Colab:

```bash
mkdir -p /content/work && cp src/*.py /content/work/ && cd /content/work
pip -q install -U peft datasets accelerate && pip -q uninstall -y torchao
python prepare_data.py                              # split + manifest + leakage check
python train.py --out /content/work/q15 --base_eval 1                         # target, v1
python train.py --out /content/work/q15b --base_eval 0 --epochs 3 --balance_pow 1.0   # target, v2
python train.py --model Qwen/Qwen2.5-0.5B-Instruct --out /content/work/q05 --bs 4 --acc 4   # draft
python eval_test.py <q15|q15b> /content/work/<dir>/adapter_best               # held-out test (once)
python attn_tests.py                                # tiled/ref/SDPA numerics + speed
python decode_tests.py /content/work/<dir>/adapter_best /content/work/q05/adapter_best
python make_assets.py <chosen> <alt>                # figures + LaTeX tables for the paper
```

Seeds are fixed (split seed 42, training seed 42). GPU kernels are not bit-deterministic, so
re-runs can differ slightly in the last decimals.

## Results, logs, split manifest

* `results/split_manifest.json` — patient IDs per split and annotation counts.
* `results/train_log*.json`, `results/val_*.json`, `results/test_results.json` — training curves,
  validation per epoch, held-out predictions and metrics (before/after tuning, baselines).
* `results/decode_results.json`, `attn_numeric.json`, `attn_perf.json` — equivalence tests and benchmarks.
* `results/error_analysis.json` — representative errors used in the paper.

## Headline results (held-out test, 206 criteria, 8 patients)

| System | Acc | Macro-F1 | Evidence µ-F1 |
|---|---|---|---|
| Majority label per type + sentence 0 | 0.626 | 0.272 | 0.253 |
| Qwen2.5-1.5B zero-shot | 0.121 | 0.172 | 0.167 |
| + LoRA (HF `generate`, default repetition penalty 1.1) | 0.723 | 0.685 | 0.559 |
| + LoRA (own engine, pure greedy) | 0.728 | 0.708 | 0.531 |

KV cache: token-identical to the no-cache reference (9/9 length combinations) and to Hugging Face greedy when
its default repetition penalty is disabled; 1.3×–9.0× faster. Tiled attention: fp32 error ≈ 1.6e-5 vs fp64, 45×
less memory than the dense reference at T=4096, but slower in pure PyTorch. Speculative decoding: token-identical
(38/40 fp16, 20/20 fp32), acceptance 86–94 %, but 0.81–0.90× the speed of cached decoding because the draft and
target cost the same per token (≈43 ms) in this eager PyTorch engine. Details and caveats: `report/main.pdf`.

## Honest notes

* The split is tiny (8 test patients, 206 criteria); rare classes have very few examples.
* Speculative decoding is verified token-identical to greedy but is not guaranteed to be faster; see the paper.
* The overall patient–trial outcome in the app is a rule-based aggregation of criterion labels, not ground truth.
* AI assistance and reuse of ideas from three public repositories are disclosed in the paper's appendix.

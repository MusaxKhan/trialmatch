
import sys, json, math, torch
sys.path.insert(0, "/content/work")
from minillm import *
torch.manual_seed(0); dev = "cuda"
def gt_attn(q, k, v, causal):
    q, k, v = q.double(), k.double(), v.double()
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal and q.shape[2] > 1: s = s.masked_fill(~causal_mask(q.shape[2], k.shape[2], q.device), float("-inf"))
    return torch.softmax(s, -1) @ v
shapes = [(1, 4, 16, 16), (2, 6, 33, 33), (1, 8, 100, 100), (1, 12, 1, 257), (1, 12, 5, 261), (2, 4, 128, 256), (1, 4, 257, 257), (1, 14, 512, 512)]
rows = []
for dtype in [torch.float32, torch.float16]:
    for (B, H, Tq, Tk) in shapes:
        for causal in [True, False]:
            for scale in [1.0, 8.0]:
                q = (torch.randn(B, H, Tq, 64, device=dev) * scale).to(dtype); k = torch.randn(B, H, Tk, 64, device=dev).to(dtype); v = torch.randn(B, H, Tk, 64, device=dev).to(dtype)
                gt = gt_attn(q, k, v, causal)
                r = {"dtype": str(dtype).split(".")[1], "B": B, "H": H, "Tq": Tq, "Tk": Tk, "causal": causal, "q_scale": scale}
                for name, fn in [("ref", lambda: ref_attention(q, k, v, causal)), ("tiled", lambda: tiled_attention(q, k, v, causal, 32, 32)), ("tiled128", lambda: tiled_attention(q, k, v, causal)), ("sdpa", lambda: sdpa_attention(q, k, v, causal))]:
                    o = fn().double(); d = (o - gt).abs()
                    r[name + "_max"] = d.max().item(); r[name + "_mean"] = d.mean().item()
                o = tiled_attention(q, k, v, causal, 32, 32).double(); r["tiled_vs_ref_max"] = (o - ref_attention(q, k, v, causal).double()).abs().max().item()
                rows.append(r)
json.dump(rows, open("/content/work/attn_numeric.json", "w"))
for dt in ["float32", "float16"]:
    sub = [r for r in rows if r["dtype"] == dt]
    print(dt, "cases", len(sub), "max err vs fp64 ->", {n: "%.2e" % max(r[n + "_max"] for r in sub) for n in ["ref", "tiled", "tiled128", "sdpa"]}, " tiled-vs-ref max %.2e" % max(r["tiled_vs_ref_max"] for r in sub))
# speed / memory (fp16, B=1, H=12, D=128, causal prefill)
perf = []
for T in [256, 512, 1024, 2048, 4096]:
    q, k, v = [torch.randn(1, 12, T, 128, device=dev, dtype=torch.float16) for _ in range(3)]
    r = {"T": T}
    for name, fn in [("ref", ref_attention), ("tiled", tiled_attention), ("sdpa", sdpa_attention)]:
        st, res = bench(lambda: fn(q, k, v, True), warmup=2, reps=5)
        r[name + "_ms"] = st["median_s"] * 1000; r[name + "_peakMB"] = st["peak_extra_MB"]
    perf.append(r); print(r)
json.dump(perf, open("/content/work/attn_perf.json", "w"))

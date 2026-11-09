import math, time, torch
import torch.nn.functional as F

# ---------------- attention kernels (q: B,H,Tq,D ; k,v: B,H,Tk,D) ----------------
def causal_mask(Tq, Tk, device):
    # query i (0-based within the Tq new tokens) sits at absolute position Tk-Tq+i, so it may see keys j <= Tk-Tq+i
    return torch.ones(Tq, Tk, dtype=torch.bool, device=device).tril(Tk - Tq)

def ref_attention(q, k, v, causal=True):
    """Reference scaled dot-product attention. Full score matrix, explicit max-shifted softmax in fp32."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal and q.shape[2] > 1:
        s = s.masked_fill(~causal_mask(q.shape[2], k.shape[2], q.device), float("-inf"))
    s = s - s.amax(-1, keepdim=True)
    p = torch.exp(s)
    p = p / p.sum(-1, keepdim=True)
    return torch.matmul(p, v.float()).to(q.dtype)

def tiled_attention(q, k, v, causal=True, block_q=128, block_k=128):
    """Tiled attention with online softmax (FlashAttention-style): per query block keep running max m,
    running normaliser l and running un-normalised output acc; K/V are streamed in blocks and the
    Tq x Tk score matrix is never materialised."""
    B, H, Tq, D = q.shape; Tk = k.shape[2]
    scale = 1.0 / math.sqrt(D); off = Tk - Tq
    out = torch.empty(B, H, Tq, D, dtype=torch.float32, device=q.device)
    for qs in range(0, Tq, block_q):
        qe = min(qs + block_q, Tq)
        qb = q[:, :, qs:qe].float()
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, D), device=q.device)
        for ks in range(0, Tk, block_k):
            ke = min(ks + block_k, Tk)
            if causal and ks > (qe - 1) + off:
                break                                   # whole key block lies in the future of every query in this block
            s = torch.matmul(qb, k[:, :, ks:ke].float().transpose(-1, -2)) * scale
            if causal:
                qi = torch.arange(qs, qe, device=q.device)[:, None] + off
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)
            alpha = torch.exp(m - m_new)                # rescale previous partial sums to the new maximum
            l = l * alpha + p.sum(-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, v[:, :, ks:ke].float())
            m = m_new
        out[:, :, qs:qe] = acc / l
    return out.to(q.dtype)

def sdpa_attention(q, k, v, causal=True):
    """PyTorch fused / optimised backend (flash / mem-efficient / math)."""
    if q.shape[2] == 1 or not causal:
        return F.scaled_dot_product_attention(q, k, v)
    if q.shape[2] == k.shape[2]:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(q.shape[2], k.shape[2], q.device))

ATTN = {"ref": ref_attention, "tiled": tiled_attention, "sdpa": sdpa_attention}

# ---------------- rotary embeddings, GQA ----------------
def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)

def repeat_kv(x, n):
    if n == 1: return x
    B, Hk, S, D = x.shape
    return x[:, :, None].expand(B, Hk, n, S, D).reshape(B, Hk * n, S, D)

# ---------------- KV cache ----------------
class KVCache:
    """Pre-allocated per-layer K/V buffers [B,Hkv,Tmax,D]. 'len' is the number of valid positions.
    update() writes new keys/values at positions [len, len+T) and returns views of the valid prefix.
    crop(n) rolls back by moving the pointer (used by speculative decoding after a rejection)."""
    def __init__(self, n_layers, B, Hkv, Tmax, D, dtype, device):
        self.k = [torch.empty(B, Hkv, Tmax, D, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.empty(B, Hkv, Tmax, D, dtype=dtype, device=device) for _ in range(n_layers)]
        self.len = 0; self.Tmax = Tmax
    def update(self, i, k, v):
        T = k.shape[2]
        self.k[i][:, :, self.len:self.len + T] = k
        self.v[i][:, :, self.len:self.len + T] = v
        return self.k[i][:, :, :self.len + T], self.v[i][:, :, :self.len + T]
    def crop(self, n): self.len = n
    def bytes_used(self):
        return 2 * sum(t[:, :, :self.len].numel() * t.element_size() for t in self.k)

# ---------------- hand written Qwen2 decoder using the HF weights ----------------
class MiniQwen:
    def __init__(self, hf, attn="sdpa"):
        c = hf.config
        self.hf, self.attn = hf, attn
        self.L, self.H, self.Hkv = c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
        self.D = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        theta = getattr(c, "rope_theta", None)
        if theta is None: theta = c.rope_parameters["rope_theta"]
        self.layers, self.emb, self.norm, self.lm_head = hf.model.layers, hf.model.embed_tokens, hf.model.norm, hf.lm_head
        self.dtype, self.device = self.emb.weight.dtype, self.emb.weight.device
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.D, 2, device=self.device).float() / self.D))
    def new_cache(self, Tmax, B=1):
        return KVCache(self.L, B, self.Hkv, Tmax, self.D, self.dtype, self.device)
    def kv_bytes_per_token(self):
        return 2 * self.L * self.Hkv * self.D * torch.finfo(self.dtype).bits // 8

    @torch.no_grad()
    def forward(self, ids, cache, last_only=False):
        """ids: [B,T] new tokens only. Absolute positions start at cache.len (positional handling),
        new K/V are appended to the cache (cache growth), attention is causal w.r.t. absolute positions."""
        B, T = ids.shape; pos0 = cache.len
        pos = torch.arange(pos0, pos0 + T, device=self.device)
        fr = pos[:, None].float() * self.inv_freq[None]
        emb = torch.cat([fr, fr], -1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        att = ATTN[self.attn]
        x = self.emb(ids)
        for i, ly in enumerate(self.layers):
            sa = ly.self_attn
            h = ly.input_layernorm(x)
            q = sa.q_proj(h).view(B, T, self.H, self.D).transpose(1, 2)
            k = sa.k_proj(h).view(B, T, self.Hkv, self.D).transpose(1, 2)
            v = sa.v_proj(h).view(B, T, self.Hkv, self.D).transpose(1, 2)
            q = (q.float() * cos + rotate_half(q.float()) * sin).to(self.dtype)
            k = (k.float() * cos + rotate_half(k.float()) * sin).to(self.dtype)
            K, V = cache.update(i, k, v)
            o = att(q, repeat_kv(K, self.H // self.Hkv), repeat_kv(V, self.H // self.Hkv), True)
            x = x + sa.o_proj(o.transpose(1, 2).reshape(B, T, self.H * self.D))
            h = ly.post_attention_layernorm(x)
            x = x + ly.mlp.down_proj(F.silu(ly.mlp.gate_proj(h)) * ly.mlp.up_proj(h))
        cache.len = pos0 + T
        x = self.norm(x)
        if last_only: x = x[:, -1:]
        return self.lm_head(x)

# ---------------- decoding loops ----------------
def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def _stop(tok, eos): return eos is not None and tok in eos

@torch.no_grad()
def greedy_nocache(m, ids, max_new, eos=None, stats=None):
    """Reference path: NO cache reuse. Every step re-runs the full forward on prompt+generated tokens."""
    seq = ids; t0 = time.perf_counter(); out = []
    for i in range(max_new):
        c = m.new_cache(seq.shape[1])
        nxt = m.forward(seq, c, last_only=True)[:, -1].argmax(-1, keepdim=True)
        if i == 0:
            _sync(); t_first = time.perf_counter() - t0
        out.append(nxt.item()); seq = torch.cat([seq, nxt], 1)
        if _stop(out[-1], eos): break
    _sync()
    if stats is not None: stats.update(prefill=t_first, total=time.perf_counter() - t0, n=len(out))
    return out

@torch.no_grad()
def greedy_cached(m, ids, max_new, eos=None, stats=None):
    """Prefill once (fills the cache), then feed ONE new token per step; K/V of earlier tokens are reused."""
    c = m.new_cache(ids.shape[1] + max_new); t0 = time.perf_counter()
    logits = m.forward(ids, c, last_only=True)[:, -1]
    _sync(); t_pre = time.perf_counter() - t0; out = []
    for i in range(max_new):
        nxt = logits.argmax(-1, keepdim=True); out.append(nxt.item())
        if _stop(out[-1], eos) or i == max_new - 1: break
        logits = m.forward(nxt, c, last_only=True)[:, -1]
    _sync()
    if stats is not None: stats.update(prefill=t_pre, total=time.perf_counter() - t0, n=len(out), kv_bytes=c.bytes_used())
    return out

@torch.no_grad()
def speculative(target, draft, ids, max_new, K=4, eos=None, stats=None):
    """Greedy speculative decoding. Draft proposes k tokens, target verifies [pending]+k drafts in ONE forward,
    longest matching prefix is accepted, the target's own token at the first mismatch (or the bonus token) is emitted.
    Output is identical to greedy decoding with the target. Caches are rolled back with crop()."""
    dev = target.device; P = ids.shape[1]; Tmax = P + max_new + K + 2
    tc, dc = target.new_cache(Tmax), draft.new_cache(Tmax); t0 = time.perf_counter()
    first = target.forward(ids, tc, last_only=True)[:, -1].argmax(-1).item()
    draft.forward(ids, dc, last_only=True)
    out = [first]; dpend = [first]; proposed = accepted = rounds = 0
    while len(out) < max_new and not _stop(out[-1], eos):
        k = min(K, max_new - len(out)); D0 = dc.len; m = len(dpend)
        x = torch.tensor([dpend], device=dev); drafts = []
        for _ in range(k):
            d = draft.forward(x, dc, last_only=True)[:, -1].argmax(-1).item(); drafts.append(d)
            x = torch.tensor([[d]], device=dev)
        base = tc.len
        logits = target.forward(torch.tensor([[out[-1]] + drafts], device=dev), tc)
        preds = logits[0].argmax(-1).tolist()
        n = 0
        while n < k and drafts[n] == preds[n]: n += 1
        new = drafts[:n] + [preds[n]]
        out += new
        proposed += k; accepted += n; rounds += 1
        if eos is not None and any(x in eos for x in new): break   # stop token inside the accepted block: finish now
        tc.crop(base + n + 1)                          # keep pending + n accepted drafts, drop rejected K/V
        if n < k:
            dc.crop(D0 + m + n); dpend = [preds[n]]
        else:
            dpend = [drafts[k - 1], preds[k]]          # draft never saw its last proposal: feed it next round
    out = out[:max_new]
    for j, t in enumerate(out):
        if _stop(t, eos): out = out[:j + 1]; break
    _sync()
    if stats is not None:
        stats.update(total=time.perf_counter() - t0, n=len(out), proposed=proposed, accepted=accepted, rounds=rounds,
                     acceptance=accepted / max(proposed, 1), tokens_per_round=len(out) / max(rounds, 1))
    return out

# ---------------- benchmarking helper ----------------
def bench(fn, warmup=1, reps=3):
    for _ in range(warmup): fn()
    _sync(); torch.cuda.empty_cache(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    ts, res = [], None
    for _ in range(reps):
        _sync(); t = time.perf_counter(); res = fn(); _sync(); ts.append(time.perf_counter() - t)
    ts.sort()
    return {"median_s": ts[len(ts) // 2], "min_s": ts[0], "peak_extra_MB": (torch.cuda.max_memory_allocated() - base) / 2**20}, res

"""A fused forward/backward for peft LoRA Linear layers (bf16 base, fp32 LoRA weights, no dropout).

Same math and precision as peft: the LoRA branch runs in fp32 (x @ A^T @ B^T * scaling, accumulated in fp32,
added to the bf16 base output in fp32 and rounded to bf16 once). peft does this with separate full-size passes
(x cast to fp32 and saved, an fp32 copy of the output, scale, add, cast); here the cast is done on the fly and
the rank-r products are fused into one pass over the base output, and only the bf16 input is saved.
"""
import types

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: the torch path below is used
    triton = None

if triton is not None:
    @triton.jit
    def _down_kernel(M, P, OUT, N, D, stride_m, R: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # OUT[N, R] = M[N, D] @ P[R, D]^T, fp32 accumulation
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        rr = tl.arange(0, R)
        rmask = rows < N
        acc = tl.zeros((BN, R), dtype=tl.float32)
        for d0 in range(0, D, BD):
            cols = d0 + tl.arange(0, BD)
            cmask = cols < D
            m = tl.load(M + rows[:, None] * stride_m + cols[None, :], mask=rmask[:, None] & cmask[None, :],
                        other=0.0).to(tl.float32)
            p = tl.load(P + rr[:, None] * D + cols[None, :], mask=cmask[None, :], other=0.0)
            acc += tl.sum(m[:, None, :] * p[None, :, :], axis=2)
        tl.store(OUT + rows[:, None] * R + rr[None, :], acc, mask=rmask[:, None])

    @triton.jit
    def _up_kernel(BASE, V, Q, OUT, N, D, s, R: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # OUT[N, D] = round(float(BASE) + (V[N, R] @ Q[D, R]^T) * s)
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        cols = tl.program_id(1) * BD + tl.arange(0, BD)
        rmask, cmask = rows < N, cols < D
        lora = tl.zeros((BN, BD), dtype=tl.float32)
        for k in tl.static_range(R):
            vk = tl.load(V + rows * R + k, mask=rmask, other=0.0)
            qk = tl.load(Q + cols * R + k, mask=cmask, other=0.0)
            lora += vk[:, None] * qk[None, :]
        mask = rmask[:, None] & cmask[None, :]
        b = tl.load(BASE + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + rows[:, None] * D + cols[None, :], (b + lora * s).to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _outer_kernel(M, V, OUT, N, D, stride_m, n_per, R: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # OUT[split, D, R] = M[rows of split, D]^T @ V[rows of split, R], fp32 accumulation
        cols = tl.program_id(0) * BD + tl.arange(0, BD)
        split = tl.program_id(1)
        rr = tl.arange(0, R)
        cmask = cols < D
        acc = tl.zeros((BD, R), dtype=tl.float32)
        start = split * n_per
        for n0 in range(start, start + n_per, BN):
            rows = n0 + tl.arange(0, BN)
            rmask = (rows < N) & (rows < start + n_per)
            m = tl.load(M + rows[:, None] * stride_m + cols[None, :], mask=rmask[:, None] & cmask[None, :],
                        other=0.0).to(tl.float32)
            v = tl.load(V + rows[:, None] * R + rr[None, :], mask=rmask[:, None], other=0.0)
            acc += tl.sum(m[:, :, None] * v[:, None, :], axis=0)
        tl.store(OUT + split * D * R + cols[:, None] * R + rr[None, :], acc, mask=cmask[:, None])


def _gpu(t):
    return triton is not None and t.is_cuda


def _down(m, p):
    """m [N, D] (any float dtype) @ p [R, D]^T in fp32 -> [N, R] fp32."""
    if not _gpu(m):
        return m.float() @ p.t()
    n, d = m.shape
    r = p.shape[0]
    out = torch.empty(n, r, device=m.device, dtype=torch.float32)
    _down_kernel[(triton.cdiv(n, 16),)](m, p.contiguous(), out, n, d, m.stride(0), R=r, BN=16, BD=64)
    return out


def _up(base, v, q, s, out=None):
    """round(float(base) + (v [N, R] @ q [D, R]^T) * s) in base's dtype."""
    if not _gpu(base):
        return (base.float() + (v @ q.t()) * s).to(base.dtype)
    n, d = base.shape
    out = torch.empty_like(base) if out is None else out
    _up_kernel[(triton.cdiv(n, 32), triton.cdiv(d, 128))](base, v, q.contiguous(), out, n, d, s, R=v.shape[1],
                                                          BN=32, BD=128)
    return out


def _outer(m, v):
    """m [N, D]^T @ v [N, R] in fp32 -> [D, R] fp32."""
    if not _gpu(m):
        return m.float().t() @ v
    n, d = m.shape
    r = v.shape[1]
    splits = max(1, min(16, n // 256))
    n_per = triton.cdiv(n, splits)
    out = torch.empty(splits, d, r, device=m.device, dtype=torch.float32)
    _outer_kernel[(triton.cdiv(d, 32), splits)](m, v, out, n, d, m.stride(0), n_per, R=r, BN=32, BD=32)
    return out.sum(0)


class _LoraLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, a, b, scaling):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        base = F.linear(x2, weight)
        xa = _down(x2, a)
        y = _up(base, xa, b, scaling, out=base)
        ctx.save_for_backward(x2, xa, weight, a, b)
        ctx.scaling, ctx.shape = scaling, shape
        return y.view(*shape[:-1], -1)

    @staticmethod
    def backward(ctx, dy):
        x2, xa, weight, a, b = ctx.saved_tensors
        s = ctx.scaling
        dy2 = dy.reshape(-1, dy.shape[-1])
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dxa = _down(dy2, b.t()) * s  # [N, r]
        db = _outer(dy2, xa) * s  # [out, r]
        da = _outer(x2, dxa).t()  # [r, in]
        dx = None
        if ctx.needs_input_grad[0]:
            dx = _up(dy2 @ weight, dxa, a.t(), 1.0).view(ctx.shape)
        return dx, None, da, db, None


def _forward(self, x, *args, **kwargs):
    """peft lora.Linear.forward, fused when only one plain LoRA adapter (no dropout, no bias) is active."""
    active = [n for n in self.active_adapters if n in self.lora_A]
    if (args or kwargs or self.disable_adapters or self.merged or len(active) != 1 or self.base_layer.bias is not None
            or active[0] in self.lora_variant or not isinstance(self.lora_dropout[active[0]], torch.nn.Identity)
            or self.lora_A[active[0]].bias is not None or self.lora_B[active[0]].bias is not None):
        return type(self).forward(self, x, *args, **kwargs)
    n = active[0]
    return _LoraLinear.apply(x, self.base_layer.weight, self.lora_A[n].weight, self.lora_B[n].weight,
                             self.scaling[n])


def install(model):
    """Route every peft lora.Linear in model through the fused path."""
    from peft.tuners.lora.layer import Linear

    for module in model.modules():
        if isinstance(module, Linear):
            module.forward = types.MethodType(_forward, module)

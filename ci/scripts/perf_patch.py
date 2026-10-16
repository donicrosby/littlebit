"""Perf autopsy of LittleBitLinear.__quant_convert__ — one layer, every stage timed.

Questions:
  1. Where do the ~23s/layer actually go? (svd_lowrank? ITQ? rank_one? residual round?)
  2. Does ITQ converge early (drift < 1e-4) or run all 50 iterations?
  3. What does dropping ITQ iterations / disabling ITQ cost in init fidelity?
"""
import os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
import quantization.utils.quant_util  # noqa: F401 (import-order landmine)
from quantization.modules.littlebit import LittleBitLinear
from quantization.functions.binary import STEBinary
import torch, torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

def make(d_out, d_in, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(d_out, 48, generator=g) @ torch.randn(48, d_in, generator=g)) * 0.05 \
        + torch.randn(d_out, d_in, generator=g) * 0.002
    lin = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(W)
    lin = lin.half().to(dev)
    lin.__class__ = LittleBitLinear
    return lin

# ---------- instrument ITQ: per-iteration drift + time ----------
orig_itq = LittleBitLinear._compute_itq_rotation
ITQ_LOG = []
def timed_itq(self, X, n_iter=20):
    ITQ_LOG.clear()
    t0 = time.time()
    with torch.no_grad():
        _, dim = X.shape
        device = X.device
        X_f = X.float()
        R = torch.empty((dim, dim), device=device, dtype=torch.float32)
        torch.nn.init.orthogonal_(R)
        eye = torch.eye(dim, device=device)
        prev = None
        for i in range(n_iter):
            ti = time.time()
            Z = X_f @ R
            B = torch.sign(Z)
            M = B.t() @ X_f
            U_p, _, Vt_p = torch.linalg.svd(M, full_matrices=False)
            R = Vt_p.t() @ U_p.t()
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - ti
            drift = torch.linalg.matrix_norm(R.t() @ prev - eye).item() if prev is not None else float("nan")
            ITQ_LOG.append((i, dt, drift))
            prev = R
    print(f"    ITQ total {time.time()-t0:.2f}s for {len(ITQ_LOG)} iters "
          f"(mean {sum(d for _,d,_ in ITQ_LOG)/len(ITQ_LOG)*1000:.0f} ms/iter)")
    print("    drift: " + " ".join(f"{i}:{dr:.1e}" for i, _, dr in ITQ_LOG[::5]) +
          (f" ... {ITQ_LOG[-1][0]}:{ITQ_LOG[-1][2]:.1e}" if ITQ_LOG else ""))
    return R.to(X.dtype)
LittleBitLinear._compute_itq_rotation = timed_itq

# ---------- stage timings on a gate_proj-shaped layer ----------
d_out, d_in = 1024, 3072
X = make(d_out, d_in).weight.data.float()

t0 = time.time(); U_t, S_t, V_t = torch.svd_lowrank(X, q=1024)
if dev == "cuda": torch.cuda.synchronize()
print(f"\nsvd_lowrank(q=1024) on {d_out}x{d_in}: {time.time()-t0:.2f}s")

t0 = time.time(); torch.linalg.svd(X)
if dev == "cuda": torch.cuda.synchronize()
print(f"linalg.svd (full)     on {d_out}x{d_in}: {time.time()-t0:.2f}s")

# ---------- full converts: the A/B matrix ----------
def convert(shape, itq, n_iter, residual, seed=0):
    global ITQ_LOG
    lin = make(*shape, seed=seed)
    W_ref = lin.weight.data.float().clone()
    t0 = time.time()
    lin.__quant_convert__(do_train=True, quant_func=STEBinary, split_dim=1024, eff_bit=1.0,
                          residual=residual, use_itq=itq, itq_n_iter=n_iter)
    if dev == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    x = torch.randn(4, shape[1], device=dev, dtype=torch.float16)
    with torch.no_grad():
        y = lin(x).float()
    y_ref = (x @ W_ref.half().t()).float()
    rel = ((y_ref - y).norm() / y_ref.norm()).item()
    return dt, rel

print()
for shape in [(1024, 1024), (1024, 3072)]:
    for itq, n_iter, res in [(False, 0, True), (True, 50, False), (True, 50, True), (True, 15, True), (True, 5, True)]:
        dt, rel = convert(shape, itq, n_iter, res)
        tag = f"itq={'Y' if itq else 'N'}/{n_iter or '-'} res={'Y' if res else 'N'}"
        print(f"{shape[0]}x{shape[1]:4d} {tag:16s} {dt:7.2f}s  init rel err {rel:.4f}")
print("\n(last ITQ log, 50-iter config):")

"""P0 validation: LittleBit core math + full QAT smoke test (synthetic, CPU, no downloads).

Checks:
  1. Pack/unpack round-trip is bit-exact
  2. Init fidelity vs dense W at ~0.9 BPW (INFORMATIONAL — init-only; QAT exists to close this)
  3. Effective-bit formula agrees with packed byte count
  4. QAT smoke: KD-train the toy student -> loss falls
  5. Save (main.py path) -> load (hub.py path) -> inference matches pre-save student
"""
import sys, json, tempfile, pathlib, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/littlebit-fork")
# NOTE: repo import-order landmine: quantization.utils MUST load before quantization.modules.
import quantization.utils.quant_util  # noqa: F401  (breaks the cycle first)
from quantization.modules.littlebit import LittleBitLinear
from quantization.utils.binary_packer import binary_packer, binary_unpacker
from quantization.hub import LittleBitModel, LittleBitConfig
from quantization.utils.quant_util import apply_littlebit_patch
from quantization.functions.binary import STEBinary

torch.manual_seed(0)
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

# ---------- 1. pack/unpack round-trip ----------
for shape in [(64, 32), (48, 100), (7, 33)]:  # incl. non-multiple-of-32 (padding path)
    t = torch.randint(0, 2, shape, dtype=torch.int8) * 2 - 1
    packed = binary_packer(t)
    back = binary_unpacker(packed, shape)
    check(f"pack_roundtrip {shape}", torch.equal(t, back),
          f"{t.numel()} elems -> {packed.numel()} int32 words")

# ---------- 2. init fidelity (informational) + 3. byte accounting ----------
d_in, d_out = 256, 256
g = torch.Generator().manual_seed(1)
W_ref = (torch.randn(d_out, 48, generator=g) @ torch.randn(48, d_in, generator=g)) * 0.05 \
        + torch.randn(d_out, d_in, generator=g) * 0.002
lin = nn.Linear(d_in, d_out, bias=False)
with torch.no_grad():
    lin.weight.copy_(W_ref)
lin = lin.half()
lin.__class__ = LittleBitLinear
lin.__quant_convert__(do_train=True, quant_func=STEBinary, split_dim=1024, eff_bit=1.0,
                      residual=True, use_itq=True, itq_n_iter=50)
with torch.no_grad():
    for p in lin.parameters():
        p.data = p.data.half()   # repo init upcasts to fp32 (bug #6); cast to match deploy dtype

x = torch.randn(8, d_in, dtype=torch.float16)
y_ref = x @ W_ref.half().t()
y_lb = lin(x)
rel = ((y_ref.float() - y_lb.float()).norm() / y_ref.float().norm()).item()
print(f"    INFO init rel err @ {lin._eff_bit_actual.item():.2f} BPW (split={lin.split_dim}): {rel:.4f}  <- QAT's job to close")

sd = lin.state_dict()
n_bytes = sum(t.numel() * t.element_size() for t in sd.values())
bpw = n_bytes * 8 / (d_in * d_out)
check("eff_bit_matches_bytes", abs(bpw - lin._eff_bit_actual.item()) < 0.25,
      f"bpw from bytes {bpw:.3f} vs formula {lin._eff_bit_actual.item():.3f}")

# ---------- toy LM config (transformers-5-safe) ----------
from transformers import AutoModelForCausalLM, AutoConfig
def make_tiny():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    cfg.intermediate_size = 256
    cfg.max_position_embeddings = 512
    cfg.vocab_size = 1000
    cfg.layer_types = ["full_attention", "full_attention"]   # must match layer count (t5 strict validation)
    # Build DIRECTLY on CPU (no meta+to_empty): to_empty leaves buffers (rope inv_freq)
    # as garbage, which corrupts the pre-save model and masquerades as a checkpoint bug.
    # A 1M-param toy costs nothing to construct for real.
    m = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32)
    with torch.no_grad():
        for p in m.parameters():
            p.data = torch.randn_like(p.data) * 0.15   # 0.02-scale logits ~ uniform => KD loss ~1e-4, no signal
    return m

quant_args = argparse.Namespace(quant_func="STEBinary", quant_mod="LittleBitLinear",
                                split_dim=1024, eff_bit=1.0, residual=True, kv_factor=1.0,
                                min_split_dim=8, use_itq=True, itq_n_iter=50)

teacher = make_tiny(); teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = make_tiny()
student.load_state_dict(teacher.state_dict())     # distill init = decompose teacher weights
apply_littlebit_patch(student, quant_args, do_train=True)

# freeze everything except LittleBitLinear factors/scales (mirrors prepare_model_for_training)
lb_param_ids = {id(p) for m in student.modules() if isinstance(m, LittleBitLinear)
                for p in m.parameters()}
for n, p in student.named_parameters():
    p.requires_grad = id(p) in lb_param_ids
n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"    trainable (factors+scales only): {n_train:,}")

# ---------- 4. QAT smoke: KD loss must fall ----------
L2L = 10.0
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1.5e-3)
gen = torch.Generator().manual_seed(7)
def batch():
    return torch.randint(0, 1000, (8, 32), generator=gen)

def kd_loss(model, ids):
    with torch.no_grad():
        t_out = teacher(input_ids=ids, output_hidden_states=True)
    s_out = model(input_ids=ids, output_hidden_states=True)
    kl = F.kl_div(F.log_softmax(s_out.logits.float(), -1),
                  F.softmax(t_out.logits.float(), -1), reduction="batchmean")
    mse = sum(F.mse_loss(sr.float(), tr.float()) for sr, tr in
              zip(s_out.hidden_states[1:], t_out.hidden_states[1:]))
    return kl + L2L * mse, kl.item(), mse.item()

student.train()
losses = []
for step in range(250):
    ids = batch()
    loss, kl_v, mse_v = kd_loss(student, ids)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)  # HF Trainer default
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0 or step == 249:
        print(f"    step {step:3d}  loss {loss.item():.6e}  (kd {kl_v:.4e} l2l {mse_v:.4e})")
# smoothed gate: compare window means to reject single-step noise
first20 = sum(losses[:20]) / 20
last20 = sum(losses[-20:]) / 20
check("qat_loss_falls", last20 < first20 * 0.75,
      f"window means: first20 {first20:.4e} -> last20 {last20:.4e} ({(1-last20/first20)*100:.1f}% drop)")
student.eval()

# ---------- 5. save via main.py path -> load via hub.py path -> outputs match ----------
try:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "ckpt"
    tmp.mkdir(parents=True)
    ids = batch()
    with torch.no_grad():
        pre = student(input_ids=ids).logits.clone()

    quant_params = {k: getattr(quant_args, k) for k in
                    ["quant_func", "eff_bit", "split_dim", "residual", "kv_factor",
                     "min_split_dim", "quant_mod", "use_itq", "itq_n_iter"]}
    with open(tmp / "littlebit_config.json", "w") as f:
        json.dump(quant_params, f, indent=2)
    for k, v in quant_params.items():
        setattr(student.config, k, v)
    student.config.use_cache = True
    student.save_pretrained(str(tmp), safe_serialization=True)   # state_dict() packs U/V

    files = sorted(p.name for p in tmp.iterdir())
    sz = sum(p.stat().st_size for p in tmp.iterdir())
    print(f"    saved: {files}  total {sz/1e6:.2f} MB")

    loaded = LittleBitModel.from_pretrained(str(tmp), torch_dtype=torch.float32, device="cpu")
    loaded.eval()
    with torch.no_grad():
        post = loaded(input_ids=ids).logits
    d = (pre.float() - post.float()).abs().max().item()
    check("checkpoint_save_load_roundtrip", d < 0.1, f"max logit delta {d:.2e}")
except Exception as e:
    import traceback; traceback.print_exc()
    check("checkpoint_save_load_roundtrip", False, f"{type(e).__name__}: {e}")

n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {n_pass}/{len(results)} PASS ===")
sys.exit(0 if n_pass == len(results) else 1)

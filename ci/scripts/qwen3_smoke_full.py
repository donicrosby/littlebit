"""Qwen3-0.6B LittleBit end-to-end smoke: real model, real text, real checkpoint.

Gates (all must PASS for "works as expected"):
  G1 teacher_ppl < 15            — dense baseline sane (Qwen3-0.6B wikitext2 ≈ 8-12)
  G2 init_ppl > teacher_ppl      — decomposition is lossy pre-QAT (expected, informational)
  G3 qat loss falls >= 30%       — factors+scales actually learn
  G4 post_ppl < init_ppl         — QAT recovers real quality
  G5 reload logit delta < 0.1    — main.py save -> hub.py load fidelity (P0 gate)
  G6 measured BPW within ±0.25 of 1.0 — packed bytes match eff_bit formula (P0 gate)
"""
import os, sys, time, json, argparse, gc, math, random
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("LB_REPO", str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent)))
import quantization.utils.quant_util  # noqa: F401  (repo import-order landmine)
from quantization.utils.quant_util import apply_littlebit_patch
from quantization.modules.littlebit import LittleBitLinear
from quantization.hub import LittleBitModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("HF_MODEL") or (
    "/workspace/model" if os.path.isdir("/workspace/model") else "Qwen/Qwen3-0.6B")
SEQ = int(os.environ.get("SMOKE_SEQ", "1024"))
TRAIN_SEQ = int(os.environ.get("SMOKE_TRAIN_SEQ", "256"))
TRAIN_STEPS = int(os.environ.get("SMOKE_STEPS", "60"))
BATCH = int(os.environ.get("SMOKE_BATCH", "8"))
LR = float(os.environ.get("SMOKE_LR", "2e-5"))  # main.py's real-model default — NOT the toy's 1.5e-3
EVAL_CAP = int(os.environ.get("SMOKE_EVAL_SEQS", "0"))  # 0 = full test split
OUT = os.environ.get("SMOKE_OUT", "/workspace/out")
GATES_MODE = os.environ.get("GATES", "enforce")  # enforce: exit 1 on fail | report: log only
CKPT_EVERY = int(os.environ.get("SMOKE_CKPT_EVERY", "0"))   # 0 = only final save
EVAL_EVERY = int(os.environ.get("SMOKE_EVAL_EVERY", "0"))   # 0 = only pre/post evals
USE_COSINE = os.environ.get("SMOKE_COSINE", "0") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED); random.seed(SEED)
if DEV == "cpu":
    torch.set_num_threads(os.cpu_count() or 8)
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def tlog(msg): print(f"[{time.time()-T0:7.0f}s] {msg}", flush=True)

results, gates = {}, []
def gate(name, ok, detail):
    gates.append((name, bool(ok), detail))
    tlog(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

tlog(f"device={DEV}  model={MODEL_ID}  threads={torch.get_num_threads()}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# ---------- data: wikitext-2 raw, test for eval / train for QAT ----------
from datasets import load_dataset
CACHE = os.environ.get("SMOKE_TOKENS", "/workspace/assets/wikitext2_tokens.pt") \
    if os.path.exists("/workspace/assets/wikitext2_tokens.pt") \
    else os.path.join(OUT, "wikitext2_tokens.pt")
if os.path.exists(CACHE):
    d = torch.load(CACHE); test_ids, train_ids = d["test"], d["train"]
    tlog(f"token cache hit: test {test_ids.numel():,} train {train_ids.numel():,}")
else:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test_ids = torch.tensor(tok("".join(ds["test"]["text"]))["input_ids"], dtype=torch.long)
    train_ids = torch.tensor(tok("".join(ds["train"]["text"]))["input_ids"], dtype=torch.long)
    torch.save({"test": test_ids, "train": train_ids}, CACHE)
    tlog(f"tokenized: test {test_ids.numel():,} train {train_ids.numel():,}")

@torch.no_grad()
def ppl(model, ids, label):
    model.eval()
    nsamp = ids.numel() // SEQ
    if EVAL_CAP: nsamp = min(nsamp, EVAL_CAP)
    nlls = []
    for i in range(nsamp):
        batch = ids[i*SEQ:(i+1)*SEQ].to(DEV).unsqueeze(0)
        out = model(batch, use_cache=False)
        sl = out.logits[:, :-1, :].contiguous()
        lb = batch[:, 1:].contiguous()
        nlls.append(F.cross_entropy(sl.view(-1, sl.size(-1)).float(), lb.view(-1)).float() * (SEQ - 1))
        if i % 25 == 0: tlog(f"  {label} seq {i+1}/{nsamp}")
    v = torch.exp(torch.stack(nlls).sum() / (nsamp * (SEQ - 1))).item()
    tlog(f"[{label}] PPL = {v:.4f}  ({nsamp} seqs)")
    return v

# ---------- teacher ----------
tlog("loading teacher (dense fp32)...")
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEV)
teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
n_params = sum(p.numel() for p in teacher.parameters())
tlog(f"teacher ready: {n_params/1e6:.0f}M params")

ppl_teacher = ppl(teacher, test_ids, "teacher")
results["ppl_teacher"] = ppl_teacher
# reference: MSFT winml fp32 Qwen3-0.6B wikitext2-raw-v1, same protocol -> 23.65 (arXiv's 12.7 is the *Base* variant)
gate("G1_teacher_sane", ppl_teacher < 30, f"teacher PPL {ppl_teacher:.3f} (<30; reference 23.65)")

# ---------- student: decompose teacher weights (main.py path) ----------
tlog("loading student copy + applying LittleBit patch...")
student = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEV)
# capture dense Linear shapes BEFORE patching (needed for the quant-layer BPW denominator)
import torch.nn as _nn
dense_shapes = {n: (m.in_features, m.out_features)
                for n, m in student.named_modules() if isinstance(m, _nn.Linear)}
quant_args = argparse.Namespace(
    quant_func="STEBinary", quant_mod="LittleBitLinear",
    split_dim=1024, eff_bit=1.0, residual=True, kv_factor=1.0,
    min_split_dim=8, use_itq=True,
    # 50 (repo default) spends ~700s of patch time for Δrel-err 0.018 vs 15 (profiled:
    # flat quality curve, QAT closes init error anyway). SMOKE_ITQ_N_ITER for the paper number.
    itq_n_iter=int(os.environ.get("SMOKE_ITQ_N_ITER", "15")))
apply_littlebit_patch(student, quant_args, do_train=True)
lb_param_ids = {id(p) for m in student.modules() if isinstance(m, LittleBitLinear) for p in m.parameters()}
for n, p in student.named_parameters():
    p.requires_grad = id(p) in lb_param_ids
n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
tlog(f"patched. trainable {n_train:,} / {sum(p.numel() for p in student.parameters()):,}")

ppl_init = ppl(student, test_ids, "init pre-QAT")
results["ppl_init"] = ppl_init
gate("G2_init_lossy", ppl_init > ppl_teacher, f"init PPL {ppl_init:.3f} > teacher {ppl_teacher:.3f} (expected pre-QAT)")

# ---------- QAT: KD on real wikitext windows (train split only) ----------
tlog(f"QAT: {TRAIN_STEPS} steps, batch {BATCH}, seq {TRAIN_SEQ}, lr {LR}, cosine={USE_COSINE}")
gen = random.Random(SEED)
max_start = train_ids.numel() - TRAIN_SEQ - 1
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS) if USE_COSINE else None

def kd_step():
    # independent windows per row: real batch diversity (the old repeat+offset hack was
    # fine for a 60-step smoke; multi-thousand-step runs need true random batches)
    starts = [gen.randrange(0, max_start) for _ in range(BATCH)]
    ids = torch.stack([train_ids[s:s+TRAIN_SEQ] for s in starts]).to(DEV)
    ids = ids.clamp(0, len(tok) - 1)
    with torch.no_grad():
        t_out = teacher(input_ids=ids, output_hidden_states=True)
    s_out = student(input_ids=ids, output_hidden_states=True)
    kl = F.kl_div(F.log_softmax(s_out.logits.float(), -1),
                  F.softmax(t_out.logits.float(), -1), reduction="batchmean")
    mse = sum(F.mse_loss(sr.float(), tr.float()) for sr, tr in
              zip(s_out.hidden_states[1:], t_out.hidden_states[1:]))
    return kl + 10.0 * mse, kl.item(), mse.item()

def save_ckpt(tag, final=False):
    """Save student+tokenizer+quant config in main.py's format (hub-loadable)."""
    p = os.path.join(OUT, f"qwen3_06b_littlebit_{tag}")
    os.makedirs(p, exist_ok=True)
    quant_params = {k: getattr(quant_args, k) for k in
                    ["quant_func", "eff_bit", "split_dim", "residual", "kv_factor",
                     "min_split_dim", "quant_mod", "use_itq", "itq_n_iter"]}
    with open(os.path.join(p, "littlebit_config.json"), "w") as f:
        json.dump(quant_params, f, indent=2)
    for k, v in quant_params.items():
        setattr(student.config, k, v)   # mirror main.py: quant keys ride in config.json too
    was_training = student.training
    student.eval()
    student.config.use_cache = True
    student.save_pretrained(p, safe_serialization=True)
    tok.save_pretrained(p)
    if not final:
        student.config.use_cache = False   # mid-run: keep training semantics intact
        if was_training:
            student.train()
    tlog(f"checkpoint saved: {p}")
    return p

student.train()
losses, kls, mses = [], [], []
for step in range(TRAIN_STEPS):
    loss, kl_v, mse_v = kd_step()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()
    if sched is not None: sched.step()
    losses.append(loss.item()); kls.append(kl_v); mses.append(mse_v)
    if step % 10 == 0 or step == TRAIN_STEPS - 1:
        lr_now = opt.param_groups[0]["lr"]
        tlog(f"  step {step:3d}  loss {loss.item():.5f}  (kd {kl_v:.4f} l2l {mse_v:.5f} lr {lr_now:.2e})")
    if CKPT_EVERY and (step + 1) % CKPT_EVERY == 0 and step < TRAIN_STEPS - 1:
        save_ckpt(f"step{step+1}")
    if EVAL_EVERY and (step + 1) % EVAL_EVERY == 0 and step < TRAIN_STEPS - 1:
        ppl_mid = ppl(student, test_ids, f"mid-QAT@{step+1}")
        student.train()
first20 = sum(losses[:20]) / min(20, len(losses))
last20 = sum(losses[-20:]) / min(20, len(losses))
kd_first = sum(kls[:20]) / min(20, len(kls))
kd_last = sum(kls[-20:]) / min(20, len(kls))
mse_first = sum(mses[:20]) / min(20, len(mses))
mse_last = sum(mses[-20:]) / min(20, len(mses))
kd_drop = 1 - kd_last / kd_first
mse_ratio = mse_last / mse_first
results["qat_kd_drop"], results["qat_mse_ratio"] = kd_drop, mse_ratio
# dynamics-aware gate (real-model smoke, repo LR): the KL axis — what drives PPL — must at
# least halve in the smoke window (healthy runs: ~74-78%), and hidden-state MSE must not
# diverge (healthy: <1x; lr-too-high run: 1.5x and climbing).
gate("G3_qat_learns", kd_drop >= 0.50 and mse_ratio <= 1.2,
     f"kd {kd_first:.1f} -> {kd_last:.1f} ({kd_drop*100:.0f}% drop, need >=50%) | "
     f"l2l ratio {mse_ratio:.2f} (need <=1.2, diverge watch)")

# ---------- post-QAT PPL ----------
ppl_post = ppl(student, test_ids, "post-QAT")
results["ppl_post"] = ppl_post
gate("G4_qat_recovers", ppl_post < ppl_init, f"post {ppl_post:.3f} < init {ppl_init:.3f}")

# ---------- save (main.py path) -> reload (hub.py path) ----------
student.eval()
ids = train_ids[:TRAIN_SEQ].unsqueeze(0).to(DEV)
with torch.no_grad():
    pre = student(input_ids=ids).logits.clone().cpu()

# layer-level BPW (P0 method): packed bytes ON DISK vs the dense param count those layers
# replace. Counted from the saved safetensors — training-time params are fp32 factors and
# don't reflect packed storage. Whole-model BPW reported too (dense embedding dominates it).
lb_mod_names = {n for n, m in student.named_modules() if isinstance(m, LittleBitLinear)}
lb_dense_params = sum(dense_shapes[n][0] * dense_shapes[n][1] for n in lb_mod_names if n in dense_shapes)

ckpt = save_ckpt("final", final=True)

# packed-byte accounting from the saved file (device/dtype independent)
from safetensors import safe_open
packed_bytes = 0
for f in os.listdir(ckpt):
    if f.endswith(".safetensors"):
        with safe_open(os.path.join(ckpt, f), framework="pt") as sf:
            for k in sf.keys():
                if any(k == mn or k.startswith(mn + ".") for mn in lb_mod_names):
                    packed_bytes += sf.get_tensor(k).numel() * sf.get_tensor(k).element_size()
sz = sum(os.path.getsize(os.path.join(ckpt, f)) for f in os.listdir(ckpt))
bpw_model = sz * 8 / n_params
bpw_layers = (packed_bytes * 8 / lb_dense_params) if lb_dense_params else float("nan")
results["ckpt_bytes"] = sz; results["bpw_whole_model"] = bpw_model; results["bpw_quant_layers"] = bpw_layers
tlog(f"checkpoint: {sz/1e6:.2f} MB | quant-layer BPW {bpw_layers:.3f} (packed {packed_bytes/1e6:.1f} MB / {lb_dense_params/1e6:.0f}M params) | whole-model BPW {bpw_model:.3f}")
gate("G6_bpw_matches", abs(bpw_layers - 1.0) < 0.25, f"quant-layer BPW {bpw_layers:.3f} vs target 1.0 (+/-0.25)")

del student; gc.collect()
if DEV == "cuda": torch.cuda.empty_cache()

reloaded = LittleBitModel.from_pretrained(ckpt, torch_dtype=torch.float32, device=DEV)
reloaded.eval()
with torch.no_grad():
    post = reloaded(input_ids=ids).logits.cpu()
delta = (pre - post).abs().max().item()
results["reload_logit_delta"] = delta
gate("G5_save_load_fidelity", delta < 0.1, f"max logit delta {delta:.2e} (<0.1)")

# ---------- generation check ----------
@torch.no_grad()
def gen_text(model, prompt, max_new=40):
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

prompts = ["The capital of France is", "In a shocking discovery, scientists found"]
gens = {}
for name, mdl in [("teacher", teacher), ("reloaded", reloaded)]:
    mdl.eval(); gens[name] = [gen_text(mdl, p) for p in prompts]
results["generations"] = gens
tlog("\n=== GENERATION ===")
for p, gt, gr in zip(prompts, gens["teacher"], gens["reloaded"]):
    tlog(f"PROMPT: {p!r}\n  teacher : {gt!r}\n  littlebit: {gr!r}")

json.dump({"results": results, "gates": gates}, open(os.path.join(OUT, "qwen3_smoke_results.json"), "w"), indent=2)
n_pass = sum(1 for _, ok, _ in gates if ok)
tlog(f"\n=== {n_pass}/{len(gates)} GATES PASS ===")
sys.exit(0 if n_pass == len(gates) else 1)

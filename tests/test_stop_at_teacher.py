"""Unit test: SMOKE_STOP_AT_TEACHER early-stop control flow.

Executes the ACTUAL training-loop block from ci/scripts/qwen3_smoke_full.py
(sliced from source, not a transcription) against stubbed components, so the
flag's break semantics are verified without loading any model. Runs in ~2s on
CPU — no GPU, no transformers, no network.
"""
import os, sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(REPO, "ci", "scripts", "qwen3_smoke_full.py")).read()

START = SRC.index("student.train()\nlosses")  # col-0 line: real loop preamble (save_ckpt's is indented)
END = SRC.index("first20 = ")
BLOCK = SRC[START:END]
assert "STOP_AT_TEACHER" in BLOCK, "loop block no longer contains the flag — update this test"

def run_loop(stop_flag, eval_every, ppl_value, steps=12, ckpt_every=0):
    logs = []
    state = {"ckpts": 0, "evals": 0}
    param = torch.zeros(1, requires_grad=True)

    class FakeLoss:
        def backward(self): pass
        def item(self): return 1.0

    class FakeStudent:
        def train(self): pass
        def parameters(self): return [param]

    g = {
        "TRAIN_STEPS": steps, "BATCH": 1, "CKPT_EVERY": ckpt_every,
        "EVAL_EVERY": eval_every, "STOP_AT_TEACHER": stop_flag,
        "ppl_teacher": 22.877, "ppl_mid": None,
        "test_ids": None, "train_ids": None,  # names must resolve; stub ppl ignores them
        "student": FakeStudent(), "sched": None,
        "opt": torch.optim.SGD([param], lr=2e-5),
        "losses": [], "kls": [], "mses": [],
        "kd_step": lambda: (FakeLoss(), 1.0, 1.0),
        "save_ckpt": lambda tag, final=False: state.__setitem__("ckpts", state["ckpts"] + 1),
        "ppl": lambda model, ids, label: (state.__setitem__("evals", state["evals"] + 1), ppl_value)[1],
        "tlog": lambda m: logs.append(m),
        "torch": torch,
    }
    exec(BLOCK, g)  # noqa: S102 - deliberate: test the real source, verbatim
    return g["losses"], logs, state

def test_flag_off_runs_full():
    losses, logs, st = run_loop(stop_flag=False, eval_every=5, ppl_value=20.0, ckpt_every=5)
    assert len(losses) == 12, f"flag off must not break early, got {len(losses)}"
    assert st["evals"] == 2 and st["ckpts"] == 2
    assert not any("STOP_AT_TEACHER" in l for l in logs)

def test_flag_on_breaks_at_first_parity():
    losses, logs, st = run_loop(stop_flag=True, eval_every=5, ppl_value=20.0)
    assert len(losses) == 5, f"must break at step 5 (first eval <= teacher), got {len(losses)}"
    assert st["evals"] == 1 and any("SMOKE_STOP_AT_TEACHER" in l for l in logs)

def test_flag_on_no_parity_runs_full():
    losses, logs, st = run_loop(stop_flag=True, eval_every=5, ppl_value=100.0)
    assert len(losses) == 12, "PPL above teacher must run to completion"
    assert st["evals"] == 2

def test_flag_on_without_evals_never_breaks():
    losses, logs, st = run_loop(stop_flag=True, eval_every=0, ppl_value=20.0)
    assert len(losses) == 12, "no evals = no trigger point = full run (documented)"

def test_boundary_parity_exactly_equal_breaks():
    losses, logs, _ = run_loop(stop_flag=True, eval_every=5, ppl_value=22.877)
    assert len(losses) == 5, "PPL == teacher must trigger (<= semantics)"

if __name__ == "__main__":
    for fn in [test_flag_off_runs_full, test_flag_on_breaks_at_first_parity,
               test_flag_on_no_parity_runs_full, test_flag_on_without_evals_never_breaks,
               test_boundary_parity_exactly_equal_breaks]:
        fn(); print(f"PASS {fn.__name__}")
    print("5/5 PASS")

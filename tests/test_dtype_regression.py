"""Regression: (1) modules-first import must work (cycle fix), (2) bf16 layer
patch + forward must not crash on dtype mismatch (dtype fix). Both failed on upstream."""
import sys, torch, torch.nn as nn
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from quantization.modules.littlebit import LittleBitLinear  # modules FIRST — cycle proof
from quantization.functions.binary import STEBinary

lin = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
with torch.no_grad():
    lin.weight.copy_(torch.randn(64, 64) * 0.05)
lin.__class__ = LittleBitLinear
lin.__quant_convert__(do_train=True, quant_func=STEBinary, split_dim=64, eff_bit=1.0,
                      residual=True, use_itq=False)
x = torch.randn(2, 64, dtype=torch.bfloat16)
y = lin(x)
assert y.dtype == torch.bfloat16, y.dtype
sd_keys = lin.state_dict().keys()
assert not any(k.endswith(".U") or k.endswith(".V") for k in sd_keys), "raw U/V leaked into state_dict"
print("PASS: bf16 forward OK, no raw U/V in state_dict, modules-first import OK")

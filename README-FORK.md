# littlebit (fork)

Fork of SamsungLabs/LittleBit (vendored at upstream 42d658b0c79f76450b34b6a3547462c7cdc3e1a0).
License: CC BY-NC 4.0 inherited — non-commercial use only.

## Divergences from upstream (see commits after the vendor commit)
1. circular import broken (lazy imports in quant_util)
2. extra_repr / total_bit_usage attribute errors fixed
3. state_dict prefix-drop bug fixed (raw U/V no longer leak into checkpoints)
4. dtype crash fixed (factors cast to layer dtype after fp32 decomposition)
5. hub.py: config.to_dict() for transformers>=5 compat
6. dead shadowed save_artifacts removed
7. telemetry default wandb -> tensorboard (no phone-home)
tests/run_p0.py: P0 validation suite; tests/test_dtype_regression.py: regression for #4

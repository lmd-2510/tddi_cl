# RealMLP-TD numerical port attribution

`src/models/realmlp_td_member.py` adapts the numerical architecture and training
equations from David Holzmüller's Apache-2.0-licensed PyTabKit:

- Repository: https://github.com/dholzmueller/pytabkit
- Pinned reference commit: `c126ea51187c5080b91f28d352481dbd3b2194b0`
- `pytabkit/models/sklearn/default_params.py`: `RealMLP_TD_CLASS`.
- `pytabkit/models/nn_models/nn.py`: PBLD initialization/forward, std initialization,
  He+5 initialization, front scale, dropout, label smoothing.
- `pytabkit/models/nn_models/activations.py`: parametric SELU.
- `pytabkit/models/training/scheduling.py`: coslog4 and flat_cos.
- `pytabkit/models/optim/optimizers.py`: Adam with decoupled decay and parameter factors.
- `pytabkit/models/training/lightning_modules.py`: optional early stopping (2 × best epoch + 20).

This port uses ordinary PyTorch modules with class-major head tensors so that the
existing CIL expansion, EWC, A-GEM, state_dict and prediction contracts work. It
retains PBLD, unlike RealMLP-TD-S. The dependency-heavy sklearn/Lightning orchestration,
categorical processing and dataset benchmark framework are not imported.

Adaptations are explicit in the run manifest/runbook: external task-0 preprocessing,
Macro-F1 selection, continual parameter carry-over, bounded train-only initialization,
and a 1e-6 initialization standard-deviation floor for constant projections. The
optional early-stopping rule is enabled. Ties retain the first best epoch, as in the
existing repo. These choices are not described as a bit-for-bit reproduction of the
original non-continual benchmark.

The adjacent LICENSE preserves the upstream Apache 2.0 license. The local adapter
changes module layout, tensor orientation and the continual-training interface;
the upstream project is not responsible for these modifications.

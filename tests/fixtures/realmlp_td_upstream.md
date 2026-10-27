# Numerical reference provenance

`realmlp_td_upstream.json` was generated from PyTabKit commit
`c126ea51187c5080b91f28d352481dbd3b2194b0` using float64 tensors on CPU:

- PBLD: the original `PLREmbeddingsLayerCosBias.forward_cont` with `linear` activation
  and DenseNet concatenation; squared-mean-output backward supplies all four gradients.
- Optimizer: the original `OptimizerBase` / `AdamOptimizer`, three Adam steps with
  initial parameter `[1, -2]`, gradients `[.3, -.6]`, `[-.1, .2]`, `[.5, .4]`, LR `.04`,
  WD `.02`, betas `(.9, .95)`, epsilon `1e-8` and the recorded parameter factors.
- Schedules: the original `get_schedule('coslog4')` and `get_schedule('flat_cos')`.

Only framework/context plumbing was stubbed when instantiating the original classes;
the original forward, backward and optimizer equations were executed unchanged.
Tests read the frozen results without downloading or executing remote code.
Upstream license and attribution: `third_party/pytabkit/`.

# MetaTransfer v0.1 — Disjoint Channel-Importance Transfer

- Primary metric: **Top-10% precision**.
- Source score examples: **1250**; target evaluation examples: **1250**.
- Random channel permutations per target/layer: **1000**.

## Primary results

| Target | Layer | Same-index P@10% | Aligned P@10% | Random-map mean P@10% | Aligned Δ vs same |
|---|---|---:|---:|---:|---:|
| seed123 | `layer2.1.bn1` | 0.077 | 0.154 | 0.104 | +0.077 |
| seed123 | `layer3.1.bn1` | 0.077 | 0.115 | 0.102 | +0.038 |
| seed123 | `layer4.1.bn1` | 0.154 | 0.115 | 0.102 | -0.038 |
| seed456 | `layer2.1.bn1` | 0.077 | 0.154 | 0.103 | +0.077 |
| seed456 | `layer3.1.bn1` | 0.115 | 0.154 | 0.102 | +0.038 |
| seed456 | `layer4.1.bn1` | 0.115 | 0.096 | 0.102 | -0.019 |
| seed789 | `layer2.1.bn1` | 0.077 | 0.077 | 0.103 | +0.000 |
| seed789 | `layer3.1.bn1` | 0.000 | 0.077 | 0.100 | +0.077 |
| seed789 | `layer4.1.bn1` | 0.038 | 0.115 | 0.103 | +0.077 |

## Aggregate

- Mean aligned P@10%: **0.1175**
- Mean same-index P@10%: **0.0812**
- Mean random-map P@10%: **0.1025**
- Aligned beats same-index: **6/9** comparisons
- Target seeds with positive layer-averaged delta: **3/3**

## Limitations

- The 5,000-image validation partition has previously been used for checkpoint selection; this is not a pristine external benchmark.
- There are three target checkpoints; layer rows within a checkpoint are dependent and must not be treated as nine independent model replications.
- The experiment tests channel-importance transfer within one dataset and one architecture, not semantic understanding, general intelligence, architecture generation, or reduced pretraining requirements.
- Single-channel ablation importance is intervention-dependent and does not capture channel interactions or global causal structure.
- Signed-effect correlation is secondary and may be unstable for channels with near-zero effects.
- Random-mapping empirical probabilities are permutation controls, not stand-alone population-level significance tests.

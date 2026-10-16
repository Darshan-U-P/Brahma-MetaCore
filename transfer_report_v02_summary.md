# MetaTransfer v0.2 — External-image evaluation summary

**Dataset:** CIFAR-10.1 v6, 2,000 images.  
**Purpose:** Test external-image generalization of the frozen v0.1 importance-transfer procedure. This was not new-target-seed replication.

| Method | Mean Top-10% precision |
|---|---:|
| Activation-aligned | 13.0342% |
| Same-index | 13.2479% |
| Random-mapping mean | 10.1833% |

- Aligned transfer beat same-index transfer in 3/9 target-layer comparisons and tied in 1/9.
- Aligned transfer beat the random-mapping mean in 6/9 comparisons.
- The predefined directional success rule was false.
- Target seeds were still 123, 456 and 789, reused from v0.1.
- The nine layer-level comparisons are dependent; they are not nine independent replications.

**Conclusion:** the v0.1 importance-transfer advantage did not replicate on the external-image evaluation. Alignment's average precision was slightly below same-index matching. The evidence supports continued investigation, not a successful general transfer claim.

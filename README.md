# Brahma MetaCore

## The Intelligence That Builds Intelligence

Brahma MetaCore is an experimental AI research project investigating whether information extracted from an existing AI model can help analyze, construct, transfer knowledge to, or modify another model.

The long-term goal is to develop a system that can **map a model, learn from its structure and behavior, and use that information to make better model-construction or model-editing decisions**. An eventual ambition is to discover reusable construction principles that reduce the data and computation needed to achieve useful capabilities.

These are research goals, not established capabilities. Each stage must be tested against controlled baselines, and conclusions must remain proportional to the evidence.

## Current Evidence Snapshot (15 October 2026)

Brahma MetaCore has progressed from model inspection to a controlled, preliminary cross-model channel-importance transfer experiment. The strongest current evidence is **representational correspondence**; practical importance transfer is promising but not yet reliable enough to claim a general advantage.

| Finding | Measured result | Interpretation |
|---|---:|---|
| Activation alignment, `layer2.1.bn1` | 0.651 matched vs 0.247 same-index mean absolute correlation | Stronger activation correspondence after matching |
| Activation alignment, `layer3.1.bn1` | 0.598 vs 0.221 | Stronger activation correspondence after matching |
| Activation alignment, `layer4.1.bn1` | 0.692 vs 0.233 | Stronger activation correspondence after matching |
| MetaTransfer v0.1 mean Top-10% precision | 11.75% aligned vs 8.12% same-index vs 10.25% random-map mean | Initial transfer signal; the advantage varies by target and layer |

![Activation alignment graph](assets/activation_alignment_v06.png)

![MetaTransfer aggregate precision graph](assets/metatransfer_precision_aggregate.png)

**Evidence boundary:** the experiments cover CIFAR-10 and CIFAR-adapted ResNet-18 checkpoints. The held-out target-effect subset was disjoint from source-importance scoring and feature-signature examples, but it comes from a validation partition used in earlier work. These results do **not** demonstrate semantic understanding, general intelligence, successful model construction, or reduced pretraining requirements.


## Core Research Loop

1. **Map:** Inspect a model's architecture, parameters, weights, activations, and measurable behavior.

2. **Analyze:** Investigate how components affect behavior and identify potentially reusable structures or parameter relationships.

3. **Predict:** Estimate the effects of interventions or modifications before applying them.

4. **Construct or edit:** Use information from the map to initialize, transfer, modify, or design a target model.

5. **Evaluate:** Measure capabilities, retained performance, data requirements, compute cost, and regressions.

6. **Improve:** Use the measured outcomes to revise the mapping, analysis, and construction methods.

The central transition is from observing a model to using observations to make better decisions about another model.

## Research Objectives

1. **Neural-network mapping:** Inspect architectures, parameters, weights, activations, and computational relationships.

2. **Component analysis:** Investigate component contributions through controlled interventions.

3. **Predictive model understanding:** Predict intervention effects and test generalization to independently trained models.

4. **Representation alignment:** Find correspondences between internal activation patterns across models.

5. **Model construction:** Use information extracted from a source model to help initialize, transfer, or construct a target model.

6. **Model editing and knowledge integration:** Add information or task capabilities while measuring retention of existing capabilities.

7. **Parameter generation and modification:** Explore algorithms that generate, initialize, select, or modify useful parameters.

8. **Reduced training requirements:** Test whether model mapping and AI-assisted construction reduce data or compute requirements at comparable capability levels.

9. **Recursive model construction:** Investigate systems that analyze models, propose operations, evaluate outcomes, and improve their construction strategies.

---

## Experimental Setup

- **Task:** CIFAR-10 image classification.

- **Model family:** CIFAR-adapted ResNet-18.

- **Framework:** PyTorch and torchvision.

- **Hardware:** NVIDIA RTX 3050 Ti Laptop GPU, 4 GB VRAM.

- **Environment:** Windows, Python 3.12.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121.

- **Model size:** Approximately 11.17 million parameters.

- **Validation protocol:** The later training runs use 45,000 training examples and 5,000 validation examples from the CIFAR-10 training partition. The official CIFAR-10 test set was not loaded in v0.5-v0.7 or MetaTransfer v0.1. Earlier exploratory training and v0.3 did use it, so it is not treated as a pristine benchmark.

- **Reproducibility:** Experiments record seeds, checkpoints, split indices, configurations, and machine-readable JSON reports.

The hardware and software details describe the current experimental environment; results may differ on other systems.

## Experimental Progress

### 1. MetaObserver v0.1 — Neural-network inspection

**Status: Implemented and tested**

The inspection infrastructure supports:

- Model architecture and module inspection.

- Parameter counting and model metadata collection.

- Activation capture using hooks.

- Activation-statistic collection and structured reporting.

- Hook cleanup and automated tests.

The initial smoke test inspected an approximately 11.17-million-parameter ResNet-18 and captured activations from 52 modules. The smoke test used random weights, so it validated the instrumentation rather than learned capabilities.

### 2. Training and weight analysis

**Status: Completed for the current CIFAR-10 baseline**

An exploratory training run achieved a best observed test accuracy of **92.38% at epoch 19**. The official test set was evaluated repeatedly and used to select the best epoch, so this is not an unbiased final benchmark.

A later, cleaner protocol used 45,000 training examples and 5,000 validation examples and achieved **92.26% validation accuracy at epoch 20**.

A comparison of initial and trained parameters found changes across 102 state tensors and a relative global L2 parameter change of approximately 0.915. This confirms substantial parameter change during training; parameter change alone does not explain the learned mechanisms.

### 3. MetaAnalyzer v0.2 — Controlled channel interventions

**Status: Completed**

A channel-ablation pipeline evaluated **896 interventions** across three batch-normalization layers, using 5,000 validation examples and random-control channels. The full scan took approximately 35 minutes.

The experiments established a systematic way to measure changes in loss and accuracy after zeroing channel outputs. They did **not** establish a strong, consistent advantage for all selected candidate channels over controls. Selection bias must be considered when interpreting the strongest observed effects.

Channel-ablation sensitivity is not, by itself, proof that a channel represents a semantic concept or an independently identifiable capability.

### 4. MetaAnalyzer v0.3 — Frozen-channel confirmation

**Status: Completed; interpretation limited**

A follow-up evaluated 39 previously selected candidate and control channels on 10,000 CIFAR-10 test examples. The baseline achieved **91.53% accuracy**, with mean loss of approximately **0.2638**.

The layer-2 candidate group had a mean loss increase of approximately **0.00451**, compared with approximately **0.00170** for its controls. Layer-3 results were weaker, while layer-4 candidates showed a small positive mean loss change relative to controls.

These are descriptive results, not statistically conclusive evidence. The same test split had already been repeatedly evaluated during the earlier training experiment and used for model selection, so it was not an untouched confirmation set.

### 5. MetaAnalyzer v0.4 — Predicting intervention effects from model statistics

**Status: Completed; predictive advantage not demonstrated**

A Random Forest regressor predicted absolute channel-ablation loss changes from activation statistics, batch-normalization parameters, and incoming/outgoing weight norms. The dataset contained 896 channel records, with held-out channels evaluated within each layer.

| Layer | Random Forest Spearman | Best tested single-feature heuristic |

|---|---:|---:|

| `layer2.1.bn1` | 0.497 | 0.611 — outgoing weight norm |

| `layer3.1.bn1` | 0.300 | 0.330 — activation magnitude |

| `layer4.1.bn1` | 0.475 | 0.501 — incoming weight norm |

The Random Forest did not outperform the best individual heuristic in any layer. Features and labels came from the same validation split, channels are not independent observations, and holding out channels does not establish generalization to new models. This was a predictive pilot, not a demonstration of generalizable model understanding.

### 6. MetaAnalyzer v0.5 — Cross-model importance prediction

**Status: Completed; initial generalization signal, no clear win over simple heuristics**

This experiment tested transfer from a seed-42 source model to an independently trained seed-123 target model.

- Feature extraction and intervention labels used separate subsets of 2,500 examples each from the saved validation partition.

- The source predictor was evaluated against measured target-model intervention effects; target labels were not used to fit the predictor.

- The experiment measured 896 channels in each of two models, for 1,792 interventions, taking approximately 35 minutes.

Key target-model results:

| Method | Spearman correlation | Top-20 precision |

|---|---:|---:|

| Random Forest | 0.545 | 0.40 |

| Incoming weight norm | 0.550 | 0.35 |

| Outgoing weight norm | 0.524 | 0.35 |

| Activation mean absolute value | 0.479 | 0.35 |

| Activation standard deviation | 0.380 | 0.45 |

The Random Forest showed a positive cross-model rank correlation, but incoming weight norm had a slightly higher Spearman correlation and activation standard deviation had higher top-20 precision. Results differed by layer. This is preliminary evidence that some source-model statistics relate to target-model importance, not proof of a reliable general predictor.

### 7. MetaAnalyzer v0.6 — Activation-based channel alignment

**Status: Full experiment completed; alignment signal observed, importance-transfer benefit mixed**

#### Research question

Can activation signatures identify corresponding channels across independently trained models, and does that alignment improve transfer of channel-importance information compared with matching channels by index?

#### Protocol

- **Models:** Four CIFAR-10 ResNet-18 checkpoints trained with seeds 42, 123, 456, and 789.

- **Layers:** `layer2.1.bn1` (128 channels), `layer3.1.bn1` (256 channels), and `layer4.1.bn1` (512 channels).

- **Feature subset:** 2,500 validation examples used to collect per-image, spatially pooled activation signatures and model features.

- **Intervention subset:** A separate 2,500 validation examples used to measure the absolute loss change from zeroing each channel.

- **Interventions:** 896 channels per model, **3,584 total**.

- **Matching:** Absolute Pearson correlations between activation signatures, with Hungarian assignment to produce a one-to-one channel mapping.

- **Evaluation:** Compare source-model measured channel importance against target-model measured importance using same-index and activation-aligned correspondence, with random-ranking references.

- **Official CIFAR-10 test set:** Not loaded during this experiment.

- **Runtime:** 4,288.2 seconds, approximately 71.5 minutes.

- **Report:** `reports/milestone2_v06/channel_alignment.json`

- **Caches:** `reports/milestone2_v06/cache/`

#### Baseline measurements on the intervention subset

| Seed | Accuracy | Loss |

|---|---:|---:|

| 42 (source) | 92.00% | 0.245267 |

| 123 | 92.20% | 0.267273 |

| 456 | 91.56% | 0.262167 |

| 789 | 92.40% | 0.255667 |

These are measurements on the 2,500-example intervention subset, not official test-set benchmarks.

#### Finding A: activation matching improves activation correspondence

Mean absolute matched activation correlation, averaged across the three target seeds:

| Layer | Activation-matched correlation | Same-index correlation |

|---|---:|---:|

| `layer2.1.bn1` | 0.651 | 0.247 |

| `layer3.1.bn1` | 0.598 | 0.221 |

| `layer4.1.bn1` | 0.692 | 0.233 |

Matched correlations were substantially higher than same-index correlations. Fewer than 1.6% of matched channels retained the same index in any reported target-layer comparison.

**Supported conclusion:** For these models and activation signatures, activation-based matching finds more correlated channel patterns than same-index matching. This does not prove that matched channels have identical semantic functions.

#### Finding B: importance transfer is mixed

Top-20 precision for predicting the target model's 20 most important channels from source-model measured importance:

| Target seed | Layer | Same-index precision | Aligned precision | Random expectation |

|---|---|---:|---:|---:|

| 123 | Layer 2 | 0.15 | 0.25 | 0.156 |

| 123 | Layer 3 | 0.00 | 0.10 | 0.078 |

| 123 | Layer 4 | 0.10 | 0.05 | 0.039 |

| 456 | Layer 2 | 0.20 | 0.25 | 0.156 |

| 456 | Layer 3 | 0.10 | 0.20 | 0.078 |

| 456 | Layer 4 | 0.05 | 0.00 | 0.039 |

| 789 | Layer 2 | 0.20 | 0.30 | 0.156 |

| 789 | Layer 3 | 0.00 | 0.05 | 0.078 |

| 789 | Layer 4 | 0.05 | 0.20 | 0.039 |

Aligned top-20 precision was higher in **7 of 9 comparisons**, but some aligned results were below the random expectation, and improvements varied by layer and seed. Spearman correlations were generally weak. These descriptive results do not establish statistical significance or consistent importance-transfer improvement.

#### v0.6 limitations

- One dataset and one architecture family were tested.

- There are three target seeds and one source seed in this experiment.

- Channel observations within a model are correlated; they must not be treated as independent experimental replications.

- The activation mapping is derived from a finite feature subset and needs stability testing across subsets.

- The source scores are measured intervention effects, not a learned general-purpose importance predictor.

- The experiment evaluates importance correspondence, not an actual parameter-transfer, knowledge-transfer, or model-construction operation.

- No claim about general intelligence, semantic understanding, or reduced pretraining is supported by this experiment.

#### Overall v0.6 conclusion

The experiment provides a useful signal for **cross-model representation alignment**. Whether that signal yields a reliable practical advantage for transferring importance information remains unresolved. A controlled experiment that uses the alignment to modify a target model is needed to establish practical value.

---

## 8. MetaAnalyzer v0.7 — Robust Validation of Alignment

**Status: Completed; alignment robustness supported, importance-transfer benefit mixed.**

v0.7 reanalyzed v0.6 results using repeated subsets, randomized mappings, and channel-level bootstrap diagnostics. It compared nine target-layer cases (three target seeds × three layers).

| Layer | Mean matched-minus-same-index absolute activation correlation | Importance Spearman improved | Top-k precision improved |
|---|---:|---:|---:|
| `layer2.1.bn1` | +0.4043 | 1/3 target seeds | 3/3 target seeds |
| `layer3.1.bn1` | +0.3763 | 2/3 target seeds | 3/3 target seeds |
| `layer4.1.bn1` | +0.4591 | 1/3 target seeds | 1/3 target seeds |

The alignment consistently increased activation correspondence in all nine comparisons. The effects on importance prediction were inconsistent. Mapping agreement also improved as more feature examples were included: median stability was approximately 78.9% at 1,250 examples and 85.8% at 1,875 examples, measured as subset consistency on the existing feature set.

Bootstrap and random-mapping analyses were diagnostic rather than definitive population-level significance tests. The three target seeds are the principal replication units; the nine layer comparisons are not independent.

## 9. MetaTransfer v0.1 — Disjoint Channel-Importance Transfer

**Status: Completed; preliminary directional evidence, not a reliable general transfer claim.**

### Research question

Does source-to-target channel alignment help use a source model's measured channel-importance ranking to find important channels in another independently trained model?

### Protocol

- **Source:** CIFAR-adapted ResNet-18, seed 42.
- **Targets:** independently trained checkpoints with seeds 123, 456 and 789.
- **Layers:** `layer2.1.bn1` (128 channels), `layer3.1.bn1` (256 channels), and `layer4.1.bn1` (512 channels).
- **Feature signatures:** 2,500 validation images/model, reused from v0.6.
- **Source importance ranking:** 1,250 images, measured by single-channel ablation on source seed 42.
- **Target evaluation:** a separate 1,250 images, with channel effects recomputed for each target. Feature, source-scoring and target-evaluation subsets were pairwise disjoint.
- **Controls:** same-index transfer and 1,000 random mappings per target-layer comparison.
- **Total new ablations:** 3,584 (896 for each of four models).
- **Runtime:** 2,008.8 seconds (approximately 33.5 minutes).
- **Official CIFAR-10 test set:** not loaded.
- **Reports:** `reports/metatransfer_v01/transfer_report.json` and `reports/metatransfer_v01/transfer_report.md`.

### Primary result — Top-10% precision

Top-10% precision measures how many of the target's genuinely most important channels are recovered by transferring the source ranking. Results are averaged over nine target-layer comparisons.

| Method | Mean Top-10% precision |
|---|---:|
| Same-index transfer | 8.12% |
| Random-mapping control mean | 10.25% |
| Activation-aligned transfer | **11.75%** |

Aligned transfer improved mean precision by **3.63 percentage points** over same-index transfer and **1.50 percentage points** over the random-mapping mean. It beat same-index transfer in 6/9 comparisons, tied once, and beat the random-mapping mean in 6/9 comparisons. All three target seeds had a positive layer-averaged difference versus same-index transfer; however, the average aligned precision for seed 789 remained below its random-mapping mean.

![MetaTransfer precision by target and layer](assets/metatransfer_precision_by_layer.png)

### Layer-by-layer primary results

| Target | Layer | Same-index P@10% | Aligned P@10% | Random-map mean P@10% |
|---|---|---:|---:|---:|
| seed 123 | `layer2.1.bn1` | 7.69% | **15.38%** | 10.45% |
| seed 123 | `layer3.1.bn1` | 7.69% | **11.54%** | 10.25% |
| seed 123 | `layer4.1.bn1` | **15.38%** | 11.54% | 10.17% |
| seed 456 | `layer2.1.bn1` | 7.69% | **15.38%** | 10.32% |
| seed 456 | `layer3.1.bn1` | 11.54% | **15.38%** | 10.21% |
| seed 456 | `layer4.1.bn1` | **11.54%** | 9.62% | 10.24% |
| seed 789 | `layer2.1.bn1` | 7.69% | 7.69% | 10.33% |
| seed 789 | `layer3.1.bn1` | 0.00% | 7.69% | 10.03% |
| seed 789 | `layer4.1.bn1` | 3.85% | **11.54%** | 10.26% |

### Interpretation

This is **preliminary evidence that activation alignment can help transfer channel-importance rankings**, relative to same-index transfer. It is not strong evidence that aligned transfer consistently beats chance: individual randomized mappings often perform similarly, gains vary by layer, and randomization reference probabilities are not stand-alone population significance tests. Signed-effect transfer is a secondary exploratory metric and was weak/inconsistent; near-zero ablation effects can have unstable signs.

The validation partition was previously used for checkpoint selection in the project, so the evaluation subset is disjoint within this experiment but is not a pristine external benchmark. Results are limited to one task and one architecture family, with three target checkpoints. Single-channel ablation does not capture interactions among channels.

### Research-flow diagram

The current pipeline transfers an importance **ranking** for evaluation; it does not copy source weights or modify target parameters.

![Brahma MetaCore research pipeline](assets/metacore_research_pipeline.png)

```mermaid
flowchart TD
    A[Source model: seed 42] --> B[Measure source channel importance]
    C[Source and target activation signatures] --> D[Activation-based channel alignment]
    B --> E[Map source importance ranking]
    D --> E
    E --> F[Target: seeds 123, 456, 789]
    F --> G[Evaluate on disjoint target images]
    H[Same-index baseline] --> G
    I[Random-mapping controls] --> G
    G --> J[Top-10% precision, rank metrics, uncertainty]
```

### What this milestone does and does not establish

**Supported within this setup:** activation-based matching finds more-correlated channel patterns than same-index matching, and the matched ranking produced a modest mean improvement over same-index channel-importance transfer.

**Not established:** consistent superiority over random mappings; semantic equivalence of matched channels; causal understanding of whole-network function; useful weight/parameter transfer; automatic architecture construction; adding knowledge while retaining capabilities; or reduced training data/compute requirements.

## Current Research Status

**Current stage: model observation, intervention analysis, robust activation alignment, and preliminary importance transfer.**

### Established within the current experimental setup

- A working neural-network inspection and activation-capture pipeline, with automated tests.
- A reproducible CIFAR-10 training and validation workflow.
- A systematic single-channel ablation method across 896 channels in three ResNet-18 layers.
- Initial tests of predictors for channel-ablation effects; the tested Random Forest did not beat the best single-feature heuristic in the v0.4 pilot.
- Cross-model importance prediction tests on independently trained checkpoints, with no reliable advantage over simple heuristics demonstrated in v0.5.
- Activation-based channel alignment across four independently trained CIFAR-adapted ResNet-18 checkpoints.
- v0.7 validation showing stronger matched activation correlation than same-index correspondence across all nine target-layer cases.
- MetaTransfer v0.1 showing mean Top-10% precision of 11.75% for aligned ranking transfer, compared with 8.12% for same-index transfer and 10.25% for the random-mapping mean.

### Not yet demonstrated

- Consistent superiority of channel-importance transfer over random mappings or statistically reliable improvement across a larger number of independent target models.
- Generalization across architectures and datasets.
- A practical improvement from MetaCore-guided parameter transfer or actual target-model modification.
- Construction of a useful target model from a model map.
- Reliable addition of information or capabilities while preserving prior performance.
- Reduced data or compute requirements relative to conventional baselines.
- Recursive improvement of the model-construction process.

## Next Milestone: MetaTransfer v0.2 — Confirmatory Replication

**Status: Recommended next experiment; not yet completed.**

The next objective is to test whether the v0.1 advantage survives additional independently trained target models and evaluation data that were not used for checkpoint selection or method development.

1. Train at least five additional independently initialized target checkpoints under a fixed, recorded protocol.
2. Reserve a genuinely independent evaluation set before tuning the method; do not treat the previously reused validation partition as pristine.
3. Compare activation-aligned transfer with same-index transfer, random mappings, and simple activation/weight-norm ranking baselines.
4. Record per-example intervention effects where feasible, enabling paired uncertainty analysis rather than relying only on aggregate channel labels.
5. Predefine a practical success criterion and analyze outcomes at the model/seed level, reporting negative results as well as positive ones.
6. Keep all v0.1 reports and caches immutable; write v0.2 artifacts into a new versioned directory.

**Success criterion:** aligned transfer should beat both same-index and random controls consistently across independent targets, with uncertainty analysis supporting a robust advantage. This would still validate a narrow transfer method—not general model construction or reduced pretraining.

## Longer-Term Research Phases

### Phase 1 — Map and predict model behavior

**Status: In progress.**

Inspect architectures, compare initial and trained parameters, trace activations, perform interventions, align representations, and validate predictions across independently trained models.

### Phase 2 — Transfer information and construct a target model

**Status: Planned; not demonstrated.**

MetaTransfer v0.1 tests transfer of channel-importance rankings only. Actual target initialization, component/parameter transfer, or model construction remains future work and must be compared with conventional methods under matched conditions.

### Phase 3 — Edit models to add information or capabilities

**Status: Future research.**

Investigate adding new task capabilities or information while retaining prior performance. Evaluate new-task performance, retention, regressions, and compute/data costs.

### Phase 4 — Recursive model construction

**Status: Future research.**

Investigate systems that map models, propose construction or editing operations, evaluate targets, and improve their strategies from measured outcomes. This depends on earlier stages becoming reliable.

## Research Evaluation Principles

All major claims should be tested using controlled experiments.

- **Predictive validity:** Can the system forecast measurable outcomes before intervention or training?

- **Generalization:** Do findings hold across new checkpoints, random initializations, datasets, and architectures?

- **Causal evidence:** Do controlled interventions support claims about component contributions?

- **Alignment validity:** Is the mapping stable, and does it improve downstream outcomes over appropriate controls?

- **Construction benefit:** Does MetaCore-guided construction outperform conventional baselines?

- **Knowledge retention:** When editing a model, does it gain the intended capability without unacceptable loss of existing capabilities?

- **Training efficiency:** Does the approach reduce data or compute while preserving or improving target performance?

- **Statistical reliability:** Are results reproducible, with uncertainty and limitations reported?

- **Independent verification:** Can other researchers reproduce the experiments?

## Reproducibility

Experiments should record:

- Model architecture and parameter count.

- Dataset, preprocessing, and data-split details.

- Random seeds and experiment configurations.

- Training procedures and optimization settings.

- Evaluation metrics and intervention or editing results.

- Software versions, hardware, and CUDA environment.

- Model checkpoints and machine-readable experiment reports.

The project uses PyTorch and torchvision. Machine-readable reports and cached experiment outputs should be versioned by milestone; do not overwrite earlier results when running new experiments.

## Guiding Principle

**Map a model. Learn from its structure and behavior. Use that information to construct or modify another model. Measure whether the result is better.**

Brahma MetaCore investigates whether information extracted from existing AI models can become a practical resource for building and modifying future models—not merely analyzing or training them.

The main evidence figures in `assets/` are generated from the machine-readable v0.6 and MetaTransfer v0.1 reports. They visualize the measured correlation and precision values in this document; they are not independent experiments.

The ambition is broad; the evidence must be earned one experiment at a time.

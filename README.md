# Brahma MetaCore
## The Intelligence That Builds Intelligence.

Brahma MetaCore is an experimental AI research project investigating how neural network architectures, parameters, weights, activations, and internal computational mechanisms relate to model capabilities.

The long-term goal is to develop systems that can analyze existing neural networks, discover reusable principles of intelligence, and use those principles to assist in constructing new models with potentially reduced dependence on conventional large-scale pretraining.

## Research Objectives

1. **Neural Network Analysis:** Inspect trained models to understand their architectures, parameters, weights, and internal activations.
2. **Causal Component Analysis:** Investigate how individual neurons, channels, layers, and other model components contribute to model behavior through controlled interventions.
3. **Parameter Generation:** Explore algorithms capable of generating, modifying, or initializing useful neural network parameters.
4. **AI-Assisted Model Construction:** Investigate whether insights extracted from existing models can guide the construction of new architectures and parameter configurations.
5. **Reduced Training Requirements:** Experimentally evaluate whether model analysis and AI-assisted construction can reduce the data, computation, and training required to achieve useful capabilities.
6. **Recursive Model Construction:** Explore the possibility of systems that analyze, design, evaluate, and iteratively improve other AI models.

## Initial Experiment: MetaObserver

**Objective:** Establish a reproducible pipeline for inspecting neural networks and collecting information about their internal behavior.

- **Model:** ResNet-18 adapted for CIFAR-10 image classification.
- **Framework:** PyTorch.
- **Hardware:** NVIDIA RTX 3050 Ti Laptop GPU.
- **Initial implementation:** Model architecture inspection, parameter counting, activation tracing, and structured JSON reporting.
- **Validation:** Automated tests covering model construction, output dimensions, metadata collection, activation capture, hook cleanup, and parameter integrity.

The initial MetaObserver implementation successfully inspected an 11.17-million-parameter ResNet-18 and captured activations from 52 modules. Its initial smoke test used a randomly initialized model, validating the instrumentation pipeline rather than demonstrating learned capabilities.

## Experimental Progress

### 1. Neural Network Observation

Implemented model inspection and activation-capture utilities. The system can record architectural metadata and activation statistics, providing a foundation for subsequent experiments.

### 2. Training and Weight Analysis

Trained the adapted ResNet-18 on CIFAR-10 and compared its initial and trained states.

The initial training experiment reached a best observed test accuracy of 92.38% at epoch 19. Because the test set was evaluated repeatedly during training and used to identify the best epoch, this result is exploratory rather than an unbiased final benchmark.

A subsequent validation-based training pipeline achieved 92.26% validation accuracy at epoch 20, establishing a separate validation split for model selection and further analysis.

### 3. Controlled Component Interventions

Implemented channel-level ablation experiments using the validation set. A full scan evaluated 896 interventions across selected batch-normalization layers, including random-control channels.

These experiments provide preliminary measurements of how modifying individual channel outputs affects model loss and accuracy. The results also show why random controls, correctly isolated interventions, and statistical validation are necessary before attributing importance to specific components.

### 4. Predictive Model Understanding — Next Milestone

The next objective is to determine whether MetaCore can **predict the effects of interventions before performing them**.

The system will be evaluated against simple baselines, including random selection and predictions of negligible change. Successful predictions must generalize to held-out examples and ideally replicate across independently trained models.

This milestone is intended to move the project beyond observation toward a measurable ability to learn reusable principles of neural network behavior.

## Research Phases

**Phase 1 — Neural Network Understanding**

Inspect architectures, compare initial and trained parameters, trace activations, perform controlled interventions, and develop experimentally validated models of component behavior.

**Phase 2 — AI-Assisted Model Construction**

Use validated observations to guide parameter initialization, parameter modification, architecture design, and model construction. Compare these approaches against conventional initialization and training baselines under controlled computational and data budgets.

**Phase 3 — Recursive Model Construction**

Investigate systems capable of analyzing models, generating candidate architectures or parameters, evaluating the results, and iteratively improving their construction strategies.

## Research Status

**Current stage: Early experimental development, with working instrumentation, CIFAR-10 training experiments, and preliminary channel-intervention analysis.**

The project has demonstrated that its initial tools can inspect and experimentally modify a trained neural network. It has not yet demonstrated that MetaCore can reliably predict neural network behavior, generate effective model parameters, construct competitive models with substantially less training, or recursively improve its own model-building capabilities.

These remain open research questions that must be tested through controlled experiments.

## Reproducibility

Experiments will record:

- Model architecture and parameter count.
- Dataset, preprocessing, and data-split details.
- Random seeds and experiment configurations.
- Training procedures and optimization settings.
- Evaluation metrics and intervention results.
- Software versions, hardware, and CUDA environment.
- Model checkpoints and machine-readable experiment reports.

The project uses PyTorch and torchvision, with experiments designed to support repeatability, comparison against baselines, and independent verification.

## Guiding Principle

**Observe neural networks. Understand their behavior. Test what can be predicted. Use validated principles to build better models.**

Brahma MetaCore aims to investigate whether understanding how neural networks work can become a practical tool for building intelligence—not merely training it.
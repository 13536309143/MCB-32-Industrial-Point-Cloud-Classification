# MCB-32: A Controlled Study of Industrial Point Cloud Classification

[English](KEY6.md) | [简体中文](KEY6.zh-CN.md) · [Project](../../README.md) · [Recorded artifacts](../../reports/key6/)

**Experiment:** `key6` · **Completed:** 16 October 2026, 08:06 China Standard Time · **Scope:** 6 trained models, 1,800 epochs, 192 evaluations.

## Abstract

We study whole-object industrial component classification using a reorganized Mechanical Components Benchmark collection with 50,531 objects and 32 broad component labels. Six networks—PointNeXt-S/B/L/XL, PointNet and PointNet++—are trained with a common recipe on area-weighted random 2,048-point XYZ inputs. Each selected checkpoint is evaluated on the same 10,107 held-out objects under nine sampling/resolution representations and 23 additional perturbation conditions. PointNeXt-S achieves the highest clean overall accuracy (97.71%) and macro F1 (83.03%), while PointNet++ reaches 97.39% OA with lower batch-one forward latency and stronger light-outlier and occlusion performance. Scaling to PointNeXt-XL increases training cost without improving clean OA. Strong class imbalance and failures under arbitrary rotations/outliers expose limitations hidden by aggregate accuracy. These are single-seed, fixed-recipe results, not an exhaustive optimization study or evidence of real-scan deployment readiness.

## 1. Questions and study boundaries

The completed experiment asks: (1) how do six architectures compare under the same input and training recipe; (2) how well does a trained classifier tolerate a different sampling method or point count for held-out objects; and (3) which synthetic corruptions expose weaknesses? It **does not** identify the best of nine training representations, optimize each model's hyperparameters, measure training-seed variance, or evaluate a new sensor/domain. The `quick` and `full` grids in the repository are prospective plans, not measurements reported here.

Labels describe mechanical component types, combining geometric and functional properties. Classification predicts one component label per complete point cloud. It does not infer materials, defects or per-point part labels. The derivative taxonomy is specific to this preparation and should not be conflated with the original MCB class protocol.

## 2. Data and leakage controls

The original distributed partition has 40,424 training objects and 10,107 test objects. A class-stratified validation subset is drawn **only from training**, producing **36,380 train / 4,044 validation / 10,107 test** objects with split seed `20261013`. Nine point clouds per object share an identity, label and partition. The suite verifies array dtype/shape, finite coordinates, content hashes and consistent ID membership. No object ID crosses these three partitions. This does not prove independence between geometrically similar CAD families; family-level or source-level overlap remains unmeasured.

The nine representations combine area-weighted random, greedy FPS and Poisson-disk-style sample elimination at 2,048/4,096/8,192 points. Inputs contain only contiguous float32 XYZ. Source meshes are centered at their bounding-box center and scaled to unit maximum vertex radius without orientation alignment. Sampled clouds inherit that frame and are not independently normalized. See the [dataset card](../../huggingface/README.md) for candidate generation, nested sets, seeds and limitations.

The data and split bindings recorded for this experiment are:

```text
data_sha256  d559319612d13f3339483c73efcf96e03d1bd45c46a73c6a8a8e886f0d364f8b
split_sha256 a1f4dbb8bf7a612d7e9b76688a116cf97b03b2c3fce890e0e83c7e29676f2a27
```

These fingerprint an ordered data inventory and split definition, respectively; they are not the content hash of a single archive. The exact original fields are versioned in [data_binding.json](../../reports/key6/protocol/data_binding.json).

## 3. Models, training and selection

All runs use seed 42, 300 epochs, AdamW, learning rate 0.001, weight decay 0.05, effective batch 32, 10-epoch warmup and cosine decay to 1e-6. Label smoothing is 0.1, classifier dropout 0.5, and gradient norm clipping 1.0. Four data workers are used. Training augments point order, independent-axis scale in [0.8,1.25], and coordinate translation in [-0.1,0.1]; **no rotation, jitter, outlier or occlusion augmentation** is included. Test corruptions are therefore deliberate distribution shifts. Each epoch drops the incomplete final training batch (28 objects); shuffled ordering changes which objects are omitted.

Checkpoint selection prioritizes validation macro F1, then validation OA, then lower validation loss. Test data never choose the epoch. PointNet adds its feature-transform regularizer during training; validation/test loss is the same smoothed cross entropy used for all networks. The reported NLL is unsmoothed predictive negative log likelihood.

The measured hardware is one **RTX 4070, 12 GB**, with 8 vCPUs and approximately 16 GiB RAM. The runtime is Ubuntu 20.04, Python 3.11.7, PyTorch 2.7.1/cu126 and CUDA toolkit 12.6. All six runs use BF16 mixed precision and physical/effective batch 32. BF16 was selected after FP16 preflight exposed XL gradient overflow. The CUDA operators retain the original upstream kernels; compatibility edits modernize C++ tensor access in build copies.

S uses the upstream ModelNet classification encoder. B/L/XL adapt full upstream S3DIS encoders to object classification using a global pool and classifier. Their original depth/width/downsampling schedules differ from S, so this is **not a pure parameter-count scaling experiment**. PointNet includes input and feature transforms, and PointNet++ is the original SSG encoder. Complete specs/configurations are in [metrics-and-protocol.zip](../../reports/key6/metrics-and-protocol.zip), with upstream code pinned to the commits listed in the root README.

## 4. Clean-set classification

| Model | OA % | mAcc % | Macro F1 % | Smoothed CE | NLL | Best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| PointNeXt-S | 97.71 | 81.70 | 83.03 | 0.7334 | 0.1853 | 282 |
| PointNeXt-B | 96.96 | 80.09 | 80.75 | 0.7756 | 0.2317 | 262 |
| PointNeXt-L | 97.02 | 80.25 | 80.65 | 0.7779 | 0.2295 | 270 |
| PointNeXt-XL | 97.23 | 81.72 | 82.03 | 0.7731 | 0.2280 | 297 |
| PointNet | 95.72 | 74.98 | 75.40 | 0.7972 | 0.2534 | 258 |
| PointNet++ | 97.39 | 80.77 | 81.39 | 0.7395 | 0.1959 | 250 |

S improves over PointNet++ by **0.326 OA percentage points** (33 net additional correct objects) and **1.646 macro-F1 percentage points**. It improves over PointNet by 1.999 OA points. XL ranks second on macro F1 and has the slightly highest mAcc (81.72%, versus S at 81.70%), but does not exceed S's OA or macro F1. B and L perform similarly on clean OA despite different compute costs.

The comparison supports S as a promising baseline **under this recipe**. It does not establish that small networks are universally preferable, that the large encoders were fully optimized, or that any model attains a new state of the art. S also has the lowest clean NLL (0.1853), followed by PointNet++ (0.1959). Label-smoothed CE should not be confused with ordinary NLL or compared directly to losses from unrelated protocols.

![Clean classification and latency](../assets/classification_and_latency.png)

**Figure 1.** Left: sample-weighted OA versus class-balanced metrics. Right: OA against batch-one median GPU forward latency; the truncated vertical axis emphasizes small clean-set differences. OA alone hides substantial rare-class failures.

## 5. Compute and learning dynamics

| Model | Parameters M | Train-only h | Train + val h | Batch-1 p50 ms | Batch-8 objects/s |
| --- | --- | --- | --- | --- | --- |
| PointNeXt-S | 1.371 | 6.11 | 6.39 | 3.531 | 1088.5 |
| PointNeXt-B | 3.879 | 3.48 | 3.64 | 5.572 | 1432.8 |
| PointNeXt-L | 7.176 | 5.65 | 5.91 | 8.653 | 932.5 |
| PointNeXt-XL | 40.840 | 16.88 | 17.58 | 5.727 | 523.3 |
| PointNet | 3.477 | 2.46 | 2.57 | 1.515 | 3206.1 |
| PointNet++ | 1.470 | 6.10 | 6.37 | 2.293 | 1101.0 |

Training-only time is the accumulated epoch training timer. Train+validation sums both epoch timers and excludes checkpoint writes, startup, compilation, data inventory checks, tests and interruption time. The sum across all six models is **42.46 hours**. The suite-status duration for S only covers reused training artifacts and subsequent tests; it must not be interpreted as the complete S training time. Its 300 recorded epochs are all included here, including the 118 epochs preceding a recovered interruption.

XL costs approximately **2.75×** S's train+validation time without improving its clean OA. B trains faster than S despite more parameters; its more aggressive spatial downsampling is one relevant architectural difference. Runtime is also implementation/hardware-dependent, so parameter count is not a throughput predictor.

The inference microbenchmark uses GPU-resident inputs, 10 warmups and 30 timed repetitions with synchronization. Batch-one latency excludes file loading, mesh conversion, transfers and application overhead. Batch-eight throughput uses one object expanded across the batch; it is a short forward-only benchmark, not a realistic mixed-object service workload. PointNet++ is faster than S for a single object (2.293 versus 3.531 ms); PointNet is fastest but loses accuracy and class-balanced performance. Thirty repeats do not characterize long-run latency tails or GPU frequency variation.

![Training and validation curves](../assets/training_curves.png)

**Figure 2.** Recorded 300-epoch curves for all six models. All runs finish their scheduled optimization. S reaches approximately 99.77% training OA, with lower held-out OA, indicating a generalization gap. Validation macro F1 has visible fluctuations because some classes have very few examples. PointNet's training objective includes its transform regularizer, so the training-loss panel has an additional architecture-specific term. The test comparisons use the validation-selected checkpoint rather than the last epoch.

## 6. Sampling and resolution transfer

| Model | Mean OA, other 8 % | Min–max OA, other 8 % |
| --- | --- | --- |
| PointNeXt-S | 97.52 | 97.44–97.66 |
| PointNeXt-B | 96.88 | 96.76–97.04 |
| PointNeXt-L | 96.87 | 96.75–96.97 |
| PointNeXt-XL | 97.00 | 96.82–97.20 |
| PointNet | 95.77 | 95.66–95.96 |
| PointNet++ | 97.46 | 97.38–97.60 |

S stays within 97.44%–97.66% OA on the other eight representations, and PointNet++ stays within 97.38%–97.60%. On this dataset, changing the test sampling scheme or increasing test point count has modest impact on overall accuracy; more input points do not consistently improve results.

S obtains 83.87% macro F1 with Poisson-disk 4,096-point **test inputs**, versus 83.03% on its training representation. This cannot be read as proof that Poisson-disk 4,096-point **training** is best. The experiment keeps trained weights fixed, and extremely small classes can move the macro score. The reported eight-way mean is descriptive and excludes the original representation; the nine test sets share identities and are not independent replications.

![Cross-representation OA](../assets/cross_representation_oa.png)

**Figure 3.** Six trained checkpoints evaluated on all nine paired representations. Each row uses weights learned only on area-random 2,048 points. The narrow color range reveals small shifts, so color contrast must be interpreted with the printed percentages.

## 7. Controlled robustness tests

Each selected checkpoint is tested on all 10,107 held-out identities for every condition. Corruption randomness is deterministic per object. Coordinates are not re-centered/re-scaled after corruption.

| Corruption | Severity 1 | Severity 2 | Severity 3 | Definition |
| --- | --- | --- | --- | --- |
| Jitter | σ=.005 | σ=.01 | σ=.02 | Independent Gaussian XYZ noise, clipped to ±3σ |
| Rotation | 15° | 45° | 90° | Random axis, fixed angle |
| Scale | 1.15 | 1.30 | 1.50 | Isotropic factor or reciprocal, equiprobable |
| Translation | .05 | .10 | .20 | Fixed-length shift in a random direction |
| Dropout | 10% | 30% | 50% | Random removal; repeat retained points to restore N |
| Outliers | 1% | 5% | 10% | Replace points with uniform samples in [-1,1]³ |
| Occlusion | 10% | 30% | 50% | Projection-based planar crop; repeat retained points |

A separate permutation case shuffles rows; the SO(3) case uses a Haar-uniform rotation generated with a normalized random quaternion. Dropout keeps tensor size constant and therefore does not measure inference on a smaller-N cloud. Occlusion approximates a spatial crop, not physically correct depth-camera visibility.

Selected OA scores (%):

| Model | Jitter .005 | Rotation 15° | SO(3) | Outliers 1% | Occlusion 50% | Dropout 50% |
| --- | --- | --- | --- | --- | --- | --- |
| PointNeXt-S | 83.45 | 58.37 | 20.11 | 14.22 | 64.93 | 96.86 |
| PointNeXt-B | 94.20 | 63.31 | 21.98 | 3.20 | 42.62 | 94.87 |
| PointNeXt-L | 94.78 | 62.45 | 19.78 | 2.68 | 39.37 | 94.89 |
| PointNeXt-XL | 94.89 | 65.86 | 19.32 | 6.67 | 39.21 | 95.55 |
| PointNet | 95.06 | 47.99 | 14.43 | 5.90 | 26.06 | 94.56 |
| PointNet++ | 89.67 | 57.16 | 18.54 | 63.35 | 69.14 | 96.97 |

![Robustness OA](../assets/robustness_oa.png)

**Figure 4.** OA across representative clean/corrupted conditions, with a shared 0–100% scale. The full [192-case table](../../reports/key6/tables/all_192_tests.csv) includes all severities and macro metrics.

**Rotation is a major failure mode.** Only 15° rotations reduce clean accuracy substantially, and uniform SO(3) rotations leave every model at approximately 14%–22% OA. The absence of training rotation augmentation is consistent with this sensitivity; the experiment does not isolate all causes or evaluate rotation-invariant architectures.

**Outliers are more damaging than their count suggests.** At 1% replacement, S falls to 14.22% and B/L/XL to roughly 3%–7%; PointNet++ retains 63.35%. However, even PointNet++ falls to 7.10% at 5% outliers. This is a severe synthetic cube-outlier stress test. It is not equivalent to all real sensors' noise distributions, and relative superiority at one severity does not imply deployment robustness.

**Random missing points and spatially missing regions differ.** S retains 96.86% OA after 50% random dropout but only 64.93% after a 50% crop. PointNet++ reaches 69.14% on that crop, while B/L/XL are around 39%–43%. Redundant retained surface coverage is much easier than losing a contiguous region. The use of repeated padding must be retained when reproducing these numbers.

**The clean winner is not uniformly robust.** With σ=.005 jitter, S scores 83.45%, whereas B/L/XL score about 94%–95% and PointNet reaches 95.06%. PointNet++ outperforms S on light outliers, heavy crop and dropout. A single average corruption score would conceal these trade-offs; no benchmark-standard mCE is claimed.

**Order changes are small but not exactly neutral.** The permutation test changes a small number of outcomes. FPS starting points and neighborhood ordering can introduce order sensitivity. The results do not establish strict permutation invariance. Translation and random dropout are comparatively well tolerated under this normalization/training recipe.

## 8. Class imbalance and error anatomy

The largest class, screws/bolts, represents **32.71%** of the test set. S's 97.71% OA coexists with 81.70% mAcc and 83.03% macro F1; aggregate success is not uniform across component types. Its lowest-F1 classes are:

| Class | Test support | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- |
| flange | 1 | 0.00 | 0.00 | 0.00 |
| valve | 1 | 0.00 | 0.00 | 0.00 |
| spacer | 22 | 50.00 | 13.64 | 21.43 |
| hinge | 12 | 83.33 | 41.67 | 55.56 |
| clamp | 31 | 54.55 | 58.06 | 56.25 |
| collar | 10 | 85.71 | 60.00 | 70.59 |
| hook | 24 | 100.00 | 58.33 | 73.68 |
| castors_and_wheels | 72 | 90.77 | 81.94 | 86.13 |

Flange and valve have only **one test object each**, and all six classifiers miss those objects. Their class scores are unstable and cannot support strong comparisons. Spacer is a more substantive warning: S recalls only 3 of 22 spacers (13.64%); **17 are predicted as washers**. This may reflect geometric/label ambiguity or insufficient small-class data, but mesh-level inspection is needed before assigning a cause. Other errors include pins predicted as screws/bolts (9 objects) and studs (6 objects). Counts and class-normalized error rates should both be considered.

![Per-class F1](../assets/per_class_f1.png)

**Figure 5.** Clean F1 for every class and network. Consistent weak columns show that aggregate OA gains do not solve the long tail.

![Support versus recall](../assets/class_support_and_recall.png)

**Figure 6.** Left: held-out support by class on a logarithmic axis. Right: class recall versus support for each model. The plot describes the imbalance; it does not establish a causal effect of sample count because class difficulty and geometry also vary.

Complete per-class support/precision/recall/F1 is in [per_class_clean.csv](../../reports/key6/tables/per_class_clean.csv); S's off-diagonal errors are in [s_confusions.csv](../../reports/key6/tables/s_confusions.csv). All 32×32 confusion matrices remain in the metric archive.

## 9. Paired comparison and uncertainty

We compare S with each other checkpoint on the same object identities using a two-sided **exact McNemar test** on correctness. Holm correction controls the family of five clean-OA comparisons.

| S vs. | Only S correct | Only other correct | OA difference, pp | Exact p | Holm p |
| --- | --- | --- | --- | --- | --- |
| PointNeXt-B | 120 | 44 | 0.752 | 2.461e-09 | 9.845e-09 |
| PointNeXt-L | 119 | 49 | 0.693 | 6.646e-08 | 1.994e-07 |
| PointNeXt-XL | 96 | 47 | 0.485 | 5.104e-05 | 0.0001021 |
| PointNet | 247 | 45 | 1.999 | 6.275e-35 | 3.137e-34 |
| PointNet++ | 73 | 40 | 0.327 | 0.002456 | 0.002456 |

For S versus PointNet++, 73 objects are correct only for S and 40 only for PointNet++; the net difference is 33 of 10,107. All five adjusted p-values are below .05, **conditional on these fixed checkpoints, this test set and the usual independent-object assumption**. This is not a repeated-training significance result, does not test macro F1, and does not eliminate family-level dependence or test-set adaptation bias. Only one training seed is available. Mean±standard deviation across seeds must not be fabricated from epoch scores or from the nine correlated representations.

The preferable next confirmatory experiment fixes the recipe in advance, adds at least three independent training seeds for selected candidates, and evaluates a separate object-family/source or real-scan split. The current test set should not become a tuning set through repeated model selection based on its corruption scores.

## 10. Integrity, availability and reproducibility

The full local backup contains **2,283 files / 1,413,216,937 bytes**, including both best/final checkpoints, all epoch records and prediction NPZs. Every file matched a server-side SHA-256 inventory. All **192 NPZs / 1,940,544 predictions** were independently checked for held-out ID/label consistency, finite normalized probabilities, argmax labels, confusion matrices, OA, mAcc and macro F1. Recomputed metrics matched recorded values. This does not independently rederive test loss from original logits; the archived scalar loss is the training code's evaluation output.

The GitHub publication contains configurations, full metric JSONs, histories, benchmark repetition timings, protocol and validation evidence in [reports/key6](../../reports/key6/). Checkpoints and probability NPZs remain in the local backup and are **not downloadable from this Git repository**. The data are distributed separately at [GeorgeWJJ/PointNext](https://huggingface.co/datasets/GeorgeWJJ/PointNext) as original-NPY TAR shards with exact split metadata.

The public-only reconstruction command is:

```bash
python -m pip install -r scripts/requirements-release.txt
python scripts/build_publication.py
```

It verifies the archive and recorded confusion-matrix metrics, then rebuilds six PNG/SVG figures using no GPU. Prediction-level checks require the separately retained NPZ backup. CSV metrics use 0–1 ratios; report percentages multiply by 100. Report epochs are one-based, while original epoch files use zero-based indices. Full training instructions and version pins are in the [English README](../../README.md).

An earlier S run exhausted file descriptors because each new validation loader retained persistent workers. The fix reuses persistent workers only for the training loader, releases validation/test workers and increases the process limit. A 50-cycle regression check remained stable; S resumed from its existing 118-epoch checkpoint. This operational recovery was not an additional independently trained seed.

## 11. Conclusions and prioritized follow-up

The completed study supports **PointNeXt-S for the present clean CAD recipe** and **PointNet++ as a strong accuracy/latency and robustness comparator**. It provides no measured reason to prefer XL's extra cost for clean classification under these settings. It also exposes unresolved long-tail categories, arbitrary-pose failures and outlier sensitivity.

A focused follow-up should (1) inspect spacer/washer and other rare-class confusions and improve the training data; (2) compare all nine **training** representations for S at a fixed recipe; (3) repeat selected comparisons across independent training seeds; (4) isolate rotation/noise/outlier augmentation effects without overwriting the baseline; and (5) validate on independent scanned objects and measure end-to-end deployment latency. These follow-ups are recommendations, not completed results.

## References and attribution

The dataset derives from [MCB, Kim et al., ECCV 2020](https://github.com/stnoah1/mcb). Architectures build on [PointNeXt, Qian et al., NeurIPS 2022](https://arxiv.org/abs/2206.04670), [PointNet](https://arxiv.org/abs/1612.00593), and [PointNet++](https://arxiv.org/abs/1706.02413). See [references.bib](../references.bib) and [third-party notices](../../THIRD_PARTY_NOTICES.md). Original CAD rights are separate from this repository's software license.

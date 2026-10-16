# Recorded key6 experiment

[English analysis](../../docs/reports/KEY6.md) · [中文分析](../../docs/reports/KEY6.zh-CN.md)

This directory is a publication snapshot of the completed experiment, not a new
training run. All six models used `area_random_2048`, seed 42, 300 epochs and the
same held-out objects. Raw metric ratios in CSV files are in **[0,1]**; report
tables display percentages. Analysis epoch numbers are one-based.

| Artifact | Contents |
| --- | --- |
| `tables/model_comparison.csv` | Clean scores, training time, parameters and forward latency |
| `tables/all_192_tests.csv` | All six models ×32 evaluation cases |
| `tables/cross_representation_54.csv` | Six trained models ×nine input representations |
| `tables/per_class_clean.csv` | Per-class precision/recall/F1 and support |
| `tables/paired_clean_oa.csv` | Exact paired OA comparisons with Holm correction |
| `protocol/` | Original plan, class map, hashes, environment and completion evidence |
| `metrics-and-protocol.zip` | Per-run configuration, encoder specs, 300-epoch histories, benchmark timings, checkpoint metadata and 192 unabridged metric JSON files |

The archive does not contain pickle checkpoints or per-object probability NPZs.
Those remain in the full local `results/key6/` backup (1,413,216,937 bytes). They
are not advertised as downloadable from this GitHub repository. The audit
record in `protocol/backup_verification.json` documents the local SHA-256 and
prediction-level checks; it is not a claim that all backup files are in Git.

The exact split CSV is included in the Hugging Face dataset preparation under
`metadata/experiment_splits.csv.gz`. The deterministic split procedure is also
implemented in `experiments/mcb/data.py` with split seed `20261013`.

# Data

This directory holds data manifests and samples for the Swedish LFM2.5 project.

## Structure

- `manifests/` — JSONL manifests for CPT and SFT datasets (train/eval splits)
- `samples/` — Small sample files for local testing and smoke tests

## Note

Raw and processed data files are gitignored. Only manifests and small samples are tracked.
All data provenance is recorded in manifest files with per-document source and license fields.

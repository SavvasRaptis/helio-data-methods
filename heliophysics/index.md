---
title: Heliophysics
track: heliophysics
level: foundation
status: draft
module_id: heliophysics-track
implementation: none
---

# Heliophysics

These examples apply statistical and machine-learning methods to space-physics
problems. Gaps, changing cadence, temporal dependence, rare events, physical
baselines, and data provenance are treated as part of the modeling problem.

## Examples

- [Dst Forecasting](applications/dst-forecasting/index.md) builds a
  one-hour-ahead forecast from hourly OMNI data using time-ordered splits,
  gap-safe windows, training-only scaling, and a true persistence baseline.
- [Plasma-Sheet Modeling](research-case-studies/plasma-sheet-modeling/index.ipynb)
  compares saved chronological temperature predictions and conditional
  density maps without presenting model training.
- [SEP Occurrence Forecasting](research-case-studies/sep-occurrence-forecasting/index.md)
  examines class imbalance, sample-level validation, and interpretation while
  making the archive's provenance limits explicit.
- [Coronal-Loop Reconstruction](research-case-studies/coronal-loop-reconstruction/index.md)
  uses corrected, non-overlapping partitions to compare profile
  reconstruction models.

Each example states what its evidence can support and where the available data
or validation design limits the conclusion.

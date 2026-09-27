---
title: Heliophysics
track: heliophysics
level: foundation
status: draft
module_id: heliophysics-track
implementation: none
---

# Heliophysics

These examples apply the methods of the earlier chapters to problems in
space physics. In each, the difficulty lies less in the model than in the
data: gaps and changing cadence, strong dependence between neighbouring
samples, rare events, the need for a physical baseline, and archives whose
provenance limits what can be concluded. Each notebook ends by stating what
its results support.

- [Dst Forecasting](applications/dst-forecasting/index.md): a one-hour-ahead
  forecast of the Dst index from hourly OMNI data, with year-based splits and
  persistence as the baseline to beat.
- [Plasma-Sheet Modeling](research-case-studies/plasma-sheet-modeling/index.ipynb):
  saved predictions of a neural-network plasma-sheet model compared with the
  empirical TM03 model, for ion temperature and for density maps.
- [SEP Occurrence Forecasting](research-case-studies/sep-occurrence-forecasting/index.md):
  a rare-event classification problem, the space-weather verification
  scores, and how the decision threshold changes the forecast.
- [Coronal-Loop Reconstruction](research-case-studies/coronal-loop-reconstruction/index.md):
  recovering loop heights from projected shapes, and how the choice of split
  changes the answer from excellent to useless.

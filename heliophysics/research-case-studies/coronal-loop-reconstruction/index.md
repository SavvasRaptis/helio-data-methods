---
title: Coronal-Loop Reconstruction
track: heliophysics
level: research
status: draft
module_id: coronal-loop-reconstruction
implementation: pytorch-with-keras-alternative
---

# Coronal-Loop Reconstruction

This example is adapted from Chifu and Gafeira (2021),
[*3D Solar Coronal Loop Reconstructions with Machine
Learning*](https://iopscience.iop.org/article/10.3847/2041-8213/abed53).
The saved arrays contain projected \(x\) and \(y\) loop coordinates and the
corresponding \(z\)-coordinate profiles.

The converted example corrects the overlapping split in the legacy notebook:
loops 0–2,999 are used for training, 3,000–3,749 for validation, and
3,750–4,999 for final testing. Both notebooks use an aligned Conv1D model and
show reconstruction metrics, residuals, and matched 3-D traces.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)

Reference: I. Chifu and R. Gafeira (2021), *The Astrophysical Journal
Letters*, 910, L10.

---
title: Coronal-Loop Reconstruction
track: heliophysics
level: research
status: draft
module_id: coronal-loop-reconstruction
implementation: pytorch-with-keras-alternative
---

# Coronal-Loop Reconstruction

EUV images show coronal loops in projection on the plane of the sky; their
height above the surface is not measured directly. This example is adapted
from Chifu and Gafeira (2021),
[*3D Solar Coronal Loop Reconstructions with Machine Learning*](https://iopscience.iop.org/article/10.3847/2041-8213/abed53),
and asks whether a one-dimensional convolutional network can recover the
height profile of a loop from its projected shape and three scalar
descriptors: projected length, footpoint separation, and apex angle.

The 5,000 saved loops are ordered by position, and neighbouring loops are
nearly identical. The notebooks therefore score the same model under three
splits: random loops, which leaks near-copies into the test set; groups of
50 consecutive loops, the honest estimate for regions seen in training; and
one contiguous spatial block, which tests extrapolation to a new region. The
network beats the mean height profile clearly under the first two and fails
under the third.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)

Reference: I. Chifu and R. Gafeira (2021), *The Astrophysical Journal
Letters*, 910, L10.

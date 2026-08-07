---
title: Data Splits and Leakage
track: general
level: foundation
status: placeholder
module_id: data-splits-and-leakage
implementation: framework-neutral
---

# Data Splits and Leakage

A model is trained on known examples but is valuable only if it works on
relevant examples it has not seen. Data splitting creates the evidence for
that claim. A poor split can make a model appear successful even when it has
learned information that will not be available in use.

## Learning objectives

By the end of this chapter, you should be able to:

- explain the distinct jobs of training, validation, and test data;
- fit preprocessing using training information only;
- recognize direct, indirect, grouped, and temporal leakage;
- select a split that represents the intended use of a model.

## Three different jobs

The **training set** is used to estimate model parameters such as weights and
biases. The **validation set** supports development choices: architecture,
regularization, learning rate, number of epochs, and decision thresholds. The
**test set** is reserved for the final estimate of performance after those
choices are fixed.

The test set is not simply data passed to `evaluate`. If its results influence
a development decision, it has become another validation set. A new untouched
test set would then be required for an honest final estimate.

## The split represents a future claim

A random split is reasonable only when samples are sufficiently independent
and the intended future data come from the same process. Other structures need
other partitions:

- **Grouped data:** keep measurements from the same event, object, subject, or
  observing interval together.
- **Time-dependent data:** train on earlier intervals and evaluate on later
  intervals when the use case is forecasting forward in time.
- **Spatially related data:** prevent neighboring or derived samples from
  appearing on both sides of the split.
- **Rare classes:** stratification can stabilize class proportions, but it does
  not override grouping or temporal constraints.

The correct split is the one that imitates the independence and information
boundaries of the intended application.

## What leakage looks like

**Data leakage** occurs when model development uses information unavailable at
prediction time or information from the evaluation partition. It can be
obvious, such as including the target among the inputs, or indirect:

- normalizing with a mean computed from all samples;
- selecting features before splitting;
- imputing missing values with test-set statistics;
- placing augmented versions of one observation in different partitions;
- allowing windows from the same event to cross boundaries;
- selecting a threshold after viewing test results.

The safe order is split first, then fit preprocessing on training data, then
apply the learned transformation unchanged to validation and test data.
Official scikit-learn guidance illustrates why fitting preprocessing before a
split creates optimistic estimates in its
[data-leakage examples](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## The MNIST partition used in this book

MNIST provides 60,000 designated training examples and 10,000 designated test
examples. We preserve the official test set and use a seeded, stratified split
of the original training set:

- 50,000 training samples;
- 10,000 validation samples;
- 10,000 official test samples.

The seed is 42 in both framework implementations. The test set is evaluated
only after the architecture and training procedure are fixed. The original
dataset composition is documented by
[LeCun, Cortes, and Burges](https://yann.lecun.org/exdb/mnist/index.html).

## Split-design checklist

Before training, write down:

1. the unit that must remain independent;
2. the time at which the prediction is made;
3. how repeated or related observations are grouped;
4. which transformations learn from data;
5. which decisions use validation results;
6. when the test set may be inspected.

## Reflection questions

Suppose each physical event produces 100 overlapping time windows. Compare:

1. randomly splitting all windows;
2. assigning complete events to partitions;
3. training on earlier events and testing on later events.

For each split, state the generalization claim it approximates and identify any
path by which information could cross the boundary.

## Common mistakes

- fitting normalization or feature selection before splitting;
- tuning repeatedly against the test set;
- confusing a reproducible random split with a scientifically appropriate one;
- ignoring relationships among samples;
- reporting only one convenient split when the result is split-sensitive.

## Summary

Training, validation, and test sets answer different questions. Their
boundaries must reflect the intended prediction setting, and every learned
preprocessing step belongs inside the training boundary.

## References

- [scikit-learn: cross-validation and held-out evaluation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)

---
title: Model Evaluation
track: general
level: foundation
status: placeholder
module_id: model-evaluation
implementation: framework-neutral
---

# Model Evaluation

A metric compresses model behavior into evidence for a particular question.
No metric is universally best: the useful choice depends on the target,
baseline, class balance, and consequences of different errors.

## Learning objectives

By the end of this chapter, you should be able to:

- read a multiclass confusion matrix;
- calculate and interpret accuracy, precision, recall, and F1;
- distinguish macro from support-weighted summaries;
- select basic regression metrics for different error questions;
- compare a model and baseline on the same held-out data.

## Begin with the baseline

Evaluation is comparative. A classification baseline may always predict the
most frequent training class. A regression baseline may predict the training
mean, median, or—when time ordering makes it meaningful—the latest
observation. Compute the baseline with the same inputs, test cases, and metric
as the learned model.

The baseline gives scale to the score. An accuracy of 90% is poor if a trivial
rule reaches 95%, and potentially informative if a defensible baseline reaches
10%.

## The confusion matrix

For a multiclass problem, entry $C_{ij}$ counts samples whose true class is
$i$ and predicted class is $j$. The diagonal contains correct predictions;
off-diagonal entries show which classes are confused.

For one class treated as positive:

- **true positives** are correctly predicted members of that class;
- **false positives** are other classes incorrectly predicted as that class;
- **false negatives** are members of that class predicted elsewhere;
- **true negatives** are all remaining correct rejections.

These counts connect aggregate scores to concrete failure modes.

## Classification metrics

**Accuracy** is the fraction of all predictions that are correct:

$$
\mathrm{accuracy} = \frac{\text{number correct}}{\text{number evaluated}}.
$$

It is easy to interpret but can conceal failure on a rare class.

**Precision** asks how often a positive prediction is correct, while
**recall** asks how many actual positive cases are found:

$$
\mathrm{precision} = \frac{TP}{TP+FP},
\qquad
\mathrm{recall} = \frac{TP}{TP+FN}.
$$

The F1 score is their harmonic mean:

$$
F_1 = 2\frac{\mathrm{precision}\,\mathrm{recall}}
{\mathrm{precision}+\mathrm{recall}}.
$$

In multiclass evaluation, compute these quantities for every class.
**Macro averaging** gives each class equal weight; **weighted averaging** gives
each class weight proportional to its support. Reporting both exposes whether
large classes dominate the summary. The
[scikit-learn metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
defines these averaging conventions and their implementations.

## Scores, decisions, and thresholds

Many classifiers first produce scores or probabilities and then select a
class. Changing a decision threshold changes false positives and false
negatives. A threshold must be chosen using training or validation data—not
the final test set.

For a first multiclass MNIST model, accuracy, per-class precision/recall/F1,
and the confusion matrix provide an interpretable starting set. Later
rare-event applications will require stronger attention to precision–recall
tradeoffs, calibration, event grouping, and uncertainty.

## Regression metrics

For errors $e_i = \hat{y}_i-y_i$:

- **mean error** exposes signed bias but allows positive and negative errors to
  cancel;
- **mean absolute error (MAE)** reports a typical error magnitude in target
  units;
- **root mean squared error (RMSE)** gives larger errors more influence because
  they are squared before averaging.

Always inspect the error distribution or representative intervals in addition
to a single number. Aggregate metrics can hide systematic failures in
scientifically important regimes.

## A defensible evaluation report

A compact report should state:

1. the untouched evaluation set and its relationship to training data;
2. the baseline;
3. the primary metric and why it fits the question;
4. complementary metrics or diagnostic plots;
5. class counts or target coverage;
6. important failure cases;
7. uncertainty or variation across justified repetitions when available.

## Reflection questions

Consider a classifier that predicts the common class correctly but misses most
rare cases. Explain how that behavior appears in:

- accuracy;
- the rare class's precision and recall;
- macro and weighted F1;
- the confusion matrix.

Then state which metric you would prioritize if missing a rare positive were
more costly than issuing a false alert.

## Common mistakes

- relying on accuracy for an imbalanced problem;
- presenting a metric without the baseline;
- selecting a threshold after inspecting test results;
- comparing models evaluated on different partitions;
- interpreting a high score as evidence of physical causation.

## Summary

Metrics answer specific questions about errors. Use a baseline, preserve the
test boundary, inspect class-level behavior, and connect every reported score
to the scientific or operational decision it is meant to inform.

## References

- [scikit-learn: metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn: classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [scikit-learn: confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---
title: Dst Forecasting
track: heliophysics
level: applied
status: draft
module_id: dst-forecasting
implementation: pytorch-with-keras-alternative
---

# Dst Forecasting

The disturbance storm-time index, Dst, measures the depression of the
horizontal geomagnetic field at low latitudes, produced mainly by the
storm-time ring current. It is a standard summary of geomagnetic storm
strength, and forecasting it from the solar wind is a classic space-weather
problem.

The notebooks forecast hourly Dst one hour ahead from OMNI2 data for
2010-2015. Each input holds the previous three hourly values of solar-wind
speed $V$, the GSM $B_z$ component, field magnitude $|B|$, and Dst. The
network is

```text
12 inputs → 50 ReLU → 30 ReLU → 1 output
```

The years are split in time: 2010-2013 for training, 2014 for validation,
and 2015 for testing. Windows never cross a data gap or a split boundary, and
the input scaling is fitted on the training years only.

The baseline is persistence, which assumes Dst does not change:

$$
\widehat{Dst}(t+1) = Dst(t).
$$

Because Dst varies slowly outside storms, persistence is hard to beat at a
one-hour horizon. The notebooks report MAE, RMSE, and $R^2$ for all hours and
for storm hours separately, the MSE skill score relative to persistence, and
a close look at the St Patrick's Day storm of 17 March 2015.

- [PyTorch notebook](pytorch/demo.ipynb)
- [Keras 3 notebook](keras/demo.ipynb)

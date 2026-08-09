---
title: Dst Forecasting
track: heliophysics
level: applied
status: draft
module_id: dst-forecasting
implementation: pytorch-with-keras-alternative
---

# Dst Forecasting

This example uses hourly OMNI2 data from 2010–2015 to predict Dst one hour
ahead. Each input contains the preceding three hourly values of solar-wind
speed, GSM \(B_z\), average magnetic-field magnitude, and Dst.

Windows are created separately inside the training years (2010–2013),
validation year (2014), and final test year (2015). Scaling is fit using only
the training data. The neural model retains the source architecture:

```text
12 inputs → 50 ReLU → 30 ReLU → 1 linear output
```

The single reference prediction is persistence. For the one-hour forecast,
it assumes that the most recently observed Dst will persist:

$$
\widehat{Dst}(t+1)=Dst(t).
$$

The notebooks report MAE, RMSE, \(R^2\), persistence skill, time traces, and
residual diagnostics for the same 2015 samples.

- [Complete PyTorch workflow](pytorch/demo.ipynb)
- [Keras 3 alternative using the Torch backend](keras/demo.ipynb)

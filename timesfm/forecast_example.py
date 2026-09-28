"""Minimal TimesFM 3.0 forecast (runs on CPU; uses CUDA if available)."""

import numpy as np
import torch
from timesfm import TimesFM3Forecaster

device = "cuda" if torch.cuda.is_available() else "cpu"
forecaster = TimesFM3Forecaster.from_pretrained(
    "google/timesfm-3.0-pytorch", device=device
)

# Two univariate series with different context lengths.
ts1 = np.linspace(0, 1, 100).astype(np.float32)
ts2 = np.sin(np.linspace(0, 24, 72)).astype(np.float32)

outputs = list(
    forecaster.predict_batch([ts1, ts2], horizon=12, return_quantiles=True)
)
for i, out in enumerate(outputs, 1):
  print(f"series {i}: forecast {out.forecast.shape}, quantiles {out.quantiles.shape}")
  print("  median:", np.round(out.forecast, 3))

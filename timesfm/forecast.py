"""Forecast a time series from a CSV with TimesFM.

Usage:
  python timesfm/forecast.py data.csv --horizon 30
  python timesfm/forecast.py data.csv --column leads --date-column day --horizon 12 --model 2.5

Writes <out>/forecast.csv (median + 10/90% range) and <out>/forecast.png, and
prints a short summary.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd


def load_series(path, column=None, date_column=None):
  """Returns (values, dates or None, column name) from a CSV."""
  df = pd.read_csv(path)
  if df.empty:
    sys.exit(f"{path} has no rows.")

  if date_column is None:
    for name in df.columns:
      if not pd.api.types.is_numeric_dtype(df[name]):
        parsed = pd.to_datetime(df[name], errors="coerce")
        if parsed.notna().mean() > 0.9:
          date_column = name
          break
  dates = None
  if date_column is not None:
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]).sort_values(date_column)
    dates = pd.DatetimeIndex(df[date_column])

  if column is None:
    numeric = [c for c in df.select_dtypes("number").columns if c != date_column]
    if not numeric:
      sys.exit(f"No numeric column found in {path}. Columns: {list(df.columns)}")
    column = numeric[-1]
  elif column not in df.columns:
    sys.exit(f"Column {column!r} not in {path}. Columns: {list(df.columns)}")

  values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float32)
  if np.isnan(values).all():
    sys.exit(f"Column {column!r} has no numeric values.")
  # Fill gaps so the model sees an evenly spaced series.
  values = pd.Series(values).interpolate(limit_direction="both").to_numpy(np.float32)
  return values, dates, column


def future_dates(dates, horizon):
  if dates is None or len(dates) < 3:
    return None
  freq = pd.infer_freq(dates)
  if freq is None:
    return None
  return pd.date_range(dates[-1], periods=horizon + 1, freq=freq)[1:]


def run_model(values, horizon, model):
  """Returns (median, p10, p90), each of shape (horizon,)."""
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"
  if model == "3.0":
    from timesfm import TimesFM3Forecaster

    forecaster = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch", device=device
    )
    out = forecaster.predict(values, horizon=horizon, return_quantiles=True)
    return out.forecast, out.quantiles[:, 0], out.quantiles[:, 8]

  import timesfm

  tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      "google/timesfm-2.5-200m-pytorch"
  )
  tfm.compile(
      timesfm.ForecastConfig(
          max_context=1024,
          max_horizon=max(256, horizon),
          normalize_inputs=True,
          use_continuous_quantile_head=True,
          fix_quantile_crossing=True,
      )
  )
  point, quantiles = tfm.forecast(horizon=horizon, inputs=[values])
  # quantiles[..., 0] is the mean; 1..9 are the 10th..90th percentiles.
  return point[0], quantiles[0, :, 1], quantiles[0, :, 9]


def save_outputs(out_dir, values, dates, column, median, p10, p90):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  horizon = len(median)
  fut = future_dates(dates, horizon)
  hist_x = dates if fut is not None else np.arange(len(values))
  fut_x = fut if fut is not None else np.arange(len(values), len(values) + horizon)

  table = pd.DataFrame(
      {
          "step": np.arange(1, horizon + 1),
          "median": np.asarray(median, np.float64),
          "p10": np.asarray(p10, np.float64),
          "p90": np.asarray(p90, np.float64),
      }
  ).round(3)
  if fut is not None:
    table.insert(1, "date", fut.strftime("%Y-%m-%d"))
  out_dir.mkdir(parents=True, exist_ok=True)
  table.to_csv(out_dir / "forecast.csv", index=False)

  fig, ax = plt.subplots(figsize=(12, 4.5))
  ax.plot(hist_x, values, color="#4c72b0", label="history")
  ax.plot(fut_x, median, color="#dd8452", label="forecast (median)")
  ax.fill_between(fut_x, p10, p90, color="#dd8452", alpha=0.25, label="10–90% range")
  ax.set_title(f"TimesFM forecast: {column}")
  ax.legend(loc="upper left")
  ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / "forecast.png", dpi=120)
  return table


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("csv", type=pathlib.Path)
  parser.add_argument("--column", help="column to forecast (default: last numeric)")
  parser.add_argument("--date-column", help="date column (default: auto-detect)")
  parser.add_argument("--horizon", type=int, default=30, help="steps ahead")
  parser.add_argument("--model", choices=["3.0", "2.5"], default="3.0",
                      help="3.0 = best, non-commercial weights; 2.5 = Apache-2.0")
  parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("forecasts"))
  args = parser.parse_args(argv)

  values, dates, column = load_series(args.csv, args.column, args.date_column)
  if len(values) < 16:
    print(f"Warning: only {len(values)} points; forecasts will be unreliable.")
  median, p10, p90 = run_model(values, args.horizon, args.model)
  table = save_outputs(args.out, values, dates, column, median, p10, p90)

  last = values[-1]
  print(f"Column: {column} ({len(values)} points, last value {last:.3f})")
  print(f"Model: TimesFM {args.model}, horizon {args.horizon}")
  print(f"End of horizon: median {median[-1]:.3f} (10–90%: {p10[-1]:.3f}–{p90[-1]:.3f})")
  print(f"Wrote {args.out / 'forecast.csv'} and {args.out / 'forecast.png'}")
  print(table.to_string(index=False))


if __name__ == "__main__":
  main()

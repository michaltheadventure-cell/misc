---
name: forecast
description: Forecast a time series with Google's TimesFM model. Use when the user shares or points to a CSV/spreadsheet of numbers over time (leads, sales, applications, spend, traffic…) or pasted numbers, and asks to forecast, predict, project, or estimate future values ("what will next month look like", "forecast the next 30 days").
---

# Forecast with TimesFM

## 1. Get the data into a CSV
- File attached or in the repo → use it. `.xlsx` → convert with pandas to CSV in the scratchpad.
- Numbers pasted in chat → write them to a CSV in the scratchpad (a `value` column, plus a `date` column if dates were given).
- Rows must be in time order and evenly spaced (daily, weekly, monthly…). Gaps are interpolated by the script; mention it if there were many.

## 2. Pick the settings
- `--column`: the series to forecast. If several numeric columns exist and the user didn't say, ask which one (or forecast each, one run per column).
- `--horizon`: steps ahead. Translate the user's words using the data's frequency ("next month" on daily data = 30). Default 30.
- `--model`: `3.0` (default, most accurate, **non-commercial weights**). Use `2.5` (Apache-2.0) if the user says the forecast is for commercial/production use — and mention the licence once either way.

## 3. Run
```shell
source .venv/bin/activate
python timesfm/forecast.py <file.csv> --column <col> --horizon <n> --out <scratchpad>/forecast
```
If `.venv` is missing, run `.claude/hooks/session-start.sh` first.

If the weight download fails with `403` / `ProxyError` for `huggingface.co`, the environment's network policy blocks it: tell the user to allow `huggingface.co` in the environment's Network access settings (cloud environment menu → Edit), and stop.

## 4. Report back
- Show the chart: send `<out>/forecast.png` to the user (SendUserFile, display `render`), and offer `forecast.csv` as a download.
- In plain language, 3–5 lines: where the series is heading (vs. the last value / recent average), the range at the end of the horizon (10–90%: "8 times in 10 the real value should land here"), and any obvious caveat (short history, big recent jump, gaps).
- A compact table only if the horizon is short (≤ 14 steps); otherwise a few key points (e.g. weekly totals).
- The model only sees this one series' history — it doesn't know about holidays, campaigns or budget changes. Say so when the user's question depends on them.

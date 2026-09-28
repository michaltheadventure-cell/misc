# TimesFM

Google Research's time-series foundation model: https://github.com/google-research/timesfm

## Setup

```shell
python3 -m venv .venv && source .venv/bin/activate
pip install -r timesfm/requirements.txt
python timesfm/forecast_example.py
```

The first run downloads the `google/timesfm-3.0-pytorch` weights from Hugging Face.
Note: TimesFM 3.0 weights are non-commercial / non-production only; 2.5 and older
weights are Apache-2.0 (`timesfm.TimesFM_2p5_200M_torch`, `google/timesfm-2.5-200m-pytorch`).

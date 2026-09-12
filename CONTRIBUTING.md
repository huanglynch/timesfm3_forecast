# Contributing

Issues and small, reviewable pull requests are welcome.

## Before you open a PR

- Do not commit `timesfm_config.json`, API keys, weights, or forecast CSV dumps.
- Keep the GUI's job narrow: one clean series in, a path and a band out.
- A-share ticker rules are load-bearing. If you touch `classify_cn_ticker`, add a comment with the exact input (`000001` vs `000001.SH`) and the expected kind.
- TimesFM 3.0 weights are non-commercial. Do not add code paths that hide that fact.

## Dev loop

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "timesfm[torch]"
TFM_APP=browser python timesfm3_forecast_gui.py
```

Prefer `TFM_APP=browser` while iterating; native window restarts are slower.

## What makes a good issue

- OS, Python version, `ENGINE.label()` (3.0 / 2.5 / not loaded)
- Data source + the exact ticker string you typed
- The log lines around the failure (`timesfm_runtime.log`)
- Whether you allowed a download or used a local cache

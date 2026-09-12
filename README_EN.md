# TimesFM 3 Forecast GUI

[中文](README.md) · English

A zero-shot time-series forecasting desktop app. Give TimesFM a clean price series; get a future path, quantile bands, and a holdout score you can actually read.

**One job only:** fetch → forecast → decide whether the path is trustworthy.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#requirements)
[![NiceGUI](https://img.shields.io/badge/UI-NiceGUI-00b4a6)](#features)
[![TimesFM 3.0](https://img.shields.io/badge/Model-TimesFM%203.0%20%2F%202.5-4285F4?logo=google)](#models--licenses)
[![License](https://img.shields.io/badge/Code-Apache--2.0-brightgreen)](LICENSE)
[![Weights](https://img.shields.io/badge/3.0%20Weights-Non--Commercial-orange)](#models--licenses)

---

## What this is

[TimesFM](https://github.com/google-research/timesfm) is Google Research’s pretrained time-series foundation model. Version 3.0 (~330M parameters) ranks at the top of fev-bench, TIME, and GIFT-Eval among foundation models, but **its weights are non-commercial / non-production**. 2.5 weights remain Apache-2.0.

This repo is not the model. It is a **NiceGUI native window** around it:

- One-click history for US / A-share / HK / index symbols
- Load TimesFM 3.0 locally or from Hugging Face; fall back to 2.5
- Point forecast plus p10 / p50 / p90 bands
- Holdout backtest (MAE / MAPE / RMSE / band coverage)
- CSV export, skins, PyInstaller-friendly paths

> A homemade fear-and-greed panel still lives behind one header button. It is not part of the forecast path.

---

## UI map

```
┌─ TimesFM 3 · Zero-shot forecast ──────────── skin ▾  Load  Help ─┐
│ Queue               │  Source  Ticker  Days  Freq  [Fetch] [Key] │
│  · AAPL_….csv       │  Horizon Context Device Model  Holdout Vol │
│ Recent results      │  [Run forecast]                            │
│                     │  ┌──────────────────────────────────────┐  │
│ Holdout metrics     │  │  History + point + p10–p90 band      │  │
│  MAE / MAPE / cover │  │  (optional) holdout vs actual        │  │
│                     │  └──────────────────────────────────────┘  │
│                     │  Log                                        │
└─────────────────────┴────────────────────────────────────────────┘
```

---

## Features

| Piece | What it does | Notes |
| --- | --- | --- |
| Sources | AKShare / Polygon / Finnhub / FMP | Yahoo removed (rate limits). China → AKShare |
| Ticker rules | `000001` = Ping An Bank; `000001.SH` / `SSEC` = SSE Composite | The sharpest foot-gun in this tool |
| Local files | CSV / Excel / Parquet / JSON | Auto-detects date / close / volume columns |
| Model | Auto 3.0 → 2.5, or pin either | Cached weights in `models/` skip the download prompt |
| Forecast | Point + quantile band | Volume as past-only covariate on 3.0 |
| Holdout | Last `horizon` steps held out | Coverage near 80% is a healthier band than 100% |
| Skins | `timesfm_skins.yaml` / `oma_skins.yaml` | Built-in “paper” theme |
| Packaging | PyInstaller-aware | Never writes into `_MEI` extract dirs |

### Ticker cheat sheet

| You type | Meaning | Source |
| --- | --- | --- |
| `AAPL` `MSFT` `NVDA` `TSLA` | US single names | Polygon / Finnhub / FMP |
| `I:SPX` `I:IXIC` `^GSPC` | US indices | Polygon |
| `600519` | Kweichow Moutai | AKShare |
| `000001` | **Ping An Bank** (not the SSE index) | AKShare |
| `000001.SH` `SSEC` | SSE Composite | AKShare |
| `000300` | CSI 300 | AKShare |
| `00700` `09988` | Tencent / Alibaba-SW | AKShare |

---

## Quick start

### Requirements

- Python 3.10+ (3.11 recommended)
- RAM: ≥ 16 GB comfortable on CPU; GPU optional
- Disk: 3.0 weights ≈ 1 GB+; 2.5 a few hundred MB
- Network: first weight download via Hugging Face; A-shares via Eastmoney (disable proxies that intercept `eastmoney.com`)

### Install

```bash
git clone https://github.com/<YOUR_USER>/timesfm3-forecast-gui.git
cd timesfm3-forecast-gui

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install "timesfm[torch]"
```

### Run

```bash
python timesfm3_forecast_gui.py
```

Native window is the default. Browser only:

```bash
TFM_APP=browser python timesfm3_forecast_gui.py
```

| Env | Default | Meaning |
| --- | --- | --- |
| `TFM_HOST` | `127.0.0.1` | Bind address |
| `TFM_PORT` | first free port | Walks forward if occupied |
| `TFM_APP` | `1` | `0` / `browser` → browser mode |

Writable state lives next to the script (or the EXE), never inside the PyInstaller unpack dir:

```
timesfm_config.json
models/
timesfm_forecast_outputs/
timesfm_runtime.log
```

---

## Workflow

### 1. Load the model

Header → **Load model**.

- Cache present under `models/` for 3.0 or 2.5 → silent load
- No cache → confirm before a ~1 GB+ download
- Preference: **auto 3→2.5**, **force 3.0**, or **force 2.5** (Apache-2.0 weights)

### 2. Get a series

**Fetch:** pick a source, type a ticker, set days/frequency, fill API keys for US vendors, click **Fetch & queue**.

**Local file:** drawer → **Files**. Column names may be English or Chinese (`date`/`日期`, `close`/`收盘`, `volume`/`成交量`).

### 3. Forecast

| Control | Practical range |
| --- | --- |
| Horizon | 20–60 daily steps; 8–26 weekly. Bands widen fast beyond that |
| Context | Default 512. Shorter drops seasonality; longer is slower on CPU |
| Holdout | On. Needs `len(series) > horizon + 16` |
| Volume covariate | Honored on 3.0; ignored on 2.5 |

Outputs: chart, metrics in the drawer, `*_forecast.csv` on disk.

### 4. How to read it

A zero-shot foundation model returns a **conditional median path and a band**, not a promise.

| You see | Safer reading |
| --- | --- |
| Point up, band huge | Mildly bullish median, lots of room both ways — not a price target |
| Holdout MAPE < 5% and coverage 70–90% | Recently well calibrated *on this symbol and frequency* |
| Coverage 100% or < 40% | Band too wide or too tight for risk use |
| Step 60 on a daily forecast | Noise. Look at the first 10–20 steps |

**Not investment advice.**

---

## Models & licenses

| Piece | License | Allowed use |
| --- | --- | --- |
| This GUI | Apache-2.0 | Use, modify, redistribute |
| TimesFM **source** (Google) | Apache-2.0 | Same |
| TimesFM **≤ 2.5 weights** | Apache-2.0 | Commercial use under that license |
| TimesFM **3.0 pretrained weights** | [timesfm-non-commercial-license-v1.0](https://huggingface.co/google/timesfm-3.0-pytorch) | Research / eval / experiment only. No revenue activity, no production end-user systems, no distilling commercial models |

The UI prefers 3.0 for quality. **Pin 2.5 for anything commercial or in production.**

Checkpoints:

- 3.0: [`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch)
- 2.5: [`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) (ICML 2024).  
3.0 post: [Google Research blog](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).

---

## FAQ

**Badge stays “not loaded”** — no local cache. Click Load and allow the download, or copy a snapshot into `models/`.

**AKShare ProxyError** — a system proxy is intercepting Eastmoney. Disable it or bypass that host.

**`000001` became a bank** — bare `000001` is the A-share listing. The index is `000001.SH` / `SSEC`.

**EXE missing `calendar.json` / `safetensors`** — rebuild with collect-all for `akshare` and `safetensors`.

**Process dies after a failed 3.0 import** — a half-imported torch stack can take 2.5 with it. Pin 2.5, or restore a complete 3.0 cache.

**Forecast ignores “this stock’s personality”** — expected. The model learned generic series shapes, not filings. If holdout is poor, shorten the horizon or stop treating it as a trading signal.

---

## Layout

```
timesfm3-forecast-gui/
├── timesfm3_forecast_gui.py
├── requirements.txt
├── LICENSE
├── NOTICE
├── README.md          # Chinese
├── README_EN.md       # this file
├── CONTRIBUTING.md
├── .gitignore
└── docs/
    ├── GITHUB_UPLOAD_GUIDE.md
    └── timesfm_help.md
```

Do not commit: `timesfm_config.json`, `models/`, `timesfm_forecast_outputs/`, `timesfm_runtime.log`.

---

## Acknowledgements

Google Research / TimesFM, NiceGUI, AKShare, Polygon, Finnhub, FMP.

Unaffiliated with Google. Not an investment product.

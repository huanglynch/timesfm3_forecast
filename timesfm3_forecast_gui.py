#!/usr/bin/env python3
"""
TimesFM 3 GUI — NiceGUI 原生窗口版
零样本时序预测 · 美股 / A股 / 港股 / 指数 · 可换肤 · 可打包 EXE

设计原则
- 目的只有一件事：拿到一条干净的价格序列，用 TimesFM 给出未来路径与区间，并让人能判断可信度。
- 界面只保留达成该目的的控件。恐惧贪婪指数等衍生玩具已删除。
- TimesFM 3.0 权重为非商用许可；本工具默认优先 3.0，缺失时回退 2.5。
"""
from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ORIG_STDOUT = getattr(sys, "__stdout__", sys.stdout)
_ORIG_STDERR = getattr(sys, "__stderr__", sys.stderr)

# ── 早期 PyInstaller ────────────────────────────────────────────────────────
import multiprocessing as _mp_early

if getattr(sys, "frozen", False):
    _mp_early.freeze_support()
del _mp_early

import numpy as np
import pandas as pd
import requests
from nicegui import app, native, ui

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


# ===========================================================================
# 路径 / 环境（兼容源码与打包 EXE）
# ===========================================================================
def _app_base() -> Path:
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _writable_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _is_bundle_tmp(path: Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    parts = set(path.parts)
    return "_mei" in s or "_MEI" in str(path) or "_MEIPASS" in parts


def _iter_search_dirs() -> List[Path]:
    """CWD → EXE/脚本旁 → 解包目录。写文件绝不落进 _MEI。"""
    dirs: List[Path] = []
    try:
        dirs.append(Path.cwd())
    except Exception:
        pass
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).parent)
        if hasattr(sys, "_MEIPASS"):
            dirs.append(Path(sys._MEIPASS))
    else:
        dirs.append(Path(__file__).resolve().parent)
    out, seen = [], set()
    for d in dirs:
        try:
            rp = d.resolve()
        except Exception:
            rp = d
        if rp in seen:
            continue
        seen.add(rp)
        out.append(d)
    return out


def find_resource(name: str) -> Optional[Path]:
    for d in _iter_search_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


def writable_dir() -> Path:
    for d in _iter_search_dirs():
        if _is_bundle_tmp(d):
            continue
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".tfm_write_probe"
            probe.write_text("1", encoding="utf-8")
            probe.unlink()
            return d
        except Exception:
            continue
    return Path.cwd()


BASE = _app_base()
ROOT = writable_dir()
CACHE_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "timesfm_forecast_outputs"
_existing_cfg = find_resource("timesfm_config.json")
if _existing_cfg is not None and not _is_bundle_tmp(_existing_cfg.parent):
    CONFIG_PATH = _existing_cfg
else:
    CONFIG_PATH = ROOT / "timesfm_config.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_LOG = ROOT / "timesfm_runtime.log"


def _emit_console(line: str, error: bool = False) -> None:
    stream = _ORIG_STDERR if error else _ORIG_STDOUT
    try:
        stream.write(line + "\n")
        stream.flush()
    except Exception:
        try:
            print(line, file=sys.stderr if error else sys.stdout, flush=True)
        except Exception:
            pass
    try:
        with open(RUNTIME_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def clog(msg: str, level: str = "INFO") -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] [{level}] {msg}"
    _emit_console(line, error=level in ("ERROR", "WARN"))


def _install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        clog("未捕获异常: " + "".join(traceback.format_exception(exc_type, exc, tb)), "ERROR")

    sys.excepthook = _hook

    def _thread_hook(args):
        clog(
            "线程异常: "
            + "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
            "ERROR",
        )

    try:
        threading.excepthook = _thread_hook  # type: ignore[attr-defined]
    except Exception:
        pass


_install_excepthook()
clog(f"启动 ROOT={ROOT} CACHE={CACHE_DIR} CONFIG={CONFIG_PATH}")

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "huggingface" / "hub"))
os.environ.setdefault("HF_ASSETS_CACHE", str(CACHE_DIR / "huggingface" / "assets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
(CACHE_DIR / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)

TFM3_REPO = "google/timesfm-3.0-pytorch"
TFM25_REPO = "google/timesfm-2.5-200m-pytorch"


def _hub_repo_dir(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def find_local_checkpoint(repo: str) -> Optional[Path]:
    """在 models/ 与 HF hub 缓存里找已下载权重，找到就走本地、不问下载。"""
    names = (
        repo.split("/")[-1],
        repo.replace("/", "--"),
        repo.replace("/", "_"),
        _hub_repo_dir(repo),
    )
    roots = [CACHE_DIR, CACHE_DIR / "huggingface" / "hub", CACHE_DIR / "hub"]
    for d in _iter_search_dirs():
        roots.extend([d / "models", d / "models" / "huggingface" / "hub"])
    seen = set()
    for root in roots:
        try:
            rp = root.resolve()
        except Exception:
            rp = root
        if rp in seen or not root.exists():
            continue
        seen.add(rp)
        for name in names:
            direct = root / name
            if (direct / "config.json").is_file() or (direct / "model.safetensors").is_file():
                return direct
            snap = direct / "snapshots"
            if snap.is_dir():
                for child in sorted(snap.iterdir(), reverse=True):
                    if (child / "config.json").is_file() or list(child.glob("*.safetensors")):
                        return child
        # 递归浅搜 snapshots
        try:
            for snap in root.glob(f"**/{_hub_repo_dir(repo)}/snapshots/*"):
                if snap.is_dir() and ((snap / "config.json").is_file() or list(snap.glob("*.safetensors"))):
                    return snap
        except Exception:
            pass
    return None


def describe_cache() -> str:
    p3 = find_local_checkpoint(TFM3_REPO)
    p25 = find_local_checkpoint(TFM25_REPO)
    bits = [
        f"3.0 缓存: {'有 → ' + str(p3) if p3 else '无'}",
        f"2.5 缓存: {'有 → ' + str(p25) if p25 else '无'}",
    ]
    return " · ".join(bits)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    "600519", "000001", "300750", "00700", "09988",
    "I:SPX", "I:IXIC", "000300",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_data_source": "akshare",
    "selected_skin": "paper",
    "device": "cpu",
    "model_preference": "auto",
    "horizon": 30,
    "context_len": 512,
    "holdout_eval": True,
    "use_volume_covariate": False,
    "last_ticker": "600519",
    "last_freq": "日 (D)",
    "drawer_open": True,
    "tickers": list(DEFAULT_TICKERS),
    "finnhub": {"api_key": "", "default_days": 365},
    "polygon": {"api_key": "", "default_days": 90},
    "fmp": {"api_key": "", "default_days": 365},
    "akshare": {"default_days": 365},
}

FREQ_MAP = {"日 (D)": "D", "周 (W)": "W", "月 (M)": "MS"}
SOURCE_LABELS = [
    "AKShare (免费A股/港股/指数)",
    "Polygon.io",
    "Finnhub",
    "Financial Modeling Prep (FMP)",
]


# ===========================================================================
# 配置
# ===========================================================================
def load_config() -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                for k, v in user.items():
                    if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
        except Exception:
            pass
    if not cfg.get("tickers"):
        cfg["tickers"] = list(DEFAULT_TICKERS)
    if str(cfg.get("default_data_source", "")).lower() in ("yahoo", "yfinance"):
        cfg["default_data_source"] = "akshare"
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        CONFIG_PATH.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


CONFIG = load_config()


# ===========================================================================
# 皮肤（兼容 oma_skins.yaml / timesfm_skins.yaml）
# ===========================================================================
_BUILTIN_SKINS = {
    "paper": {
        "id": "paper",
        "name": "印刷纸",
        "blurb": "胶版纸底色",
        "prefer_dark": False,
        "quasar": {
            "primary": "#3e5249",
            "secondary": "#5a5348",
            "accent": "#c24d1d",
            "positive": "#2f6f4e",
            "negative": "#c45c3b",
            "warning": "#b45309",
            "info": "#5a5348",
        },
        "light": {
            "paper": "#f3eee4",
            "paper_2": "#e8e0d2",
            "foam": "#fffcf6",
            "drawer": "#efe8db",
            "ink": "#1b1712",
            "ink_2": "#5a5348",
            "ink_3": "#8a8174",
            "line": "rgba(27, 23, 18, 0.10)",
            "accent": "#c24d1d",
            "glow": "rgba(186, 168, 122, 0.18)",
            "footer": "rgba(255, 252, 246, 0.92)",
            "positive": "#2f6f4e",
            "negative": "#c45c3b",
            "warning": "#b45309",
        },
    }
}


def _skin_search_paths() -> List[Path]:
    names = ("timesfm_skins.yaml", "oma_skins.yaml")
    cands: List[Path] = []
    for d in _iter_search_dirs():
        for name in names:
            cands.append(d / name)
    out, seen = [], set()
    for p in cands:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _normalize_skin(key: str, raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    light = raw.get("light") if isinstance(raw.get("light"), dict) else {}
    dark = raw.get("dark") if isinstance(raw.get("dark"), dict) else dict(light)
    quasar = raw.get("quasar") if isinstance(raw.get("quasar"), dict) else {}
    if not light:
        return None
    return {
        "id": key,
        "name": str(raw.get("name") or key),
        "blurb": str(raw.get("blurb") or ""),
        "prefer_dark": bool(raw.get("prefer_dark", False)),
        "quasar": dict(quasar),
        "light": dict(light),
        "dark": dict(dark),
    }


def load_skins() -> Dict[str, Dict[str, Any]]:
    skins: Dict[str, Dict[str, Any]] = {}
    raw = None
    if _yaml is not None:
        for p in _skin_search_paths():
            if not p.is_file():
                continue
            try:
                raw = _yaml.safe_load(p.read_text(encoding="utf-8"))
                break
            except Exception:
                continue
    src = raw if isinstance(raw, dict) else {}
    block = src.get("skins") if isinstance(src.get("skins"), dict) else src
    if isinstance(block, dict):
        for k, v in block.items():
            if k in ("gui", "ui", "default"):
                continue
            entry = _normalize_skin(str(k).strip(), v)
            if entry:
                skins[entry["id"]] = entry
    if not skins:
        skins = dict(_BUILTIN_SKINS)
    return skins


def skin_tokens(skin: Dict[str, Any]) -> Dict[str, str]:
    base = dict(skin.get("light") or {})
    if skin.get("prefer_dark"):
        base.update(skin.get("dark") or {})
    defaults = _BUILTIN_SKINS["paper"]["light"]
    out = {k: str(v) for k, v in defaults.items()}
    for k, v in base.items():
        if v is not None:
            out[k] = str(v)
    return out


SKINS = load_skins()


# ===========================================================================
# 代码归一化：A股个股 vs 指数 是本工具最容易踩的坑
# 000001 = 平安银行；000001.SH / sh000001 / 上证 = 上证综指
# ===========================================================================
A_INDEX_ALIAS = {
    "SSEC": "000001",
    "SSE": "000001",
    "上证": "000001",
    "上证指数": "000001",
    "上证综指": "000001",
    "SH000001": "000001",
    "000001.SH": "000001",
    "000001.SS": "000001",
    "SZ399001": "399001",
    "399001.SZ": "399001",
    "深证成指": "399001",
    "深成指": "399001",
    "HS300": "000300",
    "CSI300": "000300",
    "沪深300": "000300",
    "000300.SH": "000300",
    "SH000300": "000300",
    "上证50": "000016",
    "000016.SH": "000016",
    "SH000016": "000016",
    "中证500": "000905",
    "000905.SH": "000905",
    "SH000905": "000905",
    "创业板指": "399006",
    "399006.SZ": "399006",
    "SZ399006": "399006",
    "科创50": "000688",
    "000688.SH": "000688",
}
A_INDEX_SINA = {
    "000001": "sh000001",
    "399001": "sz399001",
    "000300": "sh000300",
    "000016": "sh000016",
    "000905": "sh000905",
    "399006": "sz399006",
    "000688": "sh000688",
}
POLYGON_INDEX = {
    "IXIC": "I:IXIC",
    "^IXIC": "I:IXIC",
    "NASDAQ": "I:IXIC",
    "SPX": "I:SPX",
    "^GSPC": "I:SPX",
    "SP500": "I:SPX",
    "NDX": "I:NDX",
    "^NDX": "I:NDX",
    "DJI": "I:DJI",
    "^DJI": "I:DJI",
}
YAHOO_INDEX = {
    "I:SPX": "^GSPC",
    "I:IXIC": "^IXIC",
    "I:NDX": "^NDX",
    "I:DJI": "^DJI",
    "SPX": "^GSPC",
    "IXIC": "^IXIC",
}


def _strip_proxy_env() -> Dict[str, Optional[str]]:
    keys = [
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys if k in os.environ}
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        import requests as _req
        saved["_trust_env"] = getattr(_req.sessions.Session, "trust_env", True)
        _req.sessions.Session.trust_env = False
    except Exception:
        pass
    return saved


def _restore_proxy_env(saved: Dict[str, Optional[str]]) -> None:
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    trust = saved.pop("_trust_env", None)
    if trust is not None:
        try:
            import requests as _req
            _req.sessions.Session.trust_env = bool(trust)
        except Exception:
            pass
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


def _short_err(err: BaseException, limit: int = 220) -> str:
    text = str(err).replace("\n", " ").strip()
    low = text.lower()
    if "proxy" in low or "ProxyError" in type(err).__name__:
        return (
            "东财接口被系统代理拦住（ProxyError）。"
            "请关掉系统/终端代理，或对 eastmoney.com 设直连后重试。"
        )
    if "timed out" in low or "timeout" in low:
        return "请求超时。请检查网络后重试。"
    if "calendar.json" in text:
        return "AKShare 缺交易日历文件。请用最新脚本重打包 EXE。"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _ak_period(freq_code: str) -> str:
    return {"W": "weekly", "MS": "monthly", "M": "monthly"}.get(freq_code, "daily")


def classify_cn_ticker(ticker: str) -> Tuple[str, str, str]:
    """返回 (kind, clean_code, hint)
    kind: a_stock | a_index | hk | unknown
    """
    raw = ticker.strip()
    up = raw.upper().replace(" ", "")
    if up in A_INDEX_ALIAS:
        return "a_index", A_INDEX_ALIAS[up], "A股指数"
    if up.startswith(("SH000", "SZ399", "SH399")):
        digits = "".join(ch for ch in up if ch.isdigit())
        return "a_index", digits.zfill(6)[-6:], "A股指数"
    if up.endswith((".SH", ".SS")):
        code = up.split(".")[0]
        if code in {"000001", "000300", "000016", "000905", "000688"} or code.startswith("000"):
            # 000001.SH 明确是指数；600xxx.SH 是沪市个股
            if code.startswith("000") or code.startswith("399"):
                return "a_index", code.zfill(6), "A股指数"
        return "a_stock", code.zfill(6) if code.isdigit() else code, "A股"
    if up.endswith(".SZ"):
        code = up.split(".")[0]
        if code.startswith("399"):
            return "a_index", code, "A股指数"
        return "a_stock", code.zfill(6) if code.isdigit() else code, "A股"
    if up.endswith(".HK"):
        return "hk", up[:-3].zfill(5), "港股"
    if up.isdigit() and len(up) <= 5:
        return "hk", up.zfill(5), "港股"
    if up.isdigit() and len(up) == 6:
        if up.startswith("399"):
            return "a_index", up, "A股指数"
        # 000001 无后缀按平安银行；000300/000016/000905/000688 按指数
        if up in A_INDEX_SINA and up != "000001":
            return "a_index", up, "A股指数"
        return "a_stock", up, "A股"
    return "unknown", raw, "未识别"


def _pick_close_col(df: pd.DataFrame) -> str:
    for name in ("value", "close", "adj close", "收盘", "收盘价", "Close", "c"):
        if name in df.columns:
            return name
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ("close", "收盘", "value", "price")):
            return col
    return df.columns[-1]


def _pick_date_col(df: pd.DataFrame) -> str:
    for name in ("date", "Date", "日期", "time", "datetime", "时间"):
        if name in df.columns:
            return name
    return df.columns[0]


def _pick_volume_col(df: pd.DataFrame) -> Optional[str]:
    for name in ("volume", "Volume", "成交量", "vol", "v"):
        if name in df.columns:
            return name
    return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_date_col(df)
    val_col = _pick_close_col(df)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
    })
    vol_col = _pick_volume_col(df)
    if vol_col is not None:
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.date
    return out


# ===========================================================================
# 数据源
# ===========================================================================
def _weekday_calendar_json() -> str:
    dates: List[str] = []
    d = datetime(1990, 12, 19)
    end = datetime(2032, 12, 31)
    while d <= end:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return json.dumps(dates)


def _ensure_akshare_assets() -> None:
    """EXE 解包目录常缺 akshare/file_fold/calendar.json。补一份交易日历。"""
    import akshare as ak

    pkg = Path(ak.__file__).resolve().parent
    fold = pkg / "file_fold"
    cal = fold / "calendar.json"
    if cal.exists() and cal.stat().st_size > 100:
        return
    payload = _weekday_calendar_json()
    tried: List[Path] = [cal, ROOT / "akshare_file_fold" / "calendar.json"]
    written: Optional[Path] = None
    for dest in tried:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(payload, encoding="utf-8")
            written = dest
            break
        except OSError:
            continue
    if written is None:
        raise RuntimeError(
            "无法写入 akshare calendar.json。"
            "请用最新 build_timesfm_gui.py 重打包（collect-all akshare）。"
        )
    if written != cal:
        try:
            import akshare.file_fold as ff  # type: ignore
        except Exception:
            ff = None
        clog(f"AKShare 日历写到 {written}（包内只读）", "WARN")
    else:
        clog(f"已补齐 AKShare 日历 {cal}", "STEP")


def fetch_polygon(ticker: str, days: int, api_key: str, freq_code: str = "D") -> pd.DataFrame:
    if not api_key:
        raise ValueError("请先填写 Polygon API Key")
    original = ticker
    up = ticker.upper().strip()
    if up in POLYGON_INDEX:
        ticker = POLYGON_INDEX[up]
    elif up.startswith("^"):
        ticker = "I:" + up[1:]
    end = datetime.now().date()
    start = end - timedelta(days=days)
    timespan = {"H": "hour", "W": "week", "M": "month", "MS": "month"}.get(freq_code, "day")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}"
    r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Polygon 无数据：{original} → {ticker} | {data.get('status')} {data.get('error', '')}")
    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"c": "value", "v": "volume"})
    cols = ["date", "value"] + (["volume"] if "volume" in df.columns else [])
    out = df[cols].copy()
    out["date"] = out["date"].dt.date
    return out.sort_values("date").reset_index(drop=True)


def fetch_finnhub(ticker: str, days: int, api_key: str, freq_code: str = "D") -> pd.DataFrame:
    if not api_key:
        raise ValueError("请先填写 Finnhub API Key")
    try:
        import finnhub
    except ImportError as e:
        raise ImportError("缺少 finnhub-python：pip install finnhub-python") from e
    resolution = {"H": "60", "W": "W", "M": "M", "MS": "M"}.get(freq_code, "D")
    to_ts = int(datetime.now().timestamp())
    from_ts = to_ts - days * 86400
    client = finnhub.Client(api_key=api_key.strip())
    res = client.stock_candles(ticker, resolution, from_ts, to_ts)
    if res.get("s") != "ok":
        raise ValueError(f"Finnhub 状态: {res.get('s')}")
    df = pd.DataFrame({
        "date": pd.to_datetime(res["t"], unit="s"),
        "value": res["c"],
        "volume": res.get("v"),
    })
    df["date"] = df["date"].dt.date
    return df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def fetch_fmp(ticker: str, days: int, api_key: str, freq_code: str = "D") -> pd.DataFrame:
    if not api_key:
        raise ValueError("请先填写 FMP API Key")
    if freq_code not in ("D", "MS"):
        raise ValueError("FMP 目前主要支持日线。周线请改 Polygon / AKShare。")
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    r = requests.get(url, params={"apikey": api_key, "timeseries": days}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "historical" not in data:
        raise ValueError("FMP 返回格式异常")
    df = pd.DataFrame(data["historical"])
    return normalize_ohlcv(df)


def fetch_akshare(ticker: str, days: int, freq_code: str = "D") -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError(
            "缺少 akshare：pip install akshare --upgrade\n"
            "国内可用 -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from e
    _ensure_akshare_assets()

    kind, code, market = classify_cn_ticker(ticker)
    period = _ak_period(freq_code)
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=max(days, 30))).strftime("%Y%m%d")
    saved = _strip_proxy_env()
    last_err: Optional[Exception] = None
    try:
        for attempt in range(3):
            try:
                raw = None
                if kind == "hk":
                    raw = ak.stock_hk_hist(
                        symbol=code, period="daily" if period == "daily" else period,
                        start_date=start_date, end_date=end_date, adjust="qfq",
                    )
                elif kind == "a_index":
                    try:
                        raw = ak.index_zh_a_hist(
                            symbol=code, period=period,
                            start_date=start_date, end_date=end_date,
                        )
                    except Exception:
                        raw = None
                    if raw is None or raw.empty:
                        sina = A_INDEX_SINA.get(code, "sh" + code)
                        raw = ak.stock_zh_index_daily(symbol=sina)
                        if raw is not None and not raw.empty:
                            raw = raw.copy()
                            if "date" in raw.columns:
                                raw["日期"] = pd.to_datetime(raw["date"])
                            if "close" in raw.columns:
                                raw["收盘"] = raw["close"]
                else:
                    # 默认当 A 股个股；000001 → 平安银行
                    if kind == "unknown" and code.isdigit() and len(code) == 6:
                        pass
                    raw = ak.stock_zh_a_hist(
                        symbol=code if code.isdigit() else ticker.strip(),
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    )
                    if (raw is None or raw.empty) and hasattr(ak, "stock_zh_a_hist_tx"):
                        raw = ak.stock_zh_a_hist_tx(
                            symbol=code, start_date=start_date, end_date=end_date, adjust="qfq",
                        )
                if raw is None or raw.empty:
                    raise ValueError(f"AKShare 未返回 {market} {ticker}（规范化 {code}）")
                df = normalize_ohlcv(raw)
                if df.empty:
                    raise ValueError("AKShare 解析后为空")
                return df
            except Exception as e:
                last_err = e
                if attempt < 2:
                    import time as _t
                    _t.sleep(1.6)
        raise last_err or RuntimeError("AKShare 失败")
    finally:
        _restore_proxy_env(saved)


def fetch_market(source: str, ticker: str, days: int, api_key: str, freq_code: str) -> pd.DataFrame:
    s = source.lower()
    if "yahoo" in s or "yfinance" in s:
        raise ValueError("Yahoo Finance 已移除（频频限流）。A股/港股请用 AKShare，美股请用 Polygon / Finnhub / FMP。")
    if "akshare" in s or "a股" in source or "免费a" in s:
        return fetch_akshare(ticker, days, freq_code)
    if "polygon" in s:
        return fetch_polygon(ticker, days, api_key, freq_code)
    if "finnhub" in s:
        return fetch_finnhub(ticker, days, api_key, freq_code)
    if "fmp" in s or "financial" in s:
        return fetch_fmp(ticker, days, api_key, freq_code)
    raise ValueError(f"不支持的数据源: {source}")


def read_local_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(p)
    elif suf in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif suf == ".parquet":
        df = pd.read_parquet(p)
    elif suf == ".json":
        df = pd.read_json(p)
    else:
        raise ValueError(f"不支持的文件类型: {suf}")
    return normalize_ohlcv(df)


def _ensure_safetensors() -> None:
    """PyInstaller / 旧版 huggingface_hub 会出现 NameError: safetensors is not defined。"""
    try:
        import safetensors as _st
        import safetensors.torch as _st_torch  # noqa: F401
    except Exception as e:
        raise ImportError(
            "缺少 safetensors。源码环境: pip install safetensors；"
            "EXE 请用最新 build_timesfm_gui.py 重打包。"
        ) from e
    try:
        import huggingface_hub.hub_mixin as mixin
        if getattr(mixin, "safetensors", None) is None:
            mixin.safetensors = _st
    except Exception:
        pass
    clog("safetensors 已就绪", "STEP")


# ===========================================================================
# TimesFM 3 / 2.5 包装
# ===========================================================================
class ForecastEngine:
    def __init__(self) -> None:
        self.backend: Optional[str] = None  # "3" | "2.5"
        self.model: Any = None
        self.device: str = "cpu"
        self.error: str = ""

    def loaded(self) -> bool:
        return self.model is not None

    def label(self) -> str:
        if self.backend == "3":
            return "TimesFM 3.0"
        if self.backend == "2.5":
            return "TimesFM 2.5"
        return "未加载"

    def load(
        self,
        preference: str = "auto",
        device: str = "cpu",
        allow_download: bool = False,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.device = device
        self.error = ""
        prefer3 = preference in ("auto", "3")
        clog(
            f"加载模型开始 preference={preference} device={device} "
            f"allow_download={allow_download}",
            "STEP",
        )
        if prefer3:
            local3 = find_local_checkpoint(TFM3_REPO)
            clog(f"3.0 本地缓存: {local3 or '无'}", "STEP")
            if local3 is None and not allow_download:
                if preference == "3":
                    raise FileNotFoundError("本地没有 TimesFM 3.0 缓存")
                # auto 且无 3.0 缓存：不要 import timesfm3。
                # EXE 里半途失败的 torch import 会让后续 2.5 也 abort。
                clog("无 3.0 缓存，auto 直接走 2.5", "STEP")
            else:
                try:
                    self._load_v3(device, local3, allow_download)
                    clog(f"TimesFM 3.0 加载完成 ← {local3 or TFM3_REPO}", "STEP")
                    return
                except Exception as e:
                    self.error = f"TimesFM 3 加载失败: {e}"
                    clog(self.error, "ERROR")
                    clog(traceback.format_exc(), "ERROR")
                    if preference == "3":
                        raise
                    clog("3.0 失败，尝试 2.5（若进程随后 abort，请用本目录新 build 脚本重打包）", "WARN")
        local25 = find_local_checkpoint(TFM25_REPO)
        clog(f"2.5 本地缓存: {local25 or '无'}", "STEP")
        if local25 is None and not allow_download:
            raise FileNotFoundError("本地没有 TimesFM 2.5 缓存，且未允许下载")
        self._load_v25(device, local25, allow_download)
        clog(f"TimesFM 2.5 加载完成 ← {local25 or TFM25_REPO}", "STEP")

    def _load_v3(self, device: str, local: Optional[Path], allow_download: bool) -> None:
        _ensure_safetensors()
        from timesfm3 import TimesFM3Forecaster
        src = str(local) if local is not None else TFM3_REPO
        offline = local is not None and not allow_download
        try:
            self.model = TimesFM3Forecaster.from_pretrained(
                src, device=device, local_files_only=offline
            )
        except TypeError:
            self.model = TimesFM3Forecaster.from_pretrained(src, device=device)
        self.backend = "3"

    def _load_v25(self, device: str, local: Optional[Path], allow_download: bool) -> None:
        _ensure_safetensors()
        import torch
        import timesfm
        torch.set_float32_matmul_precision("high")
        src = str(local) if local is not None else TFM25_REPO
        offline = local is not None and not allow_download
        try:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                src, local_files_only=offline
            )
        except Exception:
            if not allow_download:
                raise
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TFM25_REPO)
        model.compile(
            timesfm.ForecastConfig(
                max_context=2048,
                max_horizon=512,
                normalize_inputs=True,
            )
        )
        try:
            if getattr(model, "global_batch_size", 0) <= 0:
                model.global_batch_size = 1
        except Exception:
            pass
        if hasattr(model, "to"):
            try:
                model = model.to("cpu" if device == "cpu" else device)
            except Exception:
                pass
        self.model = model
        self.backend = "2.5"

    def predict(
        self,
        values: np.ndarray,
        horizon: int,
        volume: Optional[np.ndarray] = None,
        use_volume: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 point, p10, p50, p90，长度 = horizon。"""
        x = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.backend == "3":
            kwargs: Dict[str, Any] = {"horizon": int(horizon), "return_quantiles": True}
            if use_volume and volume is not None and len(volume) == len(x):
                cov = np.asarray(volume, dtype=np.float32).reshape(1, -1)
                try:
                    out = self.model.predict(x, past_only_covariates=cov, **kwargs)
                except Exception:
                    out = self.model.predict(x, **kwargs)
            else:
                out = self.model.predict(x, **kwargs)
            point = np.asarray(getattr(out, "forecast", out), dtype=float).reshape(-1)
            q = getattr(out, "quantiles", None)
            if q is None:
                p10 = point * 0.95
                p50 = point
                p90 = point * 1.05
            else:
                q = np.asarray(q, dtype=float)
                if q.ndim == 3:
                    q = q[0]
                # 9 deciles: 0.1 ... 0.9 ，中位数下标 4
                p10 = q[:, 0]
                p50 = q[:, min(4, q.shape[1] - 1)]
                p90 = q[:, min(8, q.shape[1] - 1)]
            n = min(len(point), int(horizon))
            return point[:n], p10[:n], p50[:n], p90[:n]

        point_f, quant_f = self.model.forecast(horizon=int(horizon), inputs=[x])
        point = np.asarray(point_f[0], dtype=float).reshape(-1)
        q = np.asarray(quant_f[0], dtype=float)
        # 2.5: 通常 10 个分位，p10/p50/p90 ≈ 下标 1/5/9
        p10 = q[:, 1] if q.shape[1] > 1 else point * 0.95
        p50 = q[:, 5] if q.shape[1] > 5 else point
        p90 = q[:, 9] if q.shape[1] > 9 else point * 1.05
        n = min(len(point), int(horizon))
        return point[:n], p10[:n], p50[:n], p90[:n]


ENGINE = ForecastEngine()


def metrics_holdout(actual: np.ndarray, point: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(point, dtype=float)[: len(a)]
    a = a[: len(p)]
    if len(a) == 0:
        return {}
    mae = float(np.mean(np.abs(a - p)))
    mape = float(np.mean(np.abs((a - p) / np.clip(np.abs(a), 1e-8, None))) * 100)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    lo = np.asarray(p10, dtype=float)[: len(a)]
    hi = np.asarray(p90, dtype=float)[: len(a)]
    cover = float(np.mean((a >= lo) & (a <= hi)) * 100)
    last = a[-1]
    dir_ok = float(((p[-1] - last) * (a[-1] - (a[0] if len(a) == 1 else a[0])) ) >= 0)
    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "coverage": cover,
        "n": float(len(a)),
    }


# ===========================================================================
# GUI
# ===========================================================================
APP_CSS = """
html, body { height: 100%; overflow: hidden; }
.q-layout, .q-page { background: var(--tfm-paper) !important; color: var(--tfm-ink); }
.q-header { background: var(--tfm-foam) !important; border-color: var(--tfm-line) !important; color: var(--tfm-ink) !important; }
.q-footer { background: var(--tfm-footer) !important; border-color: var(--tfm-line) !important; }
.q-drawer { background: var(--tfm-drawer) !important; border-color: var(--tfm-line) !important; }
.tfm-card { background: var(--tfm-foam) !important; border: 1px solid var(--tfm-line); border-radius: 14px; }
.tfm-muted { color: var(--tfm-ink-2) !important; }
.tfm-title { color: var(--tfm-ink) !important; letter-spacing: .02em; }
.q-field__native, .q-field__input, textarea { color: var(--tfm-ink) !important; }
"""


class TimesFMGui:
    def __init__(self) -> None:
        self.cfg = CONFIG
        self.skins = SKINS
        want = str(self.cfg.get("selected_skin") or "paper")
        self.skin_id = want if want in self.skins else (next(iter(self.skins), "paper"))
        self.file_paths: List[str] = []
        self.results: List[Dict[str, Any]] = []
        self.log_q: queue.Queue = queue.Queue()
        self.ui_q: queue.Queue = queue.Queue()
        self.busy = False
        self.chart = None
        self.log_box = None
        self.queue_box = None
        self.result_box = None
        self.status = None
        self.model_badge = None
        self.metrics_label = None
        self.preview_label = None
        self._build()
        ui.timer(0.12, self._drain_queues)

    # ── 皮肤 ──────────────────────────────────────────────────────────────
    def _apply_skin(self, persist: bool = True) -> None:
        skin = self.skins.get(self.skin_id) or next(iter(self.skins.values()))
        tok = skin_tokens(skin)
        q = skin.get("quasar") or {}
        try:
            ui.colors(
                primary=q.get("primary", tok.get("accent", "#3e5249")),
                secondary=q.get("secondary", "#5a5348"),
                accent=q.get("accent", tok.get("accent", "#c24d1d")),
                positive=q.get("positive", "#2f6f4e"),
                negative=q.get("negative", "#c45c3b"),
                warning=q.get("warning", "#b45309"),
                info=q.get("info", "#5a5348"),
            )
        except Exception:
            pass
        ui.query(":root").style(
            f"--tfm-paper:{tok['paper']};--tfm-paper-2:{tok.get('paper_2', tok['paper'])};"
            f"--tfm-foam:{tok['foam']};--tfm-drawer:{tok['drawer']};"
            f"--tfm-ink:{tok['ink']};--tfm-ink-2:{tok['ink_2']};--tfm-ink-3:{tok['ink_3']};"
            f"--tfm-line:{tok['line']};--tfm-accent:{tok['accent']};"
            f"--tfm-glow:{tok['glow']};--tfm-footer:{tok['footer']};"
        )
        if persist:
            self.cfg["selected_skin"] = self.skin_id
            save_config(self.cfg)

    def _build(self) -> None:
        self._apply_skin(persist=False)
        ui.add_css(APP_CSS)

        with ui.header().classes("items-center justify-between px-4 h-12"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("show_chart").classes("text-xl")
                ui.label("TimesFM 3").classes("text-base font-semibold tfm-title")
                ui.label("零样本时序预测").classes("text-xs tfm-muted")
                self.model_badge = ui.badge("模型未加载", color="grey").props("outline dense")
            with ui.row().classes("items-center gap-2"):
                skin_opts = {k: v.get("name", k) for k, v in self.skins.items()}
                ui.select(
                    options=skin_opts,
                    value=self.skin_id,
                    on_change=self._on_skin,
                ).props("dense borderless options-dense").classes("w-28")
                ui.button("加载模型", icon="download", on_click=self.load_model).props("flat dense")
                ui.button("恐惧贪婪", icon="psychology", on_click=self.show_fear_greed).props("flat dense")
                ui.button("输出目录", icon="folder_open", on_click=self.open_output).props("flat dense")
                ui.button("帮助", icon="help_outline", on_click=self.show_help).props("flat dense")

        with ui.left_drawer(value=True, bordered=True).classes("p-3").props("width=300") as self.drawer:
            ui.label("待预测队列").classes("text-sm font-medium mb-1")
            self.queue_box = ui.column().classes("w-full gap-1")
            with ui.row().classes("w-full gap-1 mt-1"):
                ui.button("文件", icon="attach_file", on_click=self.pick_files).props("flat dense")
                ui.button("清空", icon="delete", on_click=self.clear_queue).props("flat dense")
            ui.separator().classes("my-3")
            ui.label("最近结果").classes("text-sm font-medium mb-1")
            self.result_box = ui.column().classes("w-full gap-1")
            ui.separator().classes("my-3")
            self.metrics_label = ui.markdown("回测指标将显示在这里。").classes("text-xs tfm-muted")

        with ui.column().classes("w-full p-3 gap-3"):
            with ui.card().classes("w-full tfm-card p-3"):
                with ui.row().classes("w-full items-end gap-2 flex-wrap"):
                    src_default = self._source_display(self.cfg.get("default_data_source", "yahoo"))
                    self.source = ui.select(
                        SOURCE_LABELS, value=src_default, label="数据源",
                        on_change=self._on_source,
                    ).classes("w-56").props("dense options-dense")
                    self.ticker = ui.select(
                        options=self.cfg.get("tickers", DEFAULT_TICKERS),
                        value=str(self.cfg.get("last_ticker") or "600519"),
                        label="代码",
                        with_input=True,
                        on_change=lambda e: self._persist_gui(),
                    ).classes("w-36").props("dense")
                    self.days = ui.number(
                        label="天数", value=int(self._default_days(src_default)),
                        min=7, max=4000, format="%.0f",
                    ).classes("w-24").props("dense")
                    self.freq = ui.select(
                        list(FREQ_MAP.keys()),
                        value=str(self.cfg.get("last_freq") or "日 (D)"),
                        label="频率",
                        on_change=lambda e: self._persist_gui(),
                    ).classes("w-28").props("dense options-dense")
                    ui.button("获取并加入", icon="cloud_download", on_click=self.fetch_data).props(
                        "unelevated dense"
                    )
                    ui.button("Key", icon="vpn_key", on_click=self.edit_keys).props("flat dense")

                with ui.row().classes("w-full items-end gap-2 flex-wrap mt-1"):
                    self.horizon = ui.number(
                        label="预测步长", value=int(self.cfg.get("horizon", 30)),
                        min=1, max=512, format="%.0f",
                        on_change=lambda e: self._persist_gui(),
                    ).classes("w-28").props("dense")
                    self.context_len = ui.number(
                        label="上下文长度", value=int(self.cfg.get("context_len", 512)),
                        min=32, max=2048, format="%.0f",
                        on_change=lambda e: self._persist_gui(),
                    ).classes("w-28").props("dense")
                    self.device = ui.select(
                        ["cpu", "cuda"], value=str(self.cfg.get("device", "cpu")), label="设备",
                        on_change=lambda e: self._persist_gui(),
                    ).classes("w-24").props("dense options-dense")
                    self.model_pref = ui.select(
                        {"auto": "自动 3→2.5", "3": "强制 3.0", "2.5": "强制 2.5"},
                        value=str(self.cfg.get("model_preference", "auto")),
                        label="模型",
                        on_change=lambda e: self._persist_gui(),
                    ).classes("w-36").props("dense options-dense")
                    self.holdout = ui.switch(
                        "回测评估", value=bool(self.cfg.get("holdout_eval", True)),
                        on_change=lambda e: self._persist_gui(),
                    ).props("dense")
                    self.use_vol = ui.switch(
                        "成交量协变量", value=bool(self.cfg.get("use_volume_covariate", False)),
                        on_change=lambda e: self._persist_gui(),
                    ).props("dense")
                    self.run_btn = ui.button("开始预测", icon="play_arrow", on_click=self.start_forecast).props(
                        "unelevated dense color=primary"
                    )

                ui.label(
                    "A股：600519 茅台 · 000001 平安银行 · 000001.SH 上证 · 00700 腾讯。"
                    " 中国市场用 AKShare；美股用 Polygon / Finnhub / FMP。"
                ).classes("text-xs tfm-muted mt-1")

            with ui.card().classes("w-full tfm-card p-2"):
                self.preview_label = ui.label("图表区：获取数据或完成预测后显示").classes(
                    "text-xs tfm-muted px-2 break-all"
                )
                if go is not None:
                    self.chart = ui.plotly(self._empty_fig()).classes("w-full").style("min-height: 560px")
                else:
                    self.chart = ui.label("请安装 plotly 以显示交互图表：pip install plotly")

            with ui.card().classes("w-full tfm-card p-2"):
                ui.label("日志").classes("text-xs tfm-muted")
                self.log_box = ui.log(max_lines=400).classes("w-full h-36 text-xs")

        with ui.footer().classes("px-4 py-1"):
            self.status = ui.label(f"就绪 · 输出 {OUTPUT_DIR}").classes("text-xs tfm-muted")

        self._log(f"输出目录: {OUTPUT_DIR}")
        self._log(f"配置: {CONFIG_PATH}")
        self._log(describe_cache())
        self._log("TimesFM 3.0 权重为非商用许可；商业用途请改用 2.5（Apache-2.0）。")
        self._refresh_queue()
        ui.timer(0.8, self._try_autoload_cached_model, once=True)

    def _empty_fig(self):
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            xaxis=dict(title="日期 / 步长"),
            yaxis=dict(title="价格"),
        )
        return fig

    # ── 小工具 ────────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "info") -> None:
        lv = (level or "info").upper()
        if lv in ("INFO", "ERROR", "WARN", "DEBUG", "STEP"):
            pass
        elif "失败" in msg or "错误" in msg or "中断" in msg or msg.startswith("❌"):
            lv = "ERROR"
        elif "警告" in msg or msg.startswith("⚠️"):
            lv = "WARN"
        else:
            lv = "INFO"
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_q.put(f"[{stamp}] {msg}")
        clog(msg, lv)

    def _ui(self, kind: str, payload: Any = None) -> None:
        self.ui_q.put((kind, payload))

    def _drain_queues(self) -> None:
        try:
            while True:
                line = self.log_q.get_nowait()
                if self.log_box:
                    self.log_box.push(line)
        except queue.Empty:
            pass
        try:
            while True:
                kind, payload = self.ui_q.get_nowait()
                if kind == "queue":
                    self._refresh_queue()
                elif kind == "results":
                    self._refresh_results()
                elif kind == "preview" and isinstance(payload, dict):
                    self._preview_series(payload["df"], payload["title"])
                elif kind == "chart" and isinstance(payload, dict):
                    self._render_forecast_chart(payload)
                elif kind == "status":
                    self._set_status(str(payload))
                elif kind == "badge":
                    self._sync_model_badge()
                elif kind == "notify":
                    if isinstance(payload, dict):
                        ui.notify(
                            str(payload.get("text", "")),
                            type=str(payload.get("type") or "warning"),
                            timeout=int(payload.get("timeout") or 8000),
                            close_button=True,
                            position="top",
                        )
                    else:
                        ui.notify(str(payload), type="warning", timeout=8000, close_button=True, position="top")
                elif kind == "alert":
                    self._alert(str(payload.get("title") if isinstance(payload, dict) else "提示"),
                                str(payload.get("text") if isinstance(payload, dict) else payload))
                elif kind == "tickers":
                    try:
                        self.ticker.options = self.cfg.get("tickers", DEFAULT_TICKERS)
                    except Exception:
                        pass
        except queue.Empty:
            pass

    def _set_status(self, text: str) -> None:
        if self.status:
            self.status.set_text(text)

    def _alert(self, title: str, text: str) -> None:
        with ui.dialog() as dlg, ui.card().classes("w-[520px] max-w-[94vw] p-4"):
            ui.label(title).classes("text-base font-semibold")
            ui.label(text).classes("text-sm whitespace-pre-wrap mt-2")
            with ui.row().classes("w-full justify-end mt-3"):
                ui.button("知道了", on_click=dlg.close).props("unelevated")
        dlg.open()

    def _source_display(self, key: str) -> str:
        k = (key or "akshare").lower()
        if k in ("yahoo", "yfinance", "akshare", "a股", "港股"):
            return SOURCE_LABELS[0]
        if k == "polygon":
            return SOURCE_LABELS[1]
        if k == "finnhub":
            return SOURCE_LABELS[2]
        if k in ("fmp", "financial"):
            return SOURCE_LABELS[3]
        return SOURCE_LABELS[0]

    def _source_key(self, display: str) -> str:
        if "Yahoo" in display or "AKShare" in display:
            return "akshare"
        if "Polygon" in display:
            return "polygon"
        if "Finnhub" in display:
            return "finnhub"
        return "fmp"

    def _default_days(self, display: str) -> int:
        key = self._source_key(display)
        return int(self.cfg.get(key, {}).get("default_days", 365))

    def _on_source(self, e) -> None:
        try:
            self.days.value = self._default_days(e.value)
        except Exception:
            pass
        self._persist_gui()

    def _on_skin(self, e) -> None:
        self.skin_id = e.value
        self._apply_skin(persist=True)
        self._persist_gui()
        ui.notify(f"皮肤 → {self.skins.get(self.skin_id, {}).get('name', self.skin_id)}")

    def _persist_gui(self) -> None:
        try:
            self.cfg["selected_skin"] = self.skin_id
            if getattr(self, "source", None):
                self.cfg["default_data_source"] = self._source_key(str(self.source.value))
            if getattr(self, "ticker", None) and self.ticker.value:
                self.cfg["last_ticker"] = str(self.ticker.value).strip().upper()
            if getattr(self, "freq", None) and self.freq.value:
                self.cfg["last_freq"] = str(self.freq.value)
            if getattr(self, "horizon", None) and self.horizon.value:
                self.cfg["horizon"] = int(self.horizon.value)
            if getattr(self, "context_len", None) and self.context_len.value:
                self.cfg["context_len"] = int(self.context_len.value)
            if getattr(self, "device", None) and self.device.value:
                self.cfg["device"] = str(self.device.value)
            if getattr(self, "model_pref", None) and self.model_pref.value:
                self.cfg["model_preference"] = str(self.model_pref.value)
            if getattr(self, "holdout", None):
                self.cfg["holdout_eval"] = bool(self.holdout.value)
            if getattr(self, "use_vol", None):
                self.cfg["use_volume_covariate"] = bool(self.use_vol.value)
            if getattr(self, "days", None) and self.days.value:
                src = self.cfg.get("default_data_source", "yahoo")
                self.cfg.setdefault(src, {})["default_days"] = int(self.days.value)
            save_config(self.cfg)
        except Exception:
            pass

    def _api_key(self, source_display: str) -> str:
        key = self._source_key(source_display)
        return str(self.cfg.get(key, {}).get("api_key", "") or "")

    def _needs_key(self, source_display: str) -> bool:
        return self._source_key(source_display) in ("polygon", "finnhub", "fmp")

    def _refresh_queue(self) -> None:
        if not self.queue_box:
            return
        self.queue_box.clear()
        with self.queue_box:
            if not self.file_paths:
                ui.label("空").classes("text-xs tfm-muted")
                return
            for i, p in enumerate(self.file_paths):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(Path(p).name).classes("text-xs truncate").style("max-width:200px")
                    ui.button(icon="close", on_click=lambda idx=i: self.remove_at(idx)).props(
                        "flat dense round size=sm"
                    )

    def _refresh_results(self) -> None:
        if not self.result_box:
            return
        self.result_box.clear()
        with self.result_box:
            if not self.results:
                ui.label("尚无结果").classes("text-xs tfm-muted")
                return
            for i, r in enumerate(reversed(self.results[-12:])):
                real_i = len(self.results) - 1 - i
                title = str(r.get("title", "结果"))
                short = title if len(title) <= 36 else title[:35] + "…"
                ui.button(
                    short,
                    on_click=lambda idx=real_i: self.show_result(idx),
                ).props("flat dense no-caps").classes("w-full text-left text-xs")
                ui.tooltip(title)

    def remove_at(self, idx: int) -> None:
        if 0 <= idx < len(self.file_paths):
            del self.file_paths[idx]
            self._refresh_queue()

    def clear_queue(self) -> None:
        self.file_paths.clear()
        self._refresh_queue()

    def pick_files(self) -> None:
        files = _native_open_files()
        for f in files or []:
            if f and f not in self.file_paths:
                self.file_paths.append(f)
        self._refresh_queue()
        if files:
            self._log(f"已加入 {len(files)} 个本地文件")

    def open_output(self) -> None:
        _open_path(OUTPUT_DIR)

    def show_fear_greed(self) -> None:
        path = None
        if self.file_paths:
            path = self.file_paths[-1]
        else:
            files = _native_open_files()
            path = files[0] if files else None
        if not path:
            ui.notify("请先获取或选择日线文件", type="warning")
            return
        try:
            df = read_local_table(path)
            dates = pd.to_datetime(df["date"])
            prices = df["value"].astype(float)
            if len(prices) < 30:
                ui.notify("至少需要约 30 个交易日", type="warning")
                return
            window = 20
            returns = prices.pct_change()
            momentum = prices.pct_change(periods=window)
            volatility = returns.rolling(window).std()
            roll_min_mom = momentum.expanding(min_periods=window).min()
            roll_max_mom = momentum.expanding(min_periods=window).max()
            mom_score = 100 * (momentum - roll_min_mom) / (roll_max_mom - roll_min_mom + 1e-9)
            roll_min_vol = volatility.expanding(min_periods=window).min()
            roll_max_vol = volatility.expanding(min_periods=window).max()
            vol_score = 100 * (roll_max_vol - volatility) / (roll_max_vol - roll_min_vol + 1e-9)
            fg = (0.65 * mom_score + 0.35 * vol_score).clip(0, 100)
            fg = fg.rolling(5, min_periods=1).mean()
            valid = fg.notna()
            x = [str(d.date()) for d in dates[valid]]
            y = fg[valid].tolist()
            last = float(y[-1]) if y else 50.0
            if last <= 25:
                mood = "极度恐惧"
            elif last <= 45:
                mood = "恐惧"
            elif last <= 55:
                mood = "中性"
            elif last <= 75:
                mood = "贪婪"
            else:
                mood = "极度贪婪"
            fig = None
            if go is not None:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    subplot_titles=(Path(path).name, "恐惧贪婪指数"))
                fig.add_trace(go.Scatter(x=[str(d.date()) for d in dates], y=prices.tolist(),
                                         name="收盘", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=y, name="恐惧贪婪", mode="lines"), row=2, col=1)
                fig.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.08, line_width=0, row=2, col=1)
                fig.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.08, line_width=0, row=2, col=1)
                fig.update_yaxes(range=[0, 100], row=2, col=1)
                fig.update_layout(height=560, margin=dict(l=40, r=20, t=40, b=30),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h"))
            out = OUTPUT_DIR / f"{Path(path).stem}_fear_greed.csv"
            pd.DataFrame({"date": x, "fear_greed": y}).to_csv(out, index=False, encoding="utf-8-sig")
            with ui.dialog() as dlg, ui.card().classes("w-[860px] max-w-[96vw] p-4"):
                ui.label(f"恐惧贪婪指数 · 当前 {last:.1f}（{mood}）").classes("text-base font-semibold")
                ui.label(
                    "由 20 日动量与波动率的 expanding 归一化合成，不是 CNN 官方指数。"
                ).classes("text-xs tfm-muted")
                if fig is not None:
                    ui.plotly(fig).classes("w-full")
                ui.label(f"已保存 {out.name}").classes("text-xs tfm-muted")
                ui.button("关闭", on_click=dlg.close).props("flat")
            dlg.open()
            self._log(f"恐惧贪婪 {Path(path).name}: {last:.1f} {mood}")
        except Exception as e:
            ui.notify(f"计算失败: {e}", type="negative")
            self._log(f"恐惧贪婪失败: {e}")

    def edit_keys(self) -> None:
        with ui.dialog() as dlg, ui.card().classes("w-[420px] p-4"):
            ui.label("API Key").classes("text-base font-semibold")
            ui.label("只写在本机 timesfm_config.json，不会上传。").classes("text-xs tfm-muted")
            poly = ui.input(
                "Polygon", value=self.cfg.get("polygon", {}).get("api_key", ""),
                password=True, password_toggle_button=True,
            ).classes("w-full")
            finn = ui.input(
                "Finnhub", value=self.cfg.get("finnhub", {}).get("api_key", ""),
                password=True, password_toggle_button=True,
            ).classes("w-full")
            fmp = ui.input(
                "FMP", value=self.cfg.get("fmp", {}).get("api_key", ""),
                password=True, password_toggle_button=True,
            ).classes("w-full")

            def _save():
                self.cfg.setdefault("polygon", {})["api_key"] = (poly.value or "").strip()
                self.cfg.setdefault("finnhub", {})["api_key"] = (finn.value or "").strip()
                self.cfg.setdefault("fmp", {})["api_key"] = (fmp.value or "").strip()
                save_config(self.cfg)
                ui.notify("已保存")
                dlg.close()

            with ui.row().classes("w-full justify-end"):
                ui.button("取消", on_click=dlg.close).props("flat")
                ui.button("保存", on_click=_save).props("unelevated")
        dlg.open()

    def show_help(self) -> None:
        path = find_resource("timesfm_help.md")
        if path is None:
            text = (
                "未找到 timesfm_help.md。\n"
                "请把帮助文件放在 EXE 同目录、当前工作目录或程序目录。"
            )
            src = "内置提示"
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as e:
                text = f"读取帮助失败: {e}"
            src = str(path)
        with ui.dialog() as dlg, ui.card().classes("w-[760px] max-w-[96vw] p-0 overflow-hidden"):
            with ui.row().classes("w-full items-center justify-between px-4 pt-3"):
                ui.label("使用说明").classes("text-base font-semibold")
                ui.label(src).classes("text-xs tfm-muted")
            with ui.scroll_area().classes("w-full").style("height: 70vh"):
                ui.markdown(text).classes("text-sm px-4 pb-4")
            with ui.row().classes("w-full justify-end px-4 pb-3"):
                ui.button("关闭", on_click=dlg.close).props("flat")
        dlg.open()

    # ── 模型 / 获取 / 预测 ───────────────────────────────────────────────
    def _needed_repos(self, pref: str) -> List[str]:
        if pref == "2.5":
            return [TFM25_REPO]
        if pref == "3":
            return [TFM3_REPO]
        return [TFM3_REPO, TFM25_REPO]

    def _missing_repos(self, pref: str) -> List[str]:
        return [r for r in self._needed_repos(pref) if find_local_checkpoint(r) is None]

    def _try_autoload_cached_model(self) -> None:
        if ENGINE.loaded() or self.busy:
            return
        pref = str(self.model_pref.value if self.model_pref else self.cfg.get("model_preference", "auto"))
        if pref == "auto" and find_local_checkpoint(TFM3_REPO) is None and find_local_checkpoint(TFM25_REPO) is None:
            self._log("本地无模型缓存。点击「加载模型」后才会询问是否下载。")
            return
        if pref == "3" and find_local_checkpoint(TFM3_REPO) is None:
            return
        if pref == "2.5" and find_local_checkpoint(TFM25_REPO) is None:
            return
        self._log("发现本地缓存，正在静默加载…")
        self._start_load(allow_download=False)

    def _start_load(self, allow_download: bool) -> None:
        if self.busy:
            ui.notify("正在执行其他任务", type="warning")
            return
        pref = str(self.model_pref.value)
        device = str(self.device.value)
        self.busy = True
        self._set_status("正在加载模型…")
        self._log(f"加载模型 preference={pref} device={device} download={allow_download}")

        def _job():
            try:
                ENGINE.load(preference=pref, device=device, allow_download=allow_download)
                self._log(f"已加载 {ENGINE.label()}")
            except Exception as e:
                self._log(f"加载失败: {e}")
                self._log(traceback.format_exc())
            finally:
                self.busy = False
                self._ui("badge")
                self._ui("status", f"就绪 · {ENGINE.label()}" if ENGINE.loaded() else "加载失败")

        threading.Thread(target=_job, daemon=True).start()

    def load_model(self) -> None:
        pref = str(self.model_pref.value if self.model_pref else self.cfg.get("model_preference", "auto"))
        self.cfg["model_preference"] = pref
        # 已加载但用户改成强制 3.0 / 2.5：允许切换，不要直接 return
        if ENGINE.loaded():
            same = (
                (pref == "3" and ENGINE.backend == "3")
                or (pref == "2.5" and ENGINE.backend == "2.5")
                or (pref == "auto" and ENGINE.backend in ("3", "2.5"))
            )
            missing_now = self._missing_repos(pref)
            if same and not missing_now:
                ui.notify(f"已加载 {ENGINE.label()}，无需重复下载")
                self._log(f"已加载 {ENGINE.label()}，preference={pref} 无需再下")
                return
            if same and missing_now:
                pass
            else:
                self._log(f"切换模型：当前 {ENGINE.label()} → preference={pref}")
                ENGINE.model = None
                ENGINE.backend = ""
        missing = self._missing_repos(pref)
        if not missing:
            self._start_load(allow_download=False)
            return
        names = "、".join(missing)
        with ui.dialog() as dlg, ui.card().classes("w-[480px] p-4"):
            ui.label("本地没有模型缓存").classes("text-base font-semibold")
            ui.markdown(
                f"未在 `models/` 找到：\n\n`{names}`\n\n"
                "首次大约 1GB+，会写入 EXE 旁的 `models/`。"
                "3.0 权重为非商用许可。是否现在下载？"
            ).classes("text-sm")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消", on_click=dlg.close).props("flat")

                def _yes():
                    dlg.close()
                    self._start_load(allow_download=True)

                ui.button("下载并加载", on_click=_yes).props("unelevated")
        dlg.open()

    def _sync_model_badge(self) -> None:
        if self.model_badge:
            if ENGINE.loaded():
                self.model_badge.set_text(ENGINE.label())
                self.model_badge.props("color=positive")
                self._set_status(f"就绪 · {ENGINE.label()}")
            elif self.busy:
                self.model_badge.set_text("加载中")
                self.model_badge.props("color=warning")
            else:
                self.model_badge.set_text("模型未加载")

    def fetch_data(self) -> None:
        source = str(self.source.value)
        ticker = str(self.ticker.value or "").strip().upper()
        if not ticker:
            ui.notify("请输入代码", type="warning")
            return
        days = int(self.days.value or 365)
        freq_code = FREQ_MAP.get(str(self.freq.value), "D")
        if self._needs_key(source) and not self._api_key(source):
            ui.notify("该数据源需要 API Key", type="warning")
            self.edit_keys()
            return
        if self.busy:
            ui.notify("正在执行其他任务", type="warning")
            return
        kind, code, hint = classify_cn_ticker(ticker)
        self._log(f"获取 {source} {ticker} ({hint}/{code}) {days} 天 {freq_code}")
        self.busy = True
        self._set_status("正在获取数据…")
        key = self._api_key(source)

        def _job():
            try:
                df = fetch_market(source, ticker, days, key, freq_code)
                safe_src = (
                    source.replace(" ", "_").replace("(", "").replace(")", "")
                    .replace("/", "_")
                )
                fname = f"{ticker}_{safe_src}_{freq_code}_{datetime.now().strftime('%Y%m%d')}.csv"
                path = OUTPUT_DIR / fname
                df.to_csv(path, index=False, encoding="utf-8-sig")
                self.file_paths.append(str(path))
                tickers = self.cfg.get("tickers", [])
                if ticker not in [t.upper() for t in tickers]:
                    tickers.append(ticker)
                    self.cfg["tickers"] = tickers
                    save_config(self.cfg)
                self._log(f"已保存 {fname} · {len(df)} 条")
                self._ui("notify", {"text": f"已获取 {ticker} · {len(df)} 条", "type": "positive", "timeout": 4000})
                self._ui("preview", {"df": df, "title": f"{ticker} · {hint}"})
                self._ui("queue")
                self._ui("tickers")
            except Exception as e:
                short = _short_err(e)
                self._log(f"获取失败: {e}", "error")
                self._ui("notify", {"text": f"获取失败：{short}", "type": "negative", "timeout": 12000})
                self._ui("alert", {"title": f"获取 {ticker} 失败", "text": short + "\n\n详情见底部日志。"})
                self._ui("status", "获取失败")
            finally:
                self.busy = False
                self._ui("status", "就绪")

        threading.Thread(target=_job, daemon=True).start()

    def start_forecast(self) -> None:
        if not self.file_paths:
            ui.notify("请先获取或选择数据", type="warning")
            return
        if self.busy:
            ui.notify("正在执行其他任务", type="warning")
            return
        horizon = int(self.horizon.value or 30)
        ctx = int(self.context_len.value or 512)
        holdout = bool(self.holdout.value)
        use_vol = bool(self.use_vol.value)
        freq_display = str(self.freq.value)
        freq_code = FREQ_MAP.get(freq_display, "D")
        pref = str(self.model_pref.value)
        device = str(self.device.value)
        self.cfg.update({
            "horizon": horizon,
            "context_len": ctx,
            "holdout_eval": holdout,
            "use_volume_covariate": use_vol,
            "device": device,
            "model_preference": pref,
        })
        save_config(self.cfg)
        paths = list(self.file_paths)
        self.busy = True
        self._set_status("预测中…")

        def _job():
            try:
                if not ENGINE.loaded():
                    missing = [r for r in (
                        [TFM3_REPO, TFM25_REPO] if pref == "auto" else
                        [TFM3_REPO] if pref == "3" else [TFM25_REPO]
                    ) if find_local_checkpoint(r) is None]
                    if missing and pref != "auto":
                        raise FileNotFoundError(
                            "本地无模型缓存。请先点「加载模型」确认下载。"
                        )
                    self._log("模型未加载，尝试本地缓存…")
                    ENGINE.load(preference=pref, device=device, allow_download=False)
                    self._log(f"已加载 {ENGINE.label()}")
                for path in paths:
                    self._forecast_one(path, horizon, ctx, holdout, use_vol, freq_display, freq_code)
            except Exception as e:
                self._log(f"预测中断: {e}")
                self._log(traceback.format_exc())
            finally:
                self.busy = False
                self._ui("badge")
                self._ui("status", "就绪")
                self._ui("results")

        threading.Thread(target=_job, daemon=True).start()

    def _forecast_one(
        self,
        path: str,
        horizon: int,
        ctx: int,
        holdout: bool,
        use_vol: bool,
        freq_display: str,
        freq_code: str,
    ) -> None:
        name = Path(path).stem
        self._log(f"预测 {name}")
        df = read_local_table(path) if not path.endswith(".csv") else normalize_ohlcv(pd.read_csv(path))
        values = df["value"].astype(np.float32).values
        volume = df["volume"].astype(np.float32).values if "volume" in df.columns else None
        dates = list(df["date"])
        if len(values) < 16:
            self._log(f"跳过 {name}：只有 {len(values)} 点，至少 16")
            return
        if len(values) > ctx:
            values = values[-ctx:]
            dates = dates[-ctx:]
            if volume is not None:
                volume = volume[-ctx:]

        holdout_pack = None
        if holdout and len(values) > horizon + 16:
            train, actual = values[:-horizon], values[-horizon:]
            vol_train = volume[:-horizon] if volume is not None else None
            p, lo, mid, hi = ENGINE.predict(train, horizon, vol_train, use_vol)
            n = min(len(actual), len(p))
            m = metrics_holdout(actual[:n], p[:n], lo[:n], hi[:n])
            holdout_pack = {
                "actual": actual[:n],
                "point": p[:n],
                "p10": lo[:n],
                "p50": mid[:n],
                "p90": hi[:n],
                "dates": dates[-horizon:][:n],
                "metrics": m,
            }
            self._log(
                f"回测 {name}: MAE={m.get('mae', 0):.4g} MAPE={m.get('mape', 0):.2f}% "
                f"覆盖率={m.get('coverage', 0):.1f}%"
            )

        point, p10, p50, p90 = ENGINE.predict(values, horizon, volume, use_vol)
        last_date = dates[-1] if dates else datetime.now().date()
        future_dates = _future_dates(last_date, horizon, freq_code)
        records = []
        for i in range(len(point)):
            records.append({
                "frequency": freq_display,
                "frequency_code": freq_code,
                "model": ENGINE.label(),
                "step": i + 1,
                "date": str(future_dates[i] if i < len(future_dates) else ""),
                "point_forecast": float(point[i]),
                "p10": float(p10[i]),
                "p50": float(p50[i]),
                "p90": float(p90[i]),
            })
        out_csv = OUTPUT_DIR / f"{name}_forecast.csv"
        pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")
        if holdout_pack and holdout_pack.get("metrics"):
            (OUTPUT_DIR / f"{name}_holdout_metrics.json").write_text(
                json.dumps(holdout_pack["metrics"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        item = {
            "title": f"{name} · {ENGINE.label()} · {horizon}步",
            "csv": str(out_csv),
            "history_dates": [str(d) for d in dates],
            "history_values": [float(v) for v in values],
            "future_dates": [str(d) for d in future_dates[: len(point)]],
            "point": [float(x) for x in point],
            "p10": [float(x) for x in p10],
            "p50": [float(x) for x in p50],
            "p90": [float(x) for x in p90],
            "holdout": None if not holdout_pack else {
                "dates": [str(d) for d in holdout_pack["dates"]],
                "actual": [float(x) for x in holdout_pack["actual"]],
                "point": [float(x) for x in holdout_pack["point"]],
                "p10": [float(x) for x in holdout_pack["p10"]],
                "p90": [float(x) for x in holdout_pack["p90"]],
                "metrics": holdout_pack["metrics"],
            },
        }
        self.results.append(item)
        self._ui("chart", item)
        self._ui("results")
        self._log(f"完成 {name} → {out_csv.name}")

    def _preview_series(self, df: pd.DataFrame, title: str) -> None:
        if go is None or self.chart is None or not hasattr(self.chart, "figure"):
            return
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[str(d) for d in df["date"]], y=df["value"],
            name="收盘", mode="lines", line=dict(width=2),
        ))
        fig.update_layout(
            title=None,
            margin=dict(l=48, r=16, t=16, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        )
        self.chart.figure = fig
        try:
            self.chart.update()
        except Exception:
            pass
        if self.preview_label:
            self.preview_label.set_text(f"{title} · {len(df)} 条")

    def _render_forecast_chart(self, item: Dict[str, Any]) -> None:
        if go is None or self.chart is None:
            return
        hist_x = item["history_dates"]
        hist_y = item["history_values"]
        fut_x = item["future_dates"]
        # 把最后历史点接到预测起点，线不断
        join_x = [hist_x[-1]] + fut_x if hist_x else fut_x
        join_p = [hist_y[-1]] + item["point"] if hist_y else item["point"]
        join_lo = [hist_y[-1]] + item["p10"] if hist_y else item["p10"]
        join_hi = [hist_y[-1]] + item["p90"] if hist_y else item["p90"]

        rows = 2 if item.get("holdout") else 1
        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=False,
            subplot_titles=("", "回测：预测 vs 真实") if rows == 2 else ("",),
            vertical_spacing=0.16,
            row_heights=[0.64, 0.36] if rows == 2 else [1.0],
        )
        fig.add_trace(go.Scatter(x=hist_x, y=hist_y, name="历史", mode="lines", line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=join_x + join_x[::-1],
            y=join_hi + join_lo[::-1],
            fill="toself", name="p10–p90",
            line=dict(width=0), opacity=0.25, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=join_x, y=join_p, name="点预测", mode="lines", line=dict(width=2, dash="dash")), row=1, col=1)
        if rows == 2:
            h = item["holdout"]
            fig.add_trace(go.Scatter(x=h["dates"], y=h["actual"], name="回测真实", mode="lines+markers"), row=2, col=1)
            fig.add_trace(go.Scatter(x=h["dates"], y=h["point"], name="回测预测", mode="lines+markers"), row=2, col=1)
            m = h.get("metrics") or {}
            if self.metrics_label:
                self.metrics_label.set_content(
                    f"**回测** n={int(m.get('n', 0))}  \n"
                    f"MAE `{m.get('mae', 0):.4g}` · "
                    f"MAPE `{m.get('mape', 0):.2f}%` · "
                    f"RMSE `{m.get('rmse', 0):.4g}`  \n"
                    f"p10–p90 覆盖率 `{m.get('coverage', 0):.1f}%`"
                )
        fig.update_layout(
            title=None,
            margin=dict(l=48, r=16, t=28, b=36),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                xanchor="left",
                bgcolor="rgba(255,255,255,0.55)",
                borderwidth=0,
                font=dict(size=12),
                itemsizing="constant",
            ),
            height=560 if rows == 2 else 420,
            hovermode="x unified",
        )
        fig.update_annotations(font=dict(size=13))
        self.chart.figure = fig
        try:
            self.chart.update()
        except Exception:
            pass
        if self.preview_label:
            self.preview_label.set_text(item["title"])

    def show_result(self, idx: int) -> None:
        if 0 <= idx < len(self.results):
            self._render_forecast_chart(self.results[idx])
            csv = self.results[idx].get("csv")
            if csv and Path(csv).exists():
                self._log(f"查看 {Path(csv).name}")


def _future_dates(last, horizon: int, freq_code: str) -> List:
    last = pd.to_datetime(last)
    if freq_code == "W":
        return [(last + pd.Timedelta(weeks=i + 1)).date() for i in range(horizon)]
    if freq_code in ("M", "MS"):
        return [(last + pd.DateOffset(months=i + 1)).date() for i in range(horizon)]
    out = []
    d = last
    while len(out) < horizon:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.date())
    return out


def _open_path(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        ui.notify(str(e), type="negative")


def _native_open_files() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        files = filedialog.askopenfilenames(
            title="选择时序文件",
            filetypes=[
                ("表格", "*.csv *.xlsx *.xls *.parquet *.json"),
                ("所有", "*.*"),
            ],
        )
        root.destroy()
        return list(files)
    except Exception:
        return []


# ===========================================================================
# 入口
# ===========================================================================
_GUI: Optional[TimesFMGui] = None


def _root() -> None:
    global _GUI
    if _GUI is None:
        _GUI = TimesFMGui()
        return
    try:
        _GUI._build()
    except Exception:
        _GUI = TimesFMGui()


def _port_free(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, int(port)))
        return True
    except OSError:
        return False


def pick_free_port(host: str, preferred: Optional[int] = None) -> int:
    """preferred 被占用则自动往后找；全部失败则让系统分配。"""
    candidates: List[int] = []
    if preferred and 1 <= int(preferred) <= 65535:
        candidates.append(int(preferred))
    try:
        auto = int(native.find_open_port())
        candidates.append(auto)
    except Exception as e:
        clog(f"native.find_open_port 失败: {e}", "WARN")
    candidates.extend(range(8080, 8220))
    candidates.extend(range(18700, 18820))
    seen = set()
    for port in candidates:
        if port in seen or port < 1 or port > 65535:
            continue
        seen.add(port)
        if _port_free(host, port):
            if preferred and port != preferred:
                clog(f"端口 {preferred} 已被占用，自动改用 {port}", "WARN")
            else:
                clog(f"使用端口 {port}", "STEP")
            return port
        clog(f"端口 {port} 占用，继续尝试", "WARN")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        bind_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        sock.bind((bind_host, 0))
        port = int(sock.getsockname()[1])
    clog(f"预置端口均不可用，系统分配 {port}", "WARN")
    return port


def main() -> None:
    host = (os.environ.get("TFM_HOST") or "127.0.0.1").strip()
    env_port = (os.environ.get("TFM_PORT") or "").strip()
    preferred = int(env_port) if env_port.isdigit() else None
    port = pick_free_port(host, preferred)
    use_native = (os.environ.get("TFM_APP") or "1").strip().lower() not in (
        "0", "false", "no", "browser",
    )
    kwargs: Dict[str, Any] = dict(
        title="TimesFM 3 · Forecast",
        host=host,
        port=port,
        reload=False,
        favicon="📈",
        dark=False,
    )
    if use_native:
        try:
            app.native.window_args["confirm_close"] = True
            try:
                import inspect
                import webview as _wv
                params = inspect.signature(_wv.create_window).parameters
                if "maximized" in params:
                    app.native.window_args["maximized"] = True
            except Exception:
                pass
            try:
                import tkinter as _tk
                r = _tk.Tk()
                r.withdraw()
                sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
                r.destroy()
            except Exception:
                sw, sh = 1440, 900
            kwargs.update(native=True, window_size=(sw, sh), show=False)
        except Exception:
            kwargs["show"] = True
    else:
        kwargs["show"] = True
    clog(f"TimesFM GUI  http://{host}:{port}  native={use_native}", "STEP")
    last_err: Optional[BaseException] = None
    for attempt in range(1, 8):
        kwargs["port"] = port
        try:
            ui.run(_root, **kwargs)
            return
        except OSError as e:
            last_err = e
            err = str(e).lower()
            occupied = (
                e.errno in (98, 48, 10048)
                or "address already in use" in err
                or "only one usage of each socket" in err
            )
            if not occupied:
                clog(f"启动失败: {e}", "ERROR")
                raise
            clog(f"监听 {host}:{port} 失败（占用），第 {attempt} 次换端口", "WARN")
            port = pick_free_port(host, None)
    clog(f"多次换端口仍失败: {last_err}", "ERROR")
    raise last_err if last_err else RuntimeError("无法绑定端口")


if __name__ in {"__main__", "__mp_main__"}:
    import multiprocessing
    multiprocessing.freeze_support()
    main()

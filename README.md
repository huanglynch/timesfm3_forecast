# TimesFM 3 Forecast GUI

[English](README_EN.md) · 中文

零样本时序预测桌面工具。把一条干净的价格序列交给 Google TimesFM，得到未来路径、分位区间，以及可核对的回测指标。

**只做一件事：** 取数 → 预测 → 判断这条路径值不值得信。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#环境要求)
[![NiceGUI](https://img.shields.io/badge/UI-NiceGUI-00b4a6)](#功能一览)
[![TimesFM 3.0](https://img.shields.io/badge/Model-TimesFM%203.0%20%2F%202.5-4285F4?logo=google)](#模型与许可)
[![License](https://img.shields.io/badge/Code-Apache--2.0-brightgreen)](LICENSE)
[![Weights](https://img.shields.io/badge/3.0%20Weights-Non--Commercial-orange)](#模型与许可)

---

## 这是什么

Google Research 的 [TimesFM](https://github.com/google-research/timesfm) 是预训练时序基础模型。3.0（约 3.3 亿参数）在 fev-bench / TIME / GIFT-Eval 上均为 foundation model 第一档，但 **3.0 权重禁止商用与生产部署**。2.5 权重仍是 Apache-2.0。

本仓库不是模型本身，而是一层 **NiceGUI 原生窗口**：

- 美股 / A股 / 港股 / 指数一键取数
- 本地或 Hugging Face 加载 TimesFM 3.0，缺失时自动回退 2.5
- 点预测 + p10 / p50 / p90 区间
- Holdout 回测（MAE / MAPE / RMSE / 区间覆盖率）
- 结果落盘 CSV，可换肤，可打包 EXE

> 恐惧贪婪等衍生指标仍保留为一个独立按钮，不进入主预测路径。

---

## 界面示意

```
┌─ TimesFM 3 · 零样本时序预测 ──────────────── 皮肤 ▾  加载模型  帮助 ─┐
│ 待预测队列          │  数据源  代码  天数  频率   [获取并加入] [Key]   │
│  · 600519_….csv     │  步长  上下文  设备  模型   回测  成交量协变量   │
│ 最近结果            │  [开始预测]                                      │
│                     │  ┌──────────────────────────────────────────┐  │
│ 回测指标            │  │  历史收盘 + 点预测 + p10–p90 色带         │  │
│  MAE / MAPE / 覆盖率│  │  （可选）回测：预测 vs 真实               │  │
│                     │  └──────────────────────────────────────────┘  │
│                     │  日志                                           │
└─────────────────────┴────────────────────────────────────────────────┘
```

典型输出目录：

```
timesfm_forecast_outputs/
  600519_AKShare_免费A股港股指数_D_20260912.csv
  600519_AKShare_免费A股港股指数_D_20260912_forecast.csv
  600519_AKShare_免费A股港股指数_D_20260912_holdout_metrics.json
```

---

## 功能一览

| 模块 | 做什么 | 备注 |
| --- | --- | --- |
| 数据源 | AKShare / Polygon / Finnhub / FMP | Yahoo 已移除（限流）。中国市场默认 AKShare |
| 代码归一化 | `000001` = 平安银行；`000001.SH` / `上证` = 上证综指 | 这是本工具最容易踩的坑，已单独处理 |
| 本地文件 | CSV / Excel / Parquet / JSON | 自动识别日期、收盘、成交量列名（中英皆可） |
| 模型 | 自动 3.0 → 2.5，或强制其一 | 本地 `models/` 有缓存则不问下载 |
| 预测 | 点预测 + 分位带 | 3.0 可用成交量作 past-only 协变量 |
| 回测 | 截去最后 `horizon` 步做 holdout | 覆盖率接近 80% 才说明区间有校准感 |
| 皮肤 | `timesfm_skins.yaml` / `oma_skins.yaml` | 内置「印刷纸」 |
| 打包 | PyInstaller 友好 | 写文件绝不落入 `_MEI` 临时目录 |
| 许可提示 | 启动即打印 | 3.0 非商用；商用请强制 2.5 |

### 支持的市场写法

| 你输入 | 工具理解 | 建议数据源 |
| --- | --- | --- |
| `AAPL` `MSFT` `NVDA` `TSLA` | 美股 | Polygon / Finnhub / FMP |
| `I:SPX` `I:IXIC` `^GSPC` | 美股指数 | Polygon |
| `600519` | 贵州茅台 | AKShare |
| `000001` | **平安银行**（不是上证） | AKShare |
| `000001.SH` `上证` `SSEC` | 上证综指 | AKShare |
| `000300` `沪深300` | 沪深300 | AKShare |
| `00700` `09988` | 腾讯 / 阿里-SW | AKShare |

---

## 快速开始

### 环境要求

- Python 3.10+（推荐 3.11）
- 内存：CPU 推理建议 ≥ 16 GB；GPU 可选
- 磁盘：3.0 权重约 1 GB+，2.5 约数百 MB
- 网络：首次下载权重走 Hugging Face；A 股走东方财富（请关掉会拦截 eastmoney.com 的系统代理）

### 安装

```bash
git clone https://github.com/<YOUR_USER>/timesfm3-forecast-gui.git
cd timesfm3-forecast-gui

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# TimesFM 官方包（含 2.5；3.0 随 timesfm 新版本提供 timesfm3）
pip install "timesfm[torch]"
```

国内镜像示例：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 启动

```bash
python timesfm3_forecast_gui.py
```

默认开 **原生窗口**（NiceGUI + pywebview）。只要浏览器：

```bash
TFM_APP=browser python timesfm3_forecast_gui.py
```

常用环境变量：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `TFM_HOST` | `127.0.0.1` | 监听地址 |
| `TFM_PORT` | 自动选空闲口 | 被占用会往后找 |
| `TFM_APP` | `1` | `0` / `browser` 则走浏览器 |

配置、缓存、输出都写在 **可写目录**（源码旁或 EXE 旁），不会写进 PyInstaller 解包目录：

```
timesfm_config.json          # 数据源、Key、皮肤、上次代码
models/                      # HF 权重缓存
timesfm_forecast_outputs/    # CSV / 回测 JSON
timesfm_runtime.log          # 启动与异常
```

---

## 使用步骤（建议按这个顺序）

### 1. 加载模型

点标题栏 **加载模型**。

- 本地 `models/` 已有 `google/timesfm-3.0-pytorch` 或 `google/timesfm-2.5-200m-pytorch`：静默加载，不询问。
- 没有缓存：弹窗确认后再下载（约 1 GB+）。
- 偏好：
  - **自动 3→2.5**：有 3.0 用 3.0，否则 2.5
  - **强制 3.0**：失败即停（不要半途 import 再拖垮 2.5）
  - **强制 2.5**：Apache-2.0，可商用研究以外的部署场景请走这条

### 2. 准备一条序列

两条路，任选其一。

**在线取数**

1. 选数据源（中国市场用 AKShare）
2. 输入代码（见上表）
3. 天数 / 频率（日 / 周 / 月）
4. 美股源先点 **Key** 填入 API Key（只存在本机 `timesfm_config.json`）
5. **获取并加入** → 左侧队列出现 CSV，中间画出历史收盘

**本地文件**

左侧 **文件**，选含日期与收盘价的表。列名支持 `date` / `日期`、`close` / `收盘` / `value`、`volume` / `成交量`。

### 3. 预测

| 控件 | 建议 |
| --- | --- |
| 预测步长 | 日线 20–60；周线 8–26。更长则区间会迅速变宽 |
| 上下文长度 | 默认 512。太短丢失周期，太长在 CPU 上变慢 |
| 回测评估 | 打开。序列长度需 `> horizon + 16` |
| 成交量协变量 | 仅 3.0 真正使用；2.5 会忽略 |

点 **开始预测**。完成后：

- 上图：历史 + 虚线点预测 + p10–p90 色带
- 下图（若开回测）：预测 vs 真实
- 左侧：**MAE / MAPE / RMSE / p10–p90 覆盖率**
- 磁盘：`*_forecast.csv`

### 4. 怎么读结果

零样本模型给出的是 **条件分布的中位路径与分位带**，不是承诺。

| 你看到 | 较可信的读法 |
| --- | --- |
| 点预测上翘、色带也很宽 | 「中位偏多，但上下都有空间」——不要当成目标价 |
| 回测 MAPE < 5% 且覆盖率 70–90% | 这条品种、这个频率上，模型近期校准还行 |
| 回测覆盖率 100% 或 < 40% | 区间过宽或过窄，不要用色带做风控 |
| 日线 horizon=60 的远期点 | 噪声主导。看近 10–20 步的斜率与带宽即可 |

本工具 **不是投资建议**。价格预测不能替代基本面、流动性和你自己的风控。

---

## 配置文件

`timesfm_config.json` 会在首次运行后自动生成。关键字段：

```json
{
  "default_data_source": "akshare",
  "selected_skin": "paper",
  "device": "cpu",
  "model_preference": "auto",
  "horizon": 30,
  "context_len": 512,
  "holdout_eval": true,
  "use_volume_covariate": false,
  "last_ticker": "600519",
  "tickers": ["AAPL", "600519", "00700", "I:SPX"],
  "polygon":  { "api_key": "", "default_days": 90 },
  "finnhub":  { "api_key": "", "default_days": 365 },
  "fmp":      { "api_key": "", "default_days": 365 },
  "akshare":  { "default_days": 365 }
}
```

API Key **不要提交到 Git**。仓库已在 `.gitignore` 中排除该文件。

可选皮肤文件：`timesfm_skins.yaml` 或 `oma_skins.yaml`，与脚本同目录即可。缺失时使用内置「印刷纸」。

---

## 模型与许可

| 部件 | 许可 | 你能做什么 |
| --- | --- | --- |
| 本仓库 GUI 源码 | Apache-2.0 | 使用、修改、再分发 |
| TimesFM **源码**（Google） | Apache-2.0 | 同上 |
| TimesFM **2.5 及更早权重** | Apache-2.0 | 可用于商业，仍须遵守其条款 |
| TimesFM **3.0 预训练权重** | [timesfm-non-commercial-license-v1.0](https://huggingface.co/google/timesfm-3.0-pytorch) | **仅非商用、非生产**：研究、评测、实验。不可用于营收活动、面向终端用户的生产系统，也不可用来蒸馏商用模型 |

界面默认优先 3.0，是因为预测质量通常更好；**商业或生产请在「模型」里选强制 2.5**。

权重来源：

- 3.0：[`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch)
- 2.5：[`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch)

论文：[A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)（ICML 2024）。3.0 介绍见 [Google Research 博客](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)。

---

## 常见问题

**启动后模型徽章一直是「未加载」**  
本地没有缓存。点「加载模型」并允许下载，或把已下载的 snapshot 放到 `models/`。

**AKShare 报 ProxyError / 超时**  
系统或终端代理拦了 `eastmoney.com`。关掉代理，或对该域名设直连。

**`000001` 怎么变成了银行而不是上证？**  
无后缀的 `000001` 按 A 股个股处理。指数请写 `000001.SH`、`上证` 或 `SSEC`。

**EXE 里提示缺 `calendar.json` / `safetensors`**  
用带 `collect-all akshare` 与 `safetensors` 的最新打包脚本重打。源码运行一般不会碰到。

**3.0 加载失败后进程直接退出**  
半途失败的 torch import 可能污染后续 2.5。用「强制 2.5」，或换一份完整的 3.0 缓存后再试「自动」。

**预测看起来完全没学到这只股票的个性**  
这是零样本基础模型的正常行为：它学的是「时序的通用形状」，不是某家公司的财报。回测差就缩短 horizon、换频率，或不要用它做交易信号。

---

## 仓库结构

```
timesfm3-forecast-gui/
├── timesfm3_forecast_gui.py   # 单文件应用
├── requirements.txt
├── LICENSE                    # 本仓库代码 Apache-2.0
├── NOTICE                     # 第三方与权重许可声明
├── README.md                  # 本文件（中文）
├── README_EN.md
├── CONTRIBUTING.md
├── .gitignore
└── docs/
    ├── GITHUB_UPLOAD_GUIDE.md # 从零上传到 GitHub
    └── timesfm_help.md        # 程序内「帮助」读取同一份
```

运行时生成（不要提交）：

```
timesfm_config.json
models/
timesfm_forecast_outputs/
timesfm_runtime.log
```

---

## 致谢

- [Google Research / TimesFM](https://github.com/google-research/timesfm)
- [NiceGUI](https://nicegui.io/)
- [AKShare](https://github.com/akfamily/akshare)
- Polygon · Finnhub · Financial Modeling Prep

本工具与 Google 无官方关系，也不是投资顾问产品。

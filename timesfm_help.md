# TimesFM 3 Forecast GUI · 使用说明

程序「帮助」按钮读取这份文件。请把副本同时放在仓库根目录，文件名 `timesfm_help.md`。

## 目的

拿到一条干净的价格序列，用 TimesFM 给出未来路径和 p10–p90 区间，并用 holdout 回测量一下这条路径值不值得信。

## 五步

1. **加载模型**  
   本地 `models/` 有缓存会静默加载。没有则确认后下载。商用 / 生产请选「强制 2.5」。
2. **准备数据**  
   中国市场用 AKShare。美股用 Polygon / Finnhub / FMP（先填 Key）。或直接加入本地 CSV / Excel。
3. **看代码含义**  
   `000001` = 平安银行。上证综指写 `000001.SH` 或 `上证`。`00700` = 腾讯。
4. **预测**  
   日线步长 20–60，上下文 512，打开回测。序列至少 16 点，回测还要再长一个 horizon。
5. **读图**  
   虚线是点预测，色带是 p10–p90。覆盖率接近 80% 比 100% 更有校准感。这不是目标价，也不是投资建议。

## 许可

- 本 GUI：Apache-2.0
- TimesFM 2.5 权重：Apache-2.0
- TimesFM 3.0 权重：非商用、非生产（timesfm-non-commercial-license-v1.0）

## 文件写在哪

与脚本或 EXE 同目录：

- `timesfm_config.json` — 设置与 API Key（仅本机）
- `models/` — 权重缓存
- `timesfm_forecast_outputs/` — 取数 CSV 与预测 CSV
- `timesfm_runtime.log` — 启动与异常

## 常见失败

- **ProxyError**：关掉拦截 eastmoney.com 的系统代理。
- **模型未加载**：点「加载模型」，允许下载或放入本地 snapshot。
- **EXE 缺 calendar.json / safetensors**：用最新打包脚本重打。

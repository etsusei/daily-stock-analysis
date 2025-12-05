import yfinance as yf
import pandas as pd
import io
import google.generativeai as genai
import markdown
import webbrowser
import os
import time

# ================= 用户配置区域 =================
SYMBOLS = ["IONQ", "OKLO","SMR","LUMN","UEC","MRVL","CCJ","NVDA"]  # 股票代码列表
API_KEY = "AIzaSyCqbO7kvmQdjT2Ilys8ZXMR1oWnHh5jQ3c" # Gemini API Key
MODEL_NAME = "gemini-2.5-pro" # 使用最新的稳定版模型
# ===============================================

# 配置 Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def calculate_complex_indicators(df):
    """
    计算全套指标：
    1. EMA (5, 10, 20, 30, 55, 80, 120, 135, 180)
    2. Bollinger Bands (20, 2)
    3. MACD (5, 15, 6)
    4. KDJ (9, 3, 3)
    5. RSI (14)
    """
    if df.empty: return df

    # --- 1. 批量计算 EMA 均线组 ---
    # 你的均线列表包含非常长周期的 135 和 180，需要足够历史数据
    ema_periods = [5, 10, 20, 30, 55, 80, 120, 135, 180]
    for p in ema_periods:
        # adjust=False 更加符合传统金融软件的算法
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

    # --- 2. 布林带 (Bollinger Bands) ---
    # 参数：(20, 2)。虽然你提到了 1,3，但通常 AI 分析标准是 2 倍标准差。
    # 中轨 (使用 SMA 20)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    # 标准差
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    # 上轨 & 下轨 (2倍标准差)
    df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    # --- 3. RSI (14) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- 4. MACD (5, 15, 6) 自定义参数 ---
    ema_fast = df['Close'].ewm(span=5, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=15, adjust=False).mean()
    df['MACD_DIF'] = ema_fast - ema_slow
    df['MACD_DEA'] = df['MACD_DIF'].ewm(span=6, adjust=False).mean()
    df['MACD_Hist'] = 2 * (df['MACD_DIF'] - df['MACD_DEA'])

    # --- 5. KDJ (9, 3, 3) ---
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

def get_data_slice(symbol, interval, fetch_period, slice_count, label):
    """
    抓取 -> 计算 -> 切片
    fetch_period: 为了计算 EMA180，必须抓很多数据 (比如 "max")
    slice_count: 最后只给 AI 看最近的 N 条 (比如 120)
    """
    # print(f"正在处理 {label} ...") # 减少控制台输出
    ticker = yf.Ticker(symbol)
    
    # 强制抓取最大历史数据，以确保 EMA180 能算出来
    df = ticker.history(period="max", interval=interval)
    
    if df.empty:
        return f"\n{label}: 无数据\n"

    # 计算全套指标
    df = calculate_complex_indicators(df)

    # 截取用户要求的最后 N 条
    # 如果数据不足 N 条，就取全部
    rows_to_keep = min(len(df), slice_count)
    df_slice = df.tail(rows_to_keep).copy()

    # 格式化: 只需要特定列，防止 CSV 太宽太乱 (虽然你不在乎长，但要清晰)
    # 动态生成列名列表
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'K', 'D', 'J', 
            'MACD_DIF', 'MACD_DEA', 'MACD_Hist', 
            'BB_Up', 'BB_Mid', 'BB_Low']
    # 加入所有 EMA 列
    ema_cols = [f'EMA_{p}' for p in [5, 10, 20, 30, 55, 80, 120, 135, 180]]
    cols.extend(ema_cols)

    # 检查列是否存在 (防止新股数据太少算不出 EMA180 导致报错)
    existing_cols = [c for c in cols if c in df_slice.columns]
    output_df = df_slice[existing_cols]

    # 保留 2 位小数
    output_df = output_df.round(2)
    
    # 时间格式化
    output_df.index = output_df.index.strftime('%Y-%m-%d')

    csv_buffer = io.StringIO()
    csv_buffer.write(f"dataset: {label} (Interval: {interval}, Display: Last {rows_to_keep} bars)\n")
    output_df.to_csv(csv_buffer)
    
    return csv_buffer.getvalue()

def analyze_stock(symbol):
    print(f"正在分析 {symbol} ...")
    
    full_prompt = f"分析目标: {symbol}\n"
    full_prompt += "指标说明:\n"
    full_prompt += "1. EMA组: 5,10,20,30,55,80,120,135,180 (注意：如果是新股或月线数据不足，长周期均线可能为空)\n"
    full_prompt += "2. 布林带: 参数(20, 2)\n"
    full_prompt += "3. MACD: (5,15,6) | KDJ: (9,3,3) | RSI: (14)\n"
    full_prompt += "=" * 50 + "\n\n"

    # 1. 日线: 抓取 max，截取最后 120 天
    full_prompt += get_data_slice(symbol, "1d", "max", 120, "日线 (Daily - Last 120 days)") + "\n\n"
    
    # 2. 周线: 抓取 max，截取最后 52 周 (约1年)
    full_prompt += get_data_slice(symbol, "1wk", "max", 52, "周线 (Weekly - Last 1 year)") + "\n\n"
    
    # 3. 月线: 抓取 max，截取最后 24 个月
    full_prompt += get_data_slice(symbol, "1mo", "max", 24, "月线 (Monthly - Last 2 years)")
    
    full_prompt += "\n" + "="*20 + "\n"
    full_prompt += f"""
# Role: 顶级对冲基金资深股票分析师 (Senior Hedge Fund Analyst)

## 核心任务
我是你的核心客户。请基于我提供的全套技术指标数据（日线/周线/月线），从主力的视角告诉我这个散户（最好有主力和散户思路对比）挖掘数据背后的资金意图，并为我制定接下来的交易策略。

## 输入信息
* **用户关注点：我想知道接下来我应该关注哪些点位？
* **数据来源：** 下方附带的 CSV 数据块

## 分析数据 (Data Block)
(请在发送时附上 CSV 格式数据，包含 OHLCV, EMA组, BB, MACD, KDJ, RSI)

---

## 指令：请严格按照以下框架输出分析报告

### 1. 🚨 盘前核心判断 (The Verdict)
* **趋势定性：** 读取数据中**最新一行的收盘价**，结合 EMA 均线状态，判断当前是反转、加速还是回调等？
* **量能“测谎”：** 重点分析**最近3根 K 线的成交量 (Volume)**。相比前几天，是有主力资金进场抢筹，还是缩量观望？

### 2. 实战必须盯紧的三大点位 (Key Levels to Watch)
* **⚔️ 上方阻力位（冲关点）：** 计算布林带上轨、前高或整数关口的压力。
* **🛡️ 下方支撑位（防守线）：** 找出最关键的均线支撑（如 EMA55/EMA20）。如果跌破意味着什么？
* **⚖️ 用户专属点位：** 结合我的【关注点】，指出关键位置。

### 3. 技术面深度透视 (Institutional Deep Dive)
*拒绝罗列数字，我要看逻辑：*
* **均线系统 (EMAs)：** 是否有关键的“金叉”或“一阳穿多线”等形态？牛熊分界线（EMA55/120）是否已被收复？
* **指标共振 (Indicators)：**
    * **MACD：** 动能强弱？是否出现金叉/死叉？
    * **KDJ：** J值是否过高（>90 提示超买）或过低？
    * **RSI：** 处于强势区还是弱势区？
* **大周期确认 (Weekly/Monthly)：** 周线级别是否有“包容形态”或其他趋势配合？

### 4. 交易博弈推演 (Scenario Planning)
* **情景 A (强势上攻)：** 如果开盘直接冲过阻力位，应该追涨还是减仓？
* **情景 B (回踩确认)：** 如果股价回调，哪个位置是“倒车接人”的买点？
* **情景 C (风险预警)：** 跌破哪个价格要考虑止损？

### 5. 分析师总结 (Conclusion)
* 用一句最精炼的话总结：**主力资金想干什么？我该把注意力放在哪里？**

---
**格式要求：**
1. **数据驱动：** 所有观点必须引用 CSV 中的具体数值（如成交量倍数、EMA价格）。
2. **重点突出：** 关键价格和建议请使用**加粗**。

Here is the Data:
"""

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 调用失败: {str(e)}"

def main():
    print(f"=== 批量生成全指标分析报告 ===\n")
    
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Analysis Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }
            h1 { text-align: center; color: #2c3e50; }
            .stock-card { background: #fff; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; padding: 25px; }
            .stock-title { font-size: 1.8em; color: #2980b9; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
            .analysis-content { background-color: #fafafa; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; }
            h2, h3 { color: #34495e; }
            ul, ol { padding-left: 20px; }
            code { background-color: #eee; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Gemini Stock Analysis Report</h1>
    """

    for symbol in SYMBOLS:
        analysis_text = analyze_stock(symbol)
        
        # Convert Markdown to HTML
        analysis_html = markdown.markdown(analysis_text, extensions=['extra', 'codehilite'])
        
        html_content += f"""
        <div class="stock-card">
            <div class="stock-title">{symbol}</div>
            <div class="analysis-content">
                {analysis_html}
            </div>
        </div>
        """
        # Avoid hitting rate limits
        time.sleep(2)

    html_content += """
    </body>
    </html>
    """

    # Save HTML file
    output_file = "stock_analysis_report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n分析完成! 报告已生成: {output_file}")
    
    # Open in browser
    webbrowser.open('file://' + os.path.realpath(output_file))

if __name__ == "__main__":
    main()
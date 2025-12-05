import yfinance as yf
import pandas as pd
import io
import google.generativeai as genai
import markdown
import webbrowser
import os
import time
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 用户配置区域 =================
SYMBOLS = ["IONQ", "OKLO","SMR","LUMN","UEC","MRVL","CCJ","NVDA"] # 股票代码列表
API_KEY = os.getenv("GEMINI_API_KEY")  # 从环境变量读取API密钥
PRIMARY_MODEL = "gemini-2.5-pro"      # 主要模型：质量更高但配额较低 (RPD=50)
FALLBACK_MODEL = "gemini-2.5-flash"   # 备用模型：配额更高 (RPD=250)
# ===============================================

# 检查API密钥是否存在
if not API_KEY:
    raise ValueError("请在.env文件中设置GEMINI_API_KEY环境变量")

# 配置 Gemini - 初始使用主要模型
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(PRIMARY_MODEL)
current_model_name = PRIMARY_MODEL  # 追踪当前使用的模型

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

def get_options_analysis(symbol):
    """
    智能期权分析：寻找距离今天约 2 周 (14天) 的期权，计算市场押注范围
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. 获取当前股价 (作为计算基准)
        try:
            current_price = ticker.fast_info['last_price']
        except:
            hist = ticker.history(period="1d")
            if hist.empty: return "无法获取当前股价"
            current_price = hist['Close'].iloc[-1]

        # 2. 获取到期日列表
        expirations = ticker.options
        if not expirations:
            return "无期权数据"
        
        # === 核心逻辑修改：寻找最接近 14 天后的到期日 ===
        today = datetime.now(tz=timezone(timedelta(hours=8))).date()
        target_date_str = expirations[0] # 默认兜底
        
        best_diff = 999
        target_days = 14 # <--- 设定目标为 2 周
        
        for date_str in expirations:
            exp_date = pd.to_datetime(date_str).date()
            days_diff = (exp_date - today).days
            
            # 过滤掉 3 天以内的末日轮，噪音太大
            if days_diff < 3: 
                continue
                
            # 寻找差值最小的日期
            if abs(days_diff - target_days) < best_diff:
                best_diff = abs(days_diff - target_days)
                target_date_str = date_str

        # 计算实际的剩余天数 (DTE)
        target_date = pd.to_datetime(target_date_str).date()
        dte = (target_date - today).days
        if dte < 1: dte = 1

        # 3. 获取该日期的期权链
        opt = ticker.option_chain(target_date_str)
        calls = opt.calls
        puts = opt.puts
        
        if calls.empty or puts.empty:
            return f"期权数据不足"

        # --- 计算核心数据 ---
        
        # A. 寻找最大痛点 (Max OI Walls)
        # 增加过滤器：只看当前股价上下 20% 范围内的，去除极度虚值的无效单
        filter_mask_call = (calls['strike'] > current_price * 0.8) & (calls['strike'] < current_price * 1.2)
        filter_mask_put = (puts['strike'] > current_price * 0.8) & (puts['strike'] < current_price * 1.2)
        
        filtered_calls = calls[filter_mask_call]
        filtered_puts = puts[filter_mask_put]
        
        # 如果过滤完空了，就用原始数据兜底
        if filtered_calls.empty: filtered_calls = calls
        if filtered_puts.empty: filtered_puts = puts

        max_call_oi_row = filtered_calls.loc[filtered_calls['openInterest'].idxmax()]
        max_put_oi_row = filtered_puts.loc[filtered_puts['openInterest'].idxmax()]
        
        resistance_strike = max_call_oi_row['strike']
        support_strike = max_put_oi_row['strike']

        # B. 计算两周预期波动 (Expected Move)
        # 1. 计算平均 IV (Implied Volatility)
        avg_iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2
        
        # 2. 核心公式：Expected Move = Price * IV * sqrt(Days / 365)
        expected_move_price = current_price * avg_iv * ((dte / 365.0) ** 0.5)
        
        upper_bound = current_price + expected_move_price
        lower_bound = current_price - expected_move_price
        
        # C. 情绪 PCR (成交量)
        vol_pcr = puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0

        # --- 生成报告文本 ---
        report = f"--- 🏛️ 双周博弈分析 (2-Week Outlook) ---\n"
        report += f"当前价: ${current_price:.2f} | 目标日期: {target_date_str} (未来 {dte} 天)\n"
        report += f"隐含波动率 (IV): {avg_iv*100:.2f}% (年化)\n\n"
        
        report += f"📊 **市场定价波动范围 (Expected Move):**\n"
        report += f"期权市场押注接下来的两周，股价将在 **${lower_bound:.2f} ~ ${upper_bound:.2f}** 之间波动。\n"
        report += f"(如果不发生突发黑天鹅，主力认为很难突破此区间)\n\n"
        
        report += f"🛡️ **主力攻防线 (OI Walls):**\n"
        report += f"🔴 上方阻力墙: ${resistance_strike} (OI: {int(max_call_oi_row['openInterest'])})\n"
        report += f"🟢 下方支撑墙: ${support_strike} (OI: {int(max_put_oi_row['openInterest'])})\n"
        report += f"💡 逻辑: 只有当股价强势突破 ${resistance_strike}，才可能引发伽马挤压(Gamma Squeeze)加速上涨。\n"
        
        return report

    except Exception as e:
        return f"期权分析异常: {str(e)}"

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
    
    # 4. 期权分析
    full_prompt += "\n" + get_options_analysis(symbol) + "\n"
    
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

### 5. 🌪️ 衍生品市场与情绪暗涌 (Options & Sentiment Flow)
*基于提供的期权数据进行分析：*
* **押注范围验证：** 对比期权计算出的 **Expected Move ** 与你技术分析计算的布林带或均线支撑压力。如果技术位在期权押注范围内，支撑/压力更有效；如果超出范围，说明市场并未定价该风险。
* **主力筹码墙 (Walls)：** 如果股价接近 **Call Wall**，警惕庄家为了不赔付期权而刻意打压股价。
* **波动率 (IV) 状态：** 当前 IV 是否过高？如果 IV 很高但股价不涨，是否意味着大资金在买 Put 对冲暴跌风险？

### 6. 分析师总结 (Conclusion)
* 用一句最精炼的话总结：**主力资金想干什么？我该把注意力放在哪里？**

---
**格式要求：**
1. **数据驱动：** 所有观点必须引用 CSV 中的具体数值（如成交量倍数、EMA价格）。
2. **重点突出：** 关键价格和建议请使用**加粗**。

Here is the Data:
"""

    global model, current_model_name
    max_retries = 3
    retry_delay = 30
    has_tried_fallback = False  # 标记是否已经尝试过降级
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            is_quota_error = "429" in error_msg or "quota" in error_msg.lower()
            
            if is_quota_error:
                # 如果遇到配额错误且还在使用主模型，尝试切换到备用模型
                if current_model_name == PRIMARY_MODEL and not has_tried_fallback:
                    print(f"⚠️  {PRIMARY_MODEL} 配额已用完，切换到 {FALLBACK_MODEL}")
                    model = genai.GenerativeModel(FALLBACK_MODEL)
                    current_model_name = FALLBACK_MODEL
                    has_tried_fallback = True
                    time.sleep(5)  # 短暂等待后重试
                    continue
                
                # 如果已经在使用备用模型或已尝试过降级，则等待后重试
                if attempt < max_retries - 1:
                    import re
                    match = re.search(r'retry in (\d+\.?\d*)', error_msg)
                    if match:
                        wait_time = max(float(match.group(1)), retry_delay)
                    else:
                        wait_time = retry_delay * (2 ** attempt)
                    
                    print(f"⏳ 遇到速率限制，等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return f"Gemini API 调用失败（已重试{max_retries}次）: {error_msg}"
            else:
                return f"Gemini API 调用失败: {error_msg}"
    
    return f"Gemini API 调用失败: 超过最大重试次数"

def main():
    global current_model_name
    print(f"=== 批量生成全指标分析报告 ===")
    print(f"📊 当前使用模型: {current_model_name}\n")
    
    # 1. 生成侧边栏链接 HTML
    sidebar_links = ""
    for symbol in SYMBOLS:
        sidebar_links += f'<a href="#{symbol}" onclick="closeSidebar()">{symbol}</a>\n'



    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Analysis Report</title>
        <style>
            :root {{
                --sidebar-width: 250px;
                --primary-color: #2c3e50;
                --accent-color: #2980b9;
                --bg-color: #f4f4f9;
            }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                margin: 0; 
                padding: 0; 
                background-color: var(--bg-color); 
            }}
            
            /* --- Sidebar Styling --- */
            .sidebar {{
                height: 100%;
                width: var(--sidebar-width);
                position: fixed;
                z-index: 1000;
                top: 0;
                left: 0;
                background-color: var(--primary-color);
                overflow-x: hidden;
                padding-top: 60px; /* Space for hamburger on mobile or top padding */
                transition: 0.3s;
                box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            }}
            
            .sidebar a {{
                padding: 15px 25px;
                text-decoration: none;
                font-size: 18px;
                color: #ecf0f1;
                display: block;
                transition: 0.3s;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            
            .sidebar a:hover {{
                background-color: var(--accent-color);
                padding-left: 35px; /* Slide effect */
            }}
            
            /* --- Main Content Styling --- */
            .main-content {{
                margin-left: var(--sidebar-width); /* Same as sidebar width */
                padding: 20px 40px;
                transition: margin-left 0.3s;
            }}
            
            h1 {{ text-align: center; color: var(--primary-color); margin-bottom: 40px; }}
            
            .stock-card {{ 
                background: #fff; 
                border-radius: 8px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                margin-bottom: 40px; 
                padding: 30px; 
                scroll-margin-top: 20px; /* For smooth anchor scrolling */
            }}
            
            .stock-title {{ 
                font-size: 2em; 
                color: var(--accent-color); 
                border-bottom: 2px solid #eee; 
                padding-bottom: 15px; 
                margin-bottom: 25px; 
                font-weight: bold;
            }}
            
            .analysis-content {{ background-color: #fafafa; padding: 20px; border-radius: 5px; border-left: 5px solid var(--accent-color); }}
            h2, h3 {{ color: var(--primary-color); margin-top: 1.5em; }}
            ul, ol {{ padding-left: 20px; }}
            code {{ background-color: #eee; padding: 2px 5px; border-radius: 3px; font-family: Consolas, monospace; }}

            /* --- Hamburger Menu (Mobile) --- */
            .hamburger {{
                display: none;
                position: fixed;
                top: 15px;
                left: 15px;
                z-index: 1001;
                background: var(--accent-color);
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 20px;
                cursor: pointer;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}

            /* --- Responsive Design --- */
            @media screen and (max-width: 768px) {{
                .sidebar {{
                    width: 0; /* Hidden by default on mobile */
                    padding-top: 60px;
                    width: 250px;
                    transform: translateX(-100%); /* Move off-screen */
                }}
                
                .sidebar.active {{
                    transform: translateX(0); /* Slide in */
                }}

                .main-content {{
                    margin-left: 0; /* Full width on mobile */
                    padding: 15px;
                }}
                
                .hamburger {{
                    display: block; /* Show button */
                }}
                
                /* Overlay when sidebar is open */
                .overlay {{
                    display: none;
                    position: fixed;
                    width: 100%;
                    height: 100%;
                    top: 0; 
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: rgba(0,0,0,0.5); 
                    z-index: 999; 
                    cursor: pointer;
                }}
                
                .overlay.active {{
                    display: block;
                }}
            }}
        </style>
    </head>
    <body>

    <!-- Mobile Menu Button -->
    <button class="hamburger" onclick="toggleSidebar()">☰ 目录</button>
    
    <!-- Sidebar -->
    <div class="sidebar" id="mySidebar">
        <div style="text-align: center; padding: 20px 0; color: white; font-weight: bold; font-size: 1.2em; border-bottom: 1px solid rgba(255,255,255,0.1);">
            Stock Analysis
        </div>
        {sidebar_links}
    </div>
    
    <!-- Overlay for mobile -->
    <div class="overlay" id="myOverlay" onclick="closeSidebar()"></div>

    <!-- Main Content -->
    <div class="main-content">
        <h1>Gemini持仓股分析报告 <span style="font-size: 0.5em; color: gray; display: block; margin-top: 10px;">(生成时间: {{GEN_TIME}})</span></h1>
    """

    for symbol in SYMBOLS:
        analysis_text = analyze_stock(symbol)
        analysis_html = markdown.markdown(analysis_text, extensions=['extra', 'codehilite'])
        
        # Add ID for anchor linking
        html_content += f"""
        <div id="{symbol}" class="stock-card">
            <div class="stock-title">{symbol}</div>
            <div class="analysis-content">
                {analysis_html}
            </div>
        </div>
        """
        # 根据当前模型调整延迟：
        # Pro 模型: RPM=2, 需要 35 秒
        # Flash 模型: RPM=10, 只需 7 秒
        delay = 35 if current_model_name == PRIMARY_MODEL else 7
        print(f"✅ 已完成 {symbol}，等待 {delay} 秒... (当前模型: {current_model_name})")
        time.sleep(delay)

    html_content += """
    </div> <!-- End main-content -->

    <script>
        function toggleSidebar() {
            document.getElementById("mySidebar").classList.toggle("active");
            document.getElementById("myOverlay").classList.toggle("active");
        }

        function closeSidebar() {
            // Only relevant for mobile where these classes toggle visibility
            document.getElementById("mySidebar").classList.remove("active");
            document.getElementById("myOverlay").classList.remove("active");
        }
    </script>
    </body>
    </html>
    """

    # 获取当前时间（UTC+8 北京时间）
    tz_utc8 = timezone(timedelta(hours=8))
    gen_time = datetime.now(tz=tz_utc8).strftime("%Y-%m-%d %H:%M:%S")
    html_content = html_content.replace("{GEN_TIME}", gen_time)

    # Save HTML file
    output_file = "stock_analysis_report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n分析完成! 报告已生成: {output_file}")
    
    # Open in browser
    webbrowser.open('file://' + os.path.realpath(output_file))

if __name__ == "__main__":
    main()
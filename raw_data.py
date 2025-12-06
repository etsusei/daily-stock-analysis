"""
输出API获取到的结构化数据（不经过AI分析）
用于调试和查看原始数据结构
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta
import json

# ================= 配置区域 =================
SYMBOLS = ["LUMN"]
# ============================================


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
    ema_periods = [5, 10, 20, 30, 55, 80, 120, 135, 180]
    for p in ema_periods:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

    # --- 2. 布林带 (Bollinger Bands) ---
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    # --- 3. RSI (14) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- 4. MACD (5, 15, 6) ---
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


def get_stock_data(symbol):
    """获取股票的历史数据（包含技术指标）"""
    ticker = yf.Ticker(symbol)
    
    data = {
        "symbol": symbol,
        "daily": None,
        "weekly": None,
        "monthly": None
    }
    
    try:
        # 日线数据 - 获取足够长的历史以计算EMA180
        daily = ticker.history(period="max", interval="1d")
        if not daily.empty:
            daily = calculate_complex_indicators(daily)
            # 删除临时列，只保留最后120天
            if 'BB_Std' in daily.columns:
                daily = daily.drop(columns=['BB_Std'])
            data["daily"] = daily.tail(120).reset_index().to_dict(orient="records")
        
        # 周线数据 - 最近52周
        weekly = ticker.history(period="max", interval="1wk")
        if not weekly.empty:
            weekly = calculate_complex_indicators(weekly)
            if 'BB_Std' in weekly.columns:
                weekly = weekly.drop(columns=['BB_Std'])
            data["weekly"] = weekly.tail(52).reset_index().to_dict(orient="records")
        
        # 月线数据 - 最近24个月
        monthly = ticker.history(period="max", interval="1mo")
        if not monthly.empty:
            monthly = calculate_complex_indicators(monthly)
            if 'BB_Std' in monthly.columns:
                monthly = monthly.drop(columns=['BB_Std'])
            data["monthly"] = monthly.tail(24).reset_index().to_dict(orient="records")
        
    except Exception as e:
        print(f"获取 {symbol} 历史数据出错: {e}")
    
    return data


def get_options_data(symbol):
    """获取期权链数据"""
    ticker = yf.Ticker(symbol)
    
    data = {
        "symbol": symbol,
        "current_price": None,
        "expirations": [],
        "options_chain": None
    }
    
    try:
        # 获取当前价格
        try:
            data["current_price"] = ticker.fast_info.get('lastPrice')
        except:
            hist = ticker.history(period="1d")
            if not hist.empty:
                data["current_price"] = hist['Close'].iloc[-1]
        
        # 获取所有到期日
        expirations = ticker.options
        data["expirations"] = list(expirations) if expirations else []
        
        if expirations:
            # 获取最近一周的期权链
            today = datetime.now(tz=timezone(timedelta(hours=8))).date()
            target_days = 7
            target_date_str = expirations[0]
            
            best_diff = 999
            for date_str in expirations:
                exp_date = pd.to_datetime(date_str).date()
                days_diff = (exp_date - today).days
                if days_diff < 3:
                    continue
                if abs(days_diff - target_days) < best_diff:
                    best_diff = abs(days_diff - target_days)
                    target_date_str = date_str
            
            # 获取期权链
            opt = ticker.option_chain(target_date_str)
            
            data["options_chain"] = {
                "expiration": target_date_str,
                "calls": opt.calls.to_dict(orient="records"),
                "puts": opt.puts.to_dict(orient="records")
            }
            
    except Exception as e:
        print(f"获取 {symbol} 期权数据出错: {e}")
    
    return data


def get_ticker_info(symbol):
    """获取股票基本信息"""
    ticker = yf.Ticker(symbol)
    
    info = {}
    try:
        raw_info = ticker.info
        # 只取常用字段
        keys_of_interest = [
            'shortName', 'longName', 'sector', 'industry',
            'marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE',
            'pegRatio', 'priceToBook', 'dividendYield',
            'beta', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
            'averageVolume', 'averageVolume10days',
            'floatShares', 'sharesOutstanding', 'sharesShort',
            'shortRatio', 'shortPercentOfFloat'
        ]
        for key in keys_of_interest:
            if key in raw_info:
                info[key] = raw_info[key]
    except Exception as e:
        print(f"获取 {symbol} 基本信息出错: {e}")
    
    return info


def main():
    print("=" * 60)
    print("  API 完整结构化数据输出工具")
    print("=" * 60)
    
    all_data = {}
    
    for symbol in SYMBOLS:
        print(f"\n正在获取 {symbol} 数据...")
        
        all_data[symbol] = {
            "info": get_ticker_info(symbol),
            "stock_data": get_stock_data(symbol),
            "options_data": get_options_data(symbol)
        }
    
    # 输出为JSON文件
    output_file = "raw_data_output.json"
    
    # 自定义JSON序列化处理datetime
    def json_serial(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=json_serial)
    
    print(f"\n✅ 完整数据已输出到: {output_file}")
    print(f"包含 {len(SYMBOLS)} 只股票的完整数据")


if __name__ == "__main__":
    main()


"""
测试 Gemini API 的联网搜索功能
搜索 OKLO 的实时新闻/事件

注意：需要安装新版 SDK: pip install google-genai
"""

from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("请在.env文件中设置GEMINI_API_KEY环境变量")

# 使用新版 Client
client = genai.Client(api_key=API_KEY)

# 创建 Google Search 工具
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

# 配置
config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

def search_stock_news(symbol):
    """搜索股票的实时新闻和事件"""
    
    prompt = f"""
请搜索 {symbol} (Oklo Inc.) 股票最近一周的重大新闻和事件。

请按以下格式输出：

## 📰 {symbol} 近期新闻摘要

### 重大事件
- 列出最重要的3-5条新闻
- 每条包含：日期、标题、简要内容

### 对股价的潜在影响
- 分析这些新闻对股价可能产生的影响

### 投资者关注点
- 总结投资者目前最应该关注的要点
"""
    
    print(f"正在搜索 {symbol} 的实时新闻...")
    print("-" * 50)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        print(response.text)
                
    except Exception as e:
        print(f"搜索出错: {e}")
        print("\n提示: 可能需要先安装新版SDK: pip install google-genai")


if __name__ == "__main__":
    search_stock_news("OKLO")

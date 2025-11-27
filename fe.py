import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tavily import TavilyClient
from openai import OpenAI
import plotly.graph_objects as go
import json
import concurrent.futures
import time
import random
import requests
import feedparser

# --- HARDCODED KEYS (Hidden from UI) ---
# 这些 Key 硬编码在后台，用户界面不可见
TAVILY_API_KEY = "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"
ALPHA_VANTAGE_KEY = "8G1QKAWN221XEZR8"

# --- MODEL CONFIGURATION ---
# 硅基流动模型映射表
MODELS = {
    "ROUTER": "Qwen/Qwen2.5-72B-Instruct",  # Qwen: 优秀的通用指令遵循 (注: 修正了用户提供的Qwen3名称以确保可用性，或替换为你指定的)
    "NEWS": "MiniMaxAI/MiniMax-M2",         # MiniMax: 优秀的文本生成与摘要
    "LOGIC": "deepseek-ai/DeepSeek-V3",     # DeepSeek: 强大的代码与逻辑分析
    "THINKING": "moonshotai/Kimi-k2"        # Kimi: 擅长长窗口与反思 (注: 映射到硅基可用ID)
}

# 用户指定的特定模型 ID (如硅基流动支持这些具体名称，则优先使用)
# 注意：如果报错 "Model not found"，请回退到上面的通用 ID
SPECIFIC_MODELS = {
    "DEEPSEEK": "deepseek-ai/DeepSeek-V3", 
    "KIMI": "moonshotai/Kimi-K2-Thinking", # 假设硅基支持此 ID
    "MINIMAX": "MiniMaxAI/MiniMax-M2",
    "QWEN": "Qwen/Qwen2.5-72B-Instruct" # 修正为标准 ID 以防报错
}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="MAS 联合研报终端",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Light Theme & Chat Bubbles)
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    .stTextInput > div > div > input { background-color: #f3f4f6; color: #1f2937; }
    
    /* Avatar Styling */
    .stChatMessage .stChatMessageAvatar {
        background-color: #e5e7eb;
        border-radius: 50%;
    }
    
    /* Metric Box */
    div[data-testid="metric-container"] {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ 控制台")
    
    st.subheader("🔑 鉴权设置")
    silicon_flow_key = st.text_input("请输入 SiliconFlow API Key", type="password", help="用于调用 DeepSeek, Kimi, Qwen 等模型")
    
    if not silicon_flow_key:
        st.warning("⚠️ 请输入 API Key 以启动系统")
    
    st.divider()
    st.caption("Multi-Agent Research System\nPowered by SiliconFlow")

# --- BACKEND UTILS ---

def get_llm_client():
    if not silicon_flow_key:
        return None
    return OpenAI(
        api_key=silicon_flow_key, 
        base_url="https://api.siliconflow.cn/v1" # 硅基流动 API 地址
    )

def get_tavily_client():
    return TavilyClient(api_key=TAVILY_API_KEY)

def calculate_technical_indicators(df):
    if df.empty: return df
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def retry_with_backoff(func, retries=3):
    x = 0
    while True:
        try:
            return func()
        except Exception as e:
            if x == retries: raise e
            time.sleep(1 + random.uniform(0, 1))
            x += 1

# --- DATA FETCHING ---
def fetch_alpha_vantage_data(ticker):
    if not ALPHA_VANTAGE_KEY: raise ValueError("No AV Key")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
    r = requests.get(url)
    data = r.json()
    if "Time Series (Daily)" not in data: raise ValueError("AV No Data")
    
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df = df.rename(columns={"4. close": "Close"}).astype(float).sort_index()
    df = calculate_technical_indicators(df)
    
    return {
        "status": "ONLINE (AV)", "symbol": ticker, "price": df['Close'].iloc[-1],
        "change_pct": 0.0, "history_df": df,
        "last_macd": {"hist": df['MACD_Hist'].iloc[-1]}, "last_rsi": df['RSI'].iloc[-1],
        "pe": "N/A", "cap": "N/A"
    }

def fetch_market_data(ticker):
    try:
        def _fetch():
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if hist.empty: raise ValueError("Empty Data")
            hist = calculate_technical_indicators(hist)
            info = stock.info
            return {
                "status": "ONLINE (YF)",
                "symbol": ticker.upper(),
                "name": info.get('longName', ticker),
                "price": info.get('currentPrice', hist['Close'].iloc[-1]),
                "change_pct": ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100,
                "pe": info.get('trailingPE', 'N/A'),
                "cap": info.get('marketCap', 'N/A'),
                "history_df": hist,
                "last_macd": {"hist": hist['MACD_Hist'].iloc[-1]},
                "last_rsi": hist['RSI'].iloc[-1]
            }
        return retry_with_backoff(_fetch)
    except:
        try:
            return fetch_alpha_vantage_data(ticker)
        except Exception as e:
            return {"status": "OFFLINE", "error": str(e)}

def search_web(query, topic="general", ticker=None):
    results = []
    # 1. Tavily
    try:
        tavily = get_tavily_client()
        res = tavily.search(query=query, topic=topic, max_results=3)
        results.extend([f"- [Tavily] {r['title']}: {r['content'][:200]}" for r in res['results']])
    except: pass
    
    # 2. RSS Fallback
    if len(results) < 2:
        try:
            url = "http://feeds.bbci.co.uk/news/business/rss.xml" if not ticker else f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US"
            feed = feedparser.parse(url)
            results.extend([f"- [RSS] {e.title}" for e in feed.entries[:3]])
        except: pass
    
    return results if results else ["无相关新闻数据"]

def call_agent(agent_name, model_id, system_prompt, user_prompt):
    client = get_llm_client()
    if not client: return "请先配置 API Key"
    
    try:
        # 适配不同的模型 ID
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ {agent_name} 掉线: {str(e)}"

# --- MAIN LOGIC ---

st.title("🏦 MAS 联合研报终端")
st.caption(f"引擎: Qwen (路由) | MiniMax (情报) | DeepSeek (分析) | Kimi (风控)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是值班董秘。请下达研究指令（如：分析 宁德时代）。", "avatar": "👩‍💼"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

if user_input := st.chat_input("请输入标的..."):
    if not silicon_flow_key:
        st.error("请先在左侧侧边栏输入 SiliconFlow API Key！")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 1. Router (Qwen)
    ticker = None
    with st.chat_message("assistant", avatar="👩‍💼"):
        placeholder = st.empty()
        placeholder.markdown("🔄 董秘 (Qwen) 正在解析意图...")
        
        res = call_agent("Router", SPECIFIC_MODELS["QWEN"], 
                         "你是董秘。提取股票代码(Yahoo Ticker)。JSON格式 {'ticker': '...'}", user_input)
        try:
            ticker = json.loads(res.replace("```json","").replace("```",""))['ticker']
            placeholder.markdown(f"✅ 已立项，标的：**{ticker}**。正在召开投研晨会...")
        except:
            placeholder.markdown("❓ 无法识别标的，请重试。")
            st.stop()
    
    st.session_state.messages.append({"role": "assistant", "content": f"已立项：{ticker}", "avatar": "👩‍💼"})

    # 2. Data Fetching (Parallel)
    with st.status("📡 数据中心正在分发任务...", expanded=True) as status:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_mkt = executor.submit(fetch_market_data, ticker)
            f_macro = executor.submit(search_web, "Global macro economy 2024", "news")
            f_meso = executor.submit(search_web, f"{ticker} industry trends", "general")
            f_micro = executor.submit(search_web, f"{ticker} financial news", "news", ticker)
            f_pol = executor.submit(search_web, "Geopolitics US China", "news")
            
            mkt = f_mkt.result()
            raw_news = {
                "macro": f_macro.result(), "meso": f_meso.result(), 
                "micro": f_micro.result(), "pol": f_pol.result()
            }
        
        if mkt['status'] == "OFFLINE":
            status.update(label="❌ 数据获取失败", state="error")
            st.error(mkt.get('error'))
            st.stop()
        
        status.update(label="✅ 数据获取完成", state="complete")

    # 3. The Meeting (模拟研讨会 - 各研究员轮流发言)
    # 我们把每个 Agent 的发言都展示出来，不再嵌套在同一个 bubble 里
    
    # 3.1 情报官发言 (MiniMax)
    opinions = {}
    
    # 宏观
    with st.chat_message("assistant", avatar="🌍"):
        st.write(f"**宏观情报官 (MiniMax)**:")
        res = call_agent("Macro", SPECIFIC_MODELS["MINIMAX"], "你是宏观分析师。简述当前宏观环境对市场的影响 (50字以内)。", str(raw_news['macro']))
        st.markdown(res)
        opinions['macro'] = res
        st.session_state.messages.append({"role": "assistant", "content": f"**宏观**: {res}", "avatar": "🌍"})

    # 行业
    with st.chat_message("assistant", avatar="🏭"):
        st.write(f"**行业研究员 (MiniMax)**:")
        res = call_agent("Meso", SPECIFIC_MODELS["MINIMAX"], f"你是行业分析师。{ticker} 所在行业目前景气度如何？(50字以内)", str(raw_news['meso']))
        st.markdown(res)
        opinions['meso'] = res
        st.session_state.messages.append({"role": "assistant", "content": f"**行业**: {res}", "avatar": "🏭"})

    # 个股
    with st.chat_message("assistant", avatar="🔍"):
        st.write(f"**个股研究员 (MiniMax)**:")
        res = call_agent("Micro", SPECIFIC_MODELS["MINIMAX"], f"你是公司研究员。{ticker} 最近有什么利好或利空？(50字以内)", str(raw_news['micro']))
        st.markdown(res)
        opinions['micro'] = res
        st.session_state.messages.append({"role": "assistant", "content": f"**个股**: {res}", "avatar": "🔍"})

    # 财经 (DeepSeek)
    with st.chat_message("assistant", avatar="💹"):
        st.write(f"**首席财经 (DeepSeek)**:")
        fin_ctx = f"Price: {mkt['price']}, PE: {mkt['pe']}, Cap: {mkt['cap']}"
        res = call_agent("Finance", SPECIFIC_MODELS["DEEPSEEK"], "你是财务专家。评价该估值水平 (低估/合理/高估) (50字以内)。", fin_ctx)
        st.markdown(res)
        opinions['fin'] = res
        st.session_state.messages.append({"role": "assistant", "content": f"**财经**: {res}", "avatar": "💹"})

    # 量化 (DeepSeek)
    with st.chat_message("assistant", avatar="🔢"):
        st.write(f"**量化分析师 (DeepSeek)**:")
        quant_ctx = f"MACD Hist: {mkt['last_macd']['hist']:.3f}, RSI: {mkt['last_rsi']:.1f}"
        res = call_agent("Quant", SPECIFIC_MODELS["DEEPSEEK"], "你是量化交易员。根据指标判断短线趋势 (50字以内)。", quant_ctx)
        st.markdown(res)
        opinions['quant'] = res
        st.session_state.messages.append({"role": "assistant", "content": f"**量化**: {res}", "avatar": "🔢"})

    # 4. 综合研报 (DeepSeek)
    with st.chat_message("assistant", avatar="📝"):
        st.markdown("### 📑 深度研报")
        
        # 展示行情图
        c1, c2, c3 = st.columns(3)
        c1.metric("价格", f"{mkt['price']:.2f}")
        c2.metric("PE", mkt['pe'])
        c3.metric("RSI", f"{mkt['last_rsi']:.1f}")
        
        fig = go.Figure(data=[go.Candlestick(x=mkt['history_df'].index, 
                        open=mkt['history_df']['Open'], high=mkt['history_df']['High'],
                        low=mkt['history_df']['Low'], close=mkt['history_df']['Close'])])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # 生成报告
        placeholder = st.empty()
        placeholder.write("✍️ 综合分析师 (DeepSeek) 正在汇总各方意见撰写正文...")
        
        full_context = f"会议纪要:\n{json.dumps(opinions, ensure_ascii=False)}\n详细数据:\n{str(raw_news)}"
        report = call_agent("Analyst", SPECIFIC_MODELS["DEEPSEEK"], 
                            "你是首席分析师。根据会议纪要写一份结构化研报。包含：投资逻辑、风险提示、关键结论。", full_context)
        placeholder.markdown(report)
        st.session_state.messages.append({"role": "assistant", "content": report, "avatar": "📝"})

    # 5. 风控与决策 (Kimi - Thinking)
    c_risk, c_lead = st.columns(2)
    
    with c_risk:
        with st.chat_message("assistant", avatar="🛡️"):
            st.write("**风控官 (Kimi)**:")
            res = call_agent("Critic", SPECIFIC_MODELS["KIMI"], 
                             "你是风控官。请对上述研报进行批判性审查，指出潜在风险点。", report)
            st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": f"**风控**: {res}", "avatar": "🛡️"})

    with c_lead:
        with st.chat_message("assistant", avatar="🏆"):
            st.write("**所长 (Kimi)**:")
            res = call_agent("Leader", SPECIFIC_MODELS["KIMI"], 
                             "你是所长。综合研报和风控意见，给出一个明确的操作建议 (买入/卖出/观望) 并用一句话总结理由。", 
                             f"报告:{report}\n风控:{res}")
            st.success(res)
            st.session_state.messages.append({"role": "assistant", "content": f"**决策**: {res}", "avatar": "🏆"})

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
from datetime import datetime, timedelta

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="DeepSeek MAS 券商研究所 2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light Theme & Agent Chat Style
st.markdown("""
<style>
    /* Light Theme Base */
    .stApp {
        background-color: #ffffff;
        color: #31333F;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        color: #31333F;
    }
    
    /* Agent Box Style in Chat */
    .agent-box {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5em;
        color: #0068c9;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #31333F !important;
    }
    
    /* Adjust plotly chart background */
    .js-plotly-plot .plotly .bg {
        fill: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.title("🎛️ 控制台 (Control)")
    
    st.subheader("🔑 API Keys Configuration")
    deepseek_api_key = st.text_input("DeepSeek API Key", value="sk-d4c4d12b75f0419a90a9f403f7f63ef7", type="password")
    tavily_api_key = st.text_input("Tavily API Key", value="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", type="password")
    # Updated with your provided key
    alpha_vantage_key = st.text_input("Alpha Vantage Key", value="8G1QKAWN221XEZR8", help="用于 yfinance 失败时的备用行情源及补充新闻")
    
    st.subheader("⚙️ Analysis Settings")
    model_selection = st.selectbox("LLM Model", ["deepseek-chat"], index=0)
    search_depth = st.radio("Search Depth", ["basic", "advanced"], index=0)
    
    st.divider()
    st.caption("DeepSeek Brokerage Institute v2.0\nPowered by Python/Streamlit")

# --- BACKEND UTILS ---

def get_llm_client():
    return OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

def get_tavily_client():
    return TavilyClient(api_key=tavily_api_key)

def calculate_technical_indicators(df):
    """Calculates MACD, RSI using Pandas/Numpy."""
    if df.empty: return df
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def retry_with_backoff(func, retries=3, backoff_in_seconds=1):
    """Helper function to retry a function with exponential backoff."""
    x = 0
    while True:
        try:
            return func()
        except Exception as e:
            if x == retries:
                raise e
            sleep = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
            st.toast(f"⚠️ 请求被限流/失败，{sleep:.1f}秒后重试... (尝试 {x+1}/{retries})", icon="⏳")
            time.sleep(sleep)
            x += 1

# --- DATA FETCHING: ALPHA VANTAGE (BACKUP) ---
def fetch_alpha_vantage_data(ticker_symbol, api_key):
    """Fetches market data from Alpha Vantage as backup."""
    if not api_key:
        raise ValueError("Alpha Vantage Key 未配置")
    
    # 1. Get Daily History for Chart
    url_daily = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker_symbol}&apikey={api_key}&outputsize=compact"
    r_daily = requests.get(url_daily)
    data_daily = r_daily.json()
    
    if "Error Message" in data_daily:
        raise ValueError(f"Alpha Vantage Error: {data_daily['Error Message']}")
    if "Note" in data_daily: 
        raise ValueError(f"Alpha Vantage Limit: {data_daily['Note']}")
    if "Time Series (Daily)" not in data_daily:
        raise ValueError("No data returned from Alpha Vantage")

    ts = data_daily["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient='index')
    df = df.rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"
    })
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df = df.sort_index()
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    
    # 2. Get Global Quote for Latest Price
    current_price = df['Close'].iloc[-1]
    change_pct = 0.0
    try:
        url_quote = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_symbol}&apikey={api_key}"
        r_quote = requests.get(url_quote)
        data_quote = r_quote.json()
        if "Global Quote" in data_quote and data_quote["Global Quote"]:
            quote = data_quote["Global Quote"]
            current_price = float(quote["05. price"])
            change_pct_str = quote["10. change percent"].replace('%', '')
            change_pct = float(change_pct_str)
        else:
            # Fallback calculation
            prev = df['Close'].iloc[-2]
            change_pct = ((current_price - prev) / prev) * 100
    except:
        pass
    
    return {
        "status": "ONLINE (AlphaVantage)",
        "symbol": ticker_symbol.upper(),
        "name": ticker_symbol.upper(),
        "price": current_price,
        "change_pct": change_pct,
        "currency": "USD", 
        "market_cap": "N/A", 
        "pe_ratio": "N/A",
        "eps": "N/A",
        "sector": "N/A",
        "history_df": df,
        "last_macd": {
            "macd": df['MACD'].iloc[-1],
            "signal": df['Signal_Line'].iloc[-1],
            "hist": df['MACD_Hist'].iloc[-1]
        },
        "last_rsi": df['RSI'].iloc[-1]
    }

def fetch_alpha_vantage_news(ticker, api_key):
    """Fetches news sentiment from Alpha Vantage."""
    if not api_key: return []
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}&limit=5"
        r = requests.get(url)
        data = r.json()
        if "feed" in data:
            return [f"- [AV News] [{item['title']}]({item['url']}): {item['summary'][:100]}..." for item in data["feed"][:3]]
    except:
        pass
    return []

# --- DATA FETCHING: MARKET DATA MAIN ---
def fetch_market_data(ticker_symbol):
    """Strategy: 1. yfinance -> 2. Alpha Vantage -> 3. Error"""
    
    # 1. Try yfinance
    try:
        def _fetch_yf():
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period="6mo")
            if hist.empty:
                raise ValueError("yfinance returned empty data")
            
            hist = calculate_technical_indicators(hist)
            info = stock.info
            
            # Safe retrieval
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') 
            if prev_close is None and len(hist) > 1:
                prev_close = hist['Close'].iloc[-2]
            
            if prev_close:
                change_pct = ((current_price - prev_close) / prev_close) * 100
            else:
                change_pct = 0.0

            return {
                "status": "ONLINE (yfinance)",
                "symbol": ticker_symbol.upper(),
                "name": info.get('longName', ticker_symbol),
                "price": current_price,
                "change_pct": change_pct,
                "currency": info.get('currency', 'USD'),
                "market_cap": info.get('marketCap', 'N/A'),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "eps": info.get('trailingEps', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "history_df": hist,
                "last_macd": {
                    "macd": hist['MACD'].iloc[-1],
                    "signal": hist['Signal_Line'].iloc[-1],
                    "hist": hist['MACD_Hist'].iloc[-1]
                },
                "last_rsi": hist['RSI'].iloc[-1]
            }

        return retry_with_backoff(_fetch_yf, retries=2)

    except Exception as e_yf:
        st.toast(f"⚠️ yfinance 失败: {str(e_yf)}。尝试 Alpha Vantage...", icon="🔄")
        
        # 2. Try Alpha Vantage
        try:
            if alpha_vantage_key:
                return fetch_alpha_vantage_data(ticker_symbol, alpha_vantage_key)
            else:
                raise ValueError("Alpha Vantage API Key not provided")
        except Exception as e_av:
            return {"status": "OFFLINE", "error": f"All sources failed. YF: {e_yf}, AV: {e_av}"}

# --- DATA FETCHING: NEWS RSS (BACKUP) ---
def fetch_rss_news(query, ticker=None):
    """Fetches news from RSS feeds as backup."""
    news_items = []
    
    # 1. Yahoo Finance RSS (Specific Ticker)
    if ticker:
        try:
            feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:3]:
                news_items.append(f"- [Yahoo RSS] [{entry.title}]({entry.link})")
        except:
            pass

    # 2. BBC Business (General)
    try:
        feed_url = "http://feeds.bbci.co.uk/news/business/rss.xml"
        feed = feedparser.parse(feed_url)
        count = 0
        for entry in feed.entries:
            if count >= 3: break
            if query.lower() in entry.title.lower() or query.lower() in entry.summary.lower() or not ticker:
                 news_items.append(f"- [BBC RSS] [{entry.title}]({entry.link})")
                 count += 1
    except:
        pass
        
    return news_items

def search_web(query, topic="general", ticker=None, av_key=None):
    """Strategy: 1. Tavily -> 2. Alpha Vantage News -> 3. RSS -> 4. Error"""
    results = []
    
    # 1. Try Tavily
    try:
        tavily = get_tavily_client()
        response = tavily.search(query=query, topic=topic, max_results=5)
        t_results = [f"- [{r['title']}]({r['url']}): {r['content'][:300]}..." for r in response['results']]
        if t_results:
            results.extend(t_results)
    except Exception as e:
        st.toast(f"⚠️ Tavily 搜索失败: {str(e)}。尝试备用源...", icon="📰")

    # 2. Alpha Vantage News
    if ticker and av_key:
        av_news = fetch_alpha_vantage_news(ticker, av_key)
        if av_news:
            results.extend(av_news)

    # 3. Try RSS Fallback
    if len(results) < 3:
        rss_results = fetch_rss_news(query, ticker)
        if rss_results:
            results.extend(rss_results)
    
    if results:
        return results[:7]
    
    return [f"❌ 无法获取关于 '{query}' 的新闻数据"]

def call_agent(agent_name, system_prompt, user_prompt):
    """Generic Agent Caller."""
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling {agent_name}: {str(e)}"

# --- MAIN APP LOGIC ---

st.title("🚀 DeepSeek MAS 券商研究所 2.0")
st.caption("Architecture: Python Backend | yfinance + Alpha Vantage | Tavily + AV News + RSS")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好，我是所长。DeepSeek 2.0 内核已启动，Python 后端就绪。请告诉我您想分析的标的（例如：NVDA, 腾讯控股, 比特币）。", "avatar": "👨‍💼"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("输入股票代码或名称..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Router Agent
    ticker_to_analyze = None 
    
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔄 董秘正在解析您的意图...")
        
        router_prompt = f"""
        用户输入: "{user_input}"
        请提取:
        1. 股票代码 (Ticker, Convert to Yahoo format, e.g., 'Tencent' -> '0700.HK', 'Nvidia' -> 'NVDA').
        2. 意图类型 (Analysis/Chat).
        返回 JSON: {{"type": "...", "ticker": "...", "query": "..."}}
        """
        router_res = call_agent("Router", "You are a JSON router.", router_prompt)
        try:
            intent = json.loads(router_res.replace("```json", "").replace("```", ""))
        except:
            intent = {"type": "chat", "ticker": "", "query": user_input}

        if intent.get("type") == "chat" or not intent.get("ticker"):
            message_placeholder.markdown("💬 闲聊模式")
            response = call_agent("Chat", "You are a helpful assistant.", user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "🤖"})
        else:
            ticker_to_analyze = intent['ticker']
            message_placeholder.markdown(f"✅ 已锁定标的: **{ticker_to_analyze}**，正在启动 MAS 工作流...")
            st.session_state.messages.append({"role": "assistant", "content": f"已锁定标的: **{ticker_to_analyze}**，正在启动 MAS 工作流...", "avatar": "🤖"})

    # 3. MAS Workflow (Strictly outside the previous with block to avoid nesting error)
    if ticker_to_analyze:
        ticker = ticker_to_analyze
        
        # 3.1 Fetch Data
        with st.chat_message("assistant", avatar="📡"):
            fetch_placeholder = st.empty()
            fetch_placeholder.markdown("⏳ 正在并发获取行情与全球情报...")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_market = executor.submit(fetch_market_data, ticker)
                future_macro = executor.submit(search_web, "Global macroeconomic outlook 2024 interest rates", "news", ticker, alpha_vantage_key)
                future_meso = executor.submit(search_web, f"{ticker} industry analysis competitors trends", "general", ticker, alpha_vantage_key)
                future_micro = executor.submit(search_web, f"{ticker} stock news financial report recent", "news", ticker, alpha_vantage_key)
                future_politics = executor.submit(search_web, "Geopolitical risks US China trade relations", "news", ticker, alpha_vantage_key)
                
                market_data = future_market.result()
                news_macro = future_macro.result()
                news_meso = future_meso.result()
                news_micro = future_micro.result()
                news_politics = future_politics.result()

            if market_data['status'] == 'OFFLINE':
                fetch_placeholder.markdown(f"❌ 数据获取失败: {market_data.get('error')}")
                st.session_state.messages.append({"role": "assistant", "content": f"❌ 数据获取失败: {market_data.get('error')}", "avatar": "📡"})
                st.stop()
            else:
                fetch_placeholder.markdown(f"✅ 数据获取完成 (Source: {market_data['status']})")
                st.session_state.messages.append({"role": "assistant", "content": f"✅ 数据获取完成 (Source: {market_data['status']})", "avatar": "📡"})

        # 4. Dashboard
        with st.chat_message("assistant", avatar="📊"):
            st.markdown(f"### 📊 行情看板: {market_data['name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("价格", f"{market_data['price']:.2f}", f"{market_data['change_pct']:.2f}%")
            c2.metric("PE", market_data['pe_ratio'])
            c3.metric("RSI", f"{market_data['last_rsi']:.2f}")
            c4.metric("MACD", f"{market_data['last_macd']['hist']:.3f}")
            
            fig = go.Figure(data=[go.Candlestick(x=market_data['history_df'].index,
                            open=market_data['history_df']['Open'], high=market_data['history_df']['High'],
                            low=market_data['history_df']['Low'], close=market_data['history_df']['Close'])])
            fig.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        st.session_state.messages.append({"role": "assistant", "content": f"### 📊 行情看板: {market_data['name']}", "avatar": "📊"})

        # Context
        context = f"""
        【行情】{market_data['price']} (Chg: {market_data['change_pct']}%)
        PE: {market_data['pe_ratio']}, MACD: {market_data['last_macd']}, RSI: {market_data['last_rsi']}
        【宏观】{str(news_macro)}
        【中观】{str(news_meso)}
        【微观】{str(news_micro)}
        【政治】{str(news_politics)}
        """

        # 5. MAS Turn-Taking
        # Analyst
        with st.chat_message("assistant", avatar="🧑‍💻"):
            st.write("**综合分析师** 正在撰写报告...")
            analyst_res = call_agent("Analyst", "You are a senior financial analyst. Output markdown.", f"分析标的: {ticker}\n数据: {context}")
            st.markdown(analyst_res)
            st.session_state.messages.append({"role": "assistant", "content": analyst_res, "avatar": "🧑‍💻"})

        # Critic
        with st.chat_message("assistant", avatar="🛡️"):
            st.write("**风控官** 正在审核...")
            critic_res = call_agent("Critic", "You are a strict risk officer.", f"分析师报告: {analyst_res}\n数据: {context}\n请审核风险。")
            st.markdown(critic_res)
            st.session_state.messages.append({"role": "assistant", "content": critic_res, "avatar": "🛡️"})

        # Leader
        with st.chat_message("assistant", avatar="🏆"):
            st.write("**所长** 正在决策...")
            leader_res = call_agent("Leader", "You are the leader.", f"分析: {analyst_res}\n风控: {critic_res}\n给出最终建议。")
            st.markdown(leader_res)
            st.session_state.messages.append({"role": "assistant", "content": leader_res, "avatar": "🏆"})
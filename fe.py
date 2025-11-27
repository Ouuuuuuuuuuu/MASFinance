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
import re

# --- HARDCODED KEYS (Hidden from UI) ---
TAVILY_API_KEY = "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"
ALPHA_VANTAGE_KEY = "8G1QKAWN221XEZR8"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="MAS 联合研报终端 v3.6",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Compact & Light) ---
st.markdown("""
<style>
    /* Global Font & Colors */
    .stApp { background-color: #ffffff; color: #333333; font-family: 'Source Sans Pro', sans-serif; }
    
    /* Chat Message Styling */
    .stChatMessage { padding: 1rem; }
    .stChatMessage .stChatMessageAvatar { background-color: #f0f2f6; border-radius: 50%; }
    
    /* Compact Headers in Chat */
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3 {
        font-size: 1.1em !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        color: #1f2937;
    }
    .stChatMessage p { font-size: 0.95em !important; line-height: 1.6; }
    
    /* Thinking Box */
    .thinking-box {
        font-size: 0.85em;
        color: #6b7280;
        border-left: 3px solid #e5e7eb;
        padding-left: 10px;
        margin: 5px 0;
        font-style: italic;
        background: #f9fafb;
    }
    
    /* Input Field */
    .stTextInput > div > div > input { background-color: #f9fafb; color: #1f2937; border: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "首席研究员就位。请下达调研指令（如：分析 比亚迪）。", "avatar": "👨‍🔬"}]
if "process_status" not in st.session_state:
    st.session_state.process_status = "IDLE"
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "market_data" not in st.session_state:
    st.session_state.market_data = None
if "raw_news" not in st.session_state:
    st.session_state.raw_news = {}
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0
if "last_rework_field" not in st.session_state:
    st.session_state.last_rework_field = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.subheader("🔑 鉴权设置")
    
    default_sf_key = st.secrets.get("SILICON_FLOW_KEY", "")
    silicon_flow_key = st.text_input("SiliconFlow Key", value=default_sf_key, type="password", help="请输入您的硅基流动 API Key")

    if not silicon_flow_key:
        st.warning("⚠️ 请输入 SiliconFlow API Key 以启动大模型")
    else:
        st.success("✅ 系统已就绪")
    
    st.divider()
    if st.button("🔄 重置系统状态"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- UTILS ---

def extract_json_from_markdown(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None

def get_llm_client():
    if not silicon_flow_key: return None
    return OpenAI(api_key=silicon_flow_key, base_url="https://api.siliconflow.cn/v1")

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

def fetch_from_alphavantage(ticker):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "Time Series (Daily)" not in data: return None
        
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df = df.rename(columns={"4. close": "Close", "1. open": "Open", "2. high": "High", "3. low": "Low"})
        df = df.astype(float).sort_index()
        df = calculate_technical_indicators(df)
        
        return {
            "status": "ONLINE (AV)",
            "symbol": ticker.upper(),
            "name": ticker,
            "price": df['Close'].iloc[-1],
            "change_pct": 0.0,
            "pe": "N/A",
            "cap": "N/A",
            "history_df": df,
            "last_macd": {"hist": df['MACD_Hist'].iloc[-1]},
            "last_rsi": df['RSI'].iloc[-1]
        }
    except:
        return None

def fetch_from_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty: return None
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
    except:
        return None

def fetch_market_data(ticker):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_yf = executor.submit(fetch_from_yfinance, ticker)
        future_av = executor.submit(fetch_from_alphavantage, ticker)
        
        yf_data = future_yf.result()
        if yf_data: return yf_data
        av_data = future_av.result()
        if av_data: return av_data
        
    return {"status": "OFFLINE", "error": "Market data unavailable"}

def search_web(query, topic="general"):
    try:
        tavily = get_tavily_client()
        res = tavily.search(query=query, topic=topic, max_results=5)
        # Ensure result snippet is not too long to save tokens
        return [f"- {r['title']}: {r['content'][:200]}" for r in res['results']]
    except Exception as e:
        return [f"Search Error: {str(e)}"]

def call_agent(agent_name, model_id, system_prompt, user_prompt, thinking_needed=False):
    client = get_llm_client()
    if not client: return "API Key Missing", ""
    
    # 强制注入中文指令
    final_sys_prompt = system_prompt + "\nIMPORTANT: 请务必使用中文简体 (Chinese Simplified) 回复。"
    
    if thinking_needed:
        final_sys_prompt += "\nLet's think step by step. First output your thinking process wrapped in <thinking>...</thinking>, then output your final response in Chinese."

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": final_sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        content = response.choices[0].message.content
        
        thinking = ""
        if "<thinking>" in content:
            match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
        
        return content, thinking
    except Exception as e:
        return f"⚠️ {agent_name} Error: {str(e)}", ""

# --- MODEL MAP ---
SPECIFIC_MODELS = {
    "DEEPSEEK": "deepseek-ai/DeepSeek-V3", 
    "KIMI": "moonshotai/Kimi-K2-Thinking",
    "MINIMAX": "MiniMaxAI/MiniMax-M2",
    "QWEN": "Qwen/Qwen2.5-72B-Instruct"
}

# --- MAIN UI ---

st.title("🏦 MAS 联合研报终端 v3.6")
st.caption("Powered by SiliconFlow Hybrid Models")

# 1. History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("thinking"):
            with st.expander("🧠 思考过程", expanded=False):
                st.markdown(f"_{msg['thinking']}_")

# 2. Input
if user_input := st.chat_input("请输入标的..."):
    if not silicon_flow_key:
        st.error("请配置 API Key")
        st.stop()

    st.session_state.ticker = None
    st.session_state.market_data = None
    st.session_state.raw_news = {}
    st.session_state.retry_count = 0
    st.session_state.final_report = None
    st.session_state.last_rework_field = None
    
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Router
    with st.chat_message("assistant", avatar="👩‍💼"):
        st.write("🔍 董秘正在核实...")
        # Double Search for better context
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f1 = executor.submit(search_web, f"{user_input} 股票代码", "general")
            f2 = executor.submit(search_web, f"{user_input} stock ticker", "general")
            search_res = f1.result() + f2.result()
        
        router_prompt = f"""
        用户输入: "{user_input}"
        搜索线索: {json.dumps(search_res, ensure_ascii=False)}
        请提取准确的 Yahoo Finance Ticker。
        规则：A股(6位数字+.SS/.SZ), 港股(4位数字+.HK), 美股(字母)。
        只返回JSON: {{'ticker': '...'}}
        """
        res, _ = call_agent("Router", SPECIFIC_MODELS["QWEN"], "你是董秘。", router_prompt)
        json_data = extract_json_from_markdown(res)
        
        if json_data and 'ticker' in json_data:
            st.session_state.ticker = json_data['ticker']
            st.markdown(f"✅ 确认标的：**{st.session_state.ticker}**")
            st.session_state.process_status = "ANALYZING"
            st.rerun()
        else:
            st.error(f"无法识别代码: {res}")
            st.stop()

# 3. Execution
if st.session_state.process_status == "ANALYZING" and st.session_state.ticker:
    ticker = st.session_state.ticker
    
    # A. Data Fetching
    if not st.session_state.market_data:
        with st.status("📡 全网情报搜集...", expanded=True) as status:
            mkt = fetch_market_data(ticker)
            st.session_state.market_data = mkt
            
            # 泛化搜索关键词
            queries = {
                "macro": "全球宏观经济新闻 市场趋势 2024",
                "meso": f"{ticker} 行业分析 竞争对手 市场份额",
                "micro": f"{ticker} 最新新闻 财报分析 机构评级",
                "pol": "国际地缘政治 贸易政策 风险"
            }
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {k: executor.submit(search_web, v, "news" if k != "meso" else "general") for k, v in queries.items()}
                for k, f in futures.items():
                    st.session_state.raw_news[k] = f.result()
            
            status.update(label="✅ 情报就绪", state="complete")
    
    # B. Meeting
    mkt = st.session_state.market_data
    news = st.session_state.raw_news
    opinions = {}
    
    st.divider()
    
    # Market Board
    if mkt and mkt['status'] != "OFFLINE":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("价格", f"{mkt['price']:.2f}", f"{mkt['change_pct']:.2f}%")
        c2.metric("PE", mkt.get('pe', 'N/A'))
        c3.metric("RSI", f"{mkt.get('last_rsi', 0):.1f}")
        c4.metric("MACD", f"{mkt.get('last_macd', {}).get('hist', 0):.3f}")
        
        if 'history_df' in mkt:
            fig = go.Figure(data=[go.Candlestick(x=mkt['history_df'].index,
                            open=mkt['history_df']['Open'], high=mkt['history_df']['High'],
                            low=mkt['history_df']['Low'], close=mkt['history_df']['Close'])])
            fig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
    
    # Agent Meeting
    round_num = st.session_state.retry_count + 1
    st.markdown(f"#### 🗣️ 投研会议 (Round {round_num})")
    
    if st.session_state.last_rework_field:
        st.info(f"💡 本轮针对 **{st.session_state.last_rework_field}** 进行了补充调查。")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            # Macro
            prompt = "简述宏观环境。"
            res, _ = call_agent("Macro", SPECIFIC_MODELS["MINIMAX"], "你是宏观分析师。", f"{prompt}\n情报:{str(news['macro'])}")
            st.markdown(f"**🌍 宏观**: {res}")
            opinions['macro'] = res
            
            # Micro
            res, _ = call_agent("Micro", SPECIFIC_MODELS["MINIMAX"], f"分析 {ticker} 个股。", f"情报:{str(news['micro'])}")
            st.markdown(f"**🔍 个股**: {res}")
            opinions['micro'] = res

        with col2:
            # Meso
            res, _ = call_agent("Meso", SPECIFIC_MODELS["MINIMAX"], f"分析 {ticker} 行业。", f"情报:{str(news['meso'])}")
            st.markdown(f"**🏭 行业**: {res}")
            opinions['meso'] = res
            
            # Quant
            if mkt['status'] != "OFFLINE":
                quant_ctx = f"Price:{mkt['price']}, PE:{mkt['pe']}, RSI:{mkt.get('last_rsi')}"
                res, _ = call_agent("Finance", SPECIFIC_MODELS["DEEPSEEK"], "评价估值与技术面。", quant_ctx)
                st.markdown(f"**💹 量化**: {res}")
                opinions['quant'] = res
            else:
                quant_ctx = "Market Data Offline"

    # C. Drafting
    with st.chat_message("assistant", avatar="📝"):
        st.write("✍️ **综合分析师** 正在撰写研报...")
        full_ctx = f"Opinions:{json.dumps(opinions, ensure_ascii=False)}\nMarket:{quant_ctx}"
        report_draft, _ = call_agent("Analyst", SPECIFIC_MODELS["DEEPSEEK"], 
                            "写一份结构化研报(Markdown)。包含：核心逻辑、风险提示、结论。", full_ctx)
        st.markdown(report_draft)

    # D. Chief Review
    with st.chat_message("assistant", avatar="👨‍🔬"):
        st.write("🕵️ **首席研究员** 正在审核...")
        
        is_final_round = st.session_state.retry_count >= 1
        
        review_prompt = f"""
        你是首席研究员。审查研报。
        
        当前是第 {round_num} 轮审核。
        
        1. 如果信息严重缺失且还可以返工（当前不是最后一轮），请输出指令：REWORK: [MACRO/MESO/MICRO]。
        2. 如果信息足够，或者已经是最后一轮（Round 2），请必须给出最终结论。
        
        研报内容:
        {report_draft}
        """
        
        review_res, thinking = call_agent("Chief", SPECIFIC_MODELS["KIMI"], review_prompt, "开始审核", thinking_needed=True)
        
        if thinking:
            with st.expander("🧠 思考过程", expanded=True):
                st.markdown(f"_{thinking}_")
        
        # Logic
        if "REWORK:" in review_res and not is_final_round:
            match = re.search(r"REWORK:\s*(\w+)", review_res)
            field = match.group(1).lower() if match else "micro"
            if field not in ["macro", "meso", "micro"]: field = "micro"
            
            st.session_state.last_rework_field = field
            st.warning(f"🚨 驳回：要求补充 **{field}** 领域信息。正在执行...")
            
            new_query = f"{ticker} {field} deep analysis details"
            new_info = search_web(new_query, "general")
            st.session_state.raw_news[field].extend(new_info)
            st.session_state.retry_count += 1
            time.sleep(1)
            st.rerun()
            
        else:
            st.success("✅ 审核通过")
            st.markdown(f"### 🏆 最终决策\n\n{review_res}")
            
            # Save Result
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"### 📑 最终研报 ({ticker})\n\n{report_draft}\n\n---\n**🏆 首席决策**: {review_res}", 
                "avatar": "👨‍🔬", 
                "thinking": thinking
            })
            st.session_state.process_status = "DONE"

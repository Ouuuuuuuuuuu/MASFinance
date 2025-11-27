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

# --- PAGE SETUP ---
st.set_page_config(
    page_title="MAS 联合研报终端 v3.1 (Fix)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    .stTextInput > div > div > input { background-color: #f3f4f6; color: #1f2937; }
    .stChatMessage .stChatMessageAvatar { background-color: #e5e7eb; border-radius: 50%; }
    div[data-testid="metric-container"] { background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 10px; border-radius: 8px; }
    
    .thinking-box {
        font-size: 0.85em;
        color: #6b7280;
        border-left: 3px solid #e5e7eb;
        padding-left: 10px;
        margin-bottom: 10px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION (CRITICAL FIX) ---
# 修复逻辑错误核心：所有跨 Rerun 的数据必须存入 Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "首席研究员就位。请下达调研指令（如：分析 特斯拉）。", "avatar": "👨‍🔬"}]
if "process_status" not in st.session_state:
    st.session_state.process_status = "IDLE" # IDLE, ANALYZING, REVIEWING, DONE
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "market_data" not in st.session_state:
    st.session_state.market_data = None
if "raw_news" not in st.session_state:
    st.session_state.raw_news = {}
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0
if "final_report" not in st.session_state:
    st.session_state.final_report = None

# --- SIDEBAR & SECRETS ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.subheader("🔑 鉴权设置")
    
    # 优先尝试从 st.secrets 读取，如果没配置，则显示输入框
    # 这种方式既安全（GitHub不泄露）又方便（本地测试可以直接填）
    
    default_sf_key = st.secrets.get("SILICON_FLOW_KEY", "")
    silicon_flow_key = st.text_input("SiliconFlow Key", value=default_sf_key, type="password")
    
    default_tavily = st.secrets.get("TAVILY_API_KEY", "")
    tavily_api_key = st.text_input("Tavily Key", value=default_tavily, type="password")
    
    default_av = st.secrets.get("ALPHA_VANTAGE_KEY", "")
    alpha_vantage_key = st.text_input("Alpha Vantage Key", value=default_av, type="password")

    if not silicon_flow_key or not tavily_api_key:
        st.warning("⚠️ 请填入必要的 API Key")
    
    st.divider()
    if st.button("🔄 重置系统状态"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- ROBUST UTILS ---

def extract_json_from_markdown(text):
    """Robust JSON extraction using Regex to find { ... } block."""
    try:
        # 尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        # 正则提取第一个 JSON 对象
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
    return TavilyClient(api_key=tavily_api_key)

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

def fetch_market_data(ticker):
    try:
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
    except:
        return {"status": "OFFLINE", "error": "Market data unavailable"}

def search_web(query, topic="general"):
    try:
        tavily = get_tavily_client()
        res = tavily.search(query=query, topic=topic, max_results=5)
        return [f"- {r['title']}: {r['content'][:300]}" for r in res['results']]
    except Exception as e:
        return [f"Search Error: {str(e)}"]

def call_agent(agent_name, model_id, system_prompt, user_prompt, thinking_needed=False):
    client = get_llm_client()
    if not client: return "API Key Missing", ""
    
    final_sys_prompt = system_prompt
    if thinking_needed:
        final_sys_prompt += "\nIMPORTANT: You MUST first output your internal thinking process wrapped in <thinking>...</thinking> tags, then output your final response."

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
        
        # Parse Thinking
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

# --- MAIN UI LOGIC ---

st.title("🏦 MAS 联合研报终端 v3.1")
st.caption(f"混合模型引擎: Qwen (路由) | MiniMax (情报) | DeepSeek (分析) | Kimi (首席研究)")

# 1. Chat History Rendering
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("thinking"):
            with st.expander("🧠 思考过程 (Thinking Chain)", expanded=False):
                st.markdown(f"_{msg['thinking']}_")

# 2. Input Handler
if user_input := st.chat_input("请输入标的..."):
    if not silicon_flow_key or not tavily_api_key:
        st.error("请先在侧边栏配置 API Key")
        st.stop()

    # Clear previous session data for new query
    st.session_state.ticker = None
    st.session_state.market_data = None
    st.session_state.raw_news = {}
    st.session_state.retry_count = 0
    st.session_state.final_report = None
    
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Router Step
    with st.chat_message("assistant", avatar="👩‍💼"):
        st.write("🔄 董秘正在立项...")
        res, _ = call_agent("Router", SPECIFIC_MODELS["QWEN"], 
                            "提取Yahoo Ticker。返回JSON {'ticker': '...'}", user_input)
        
        json_data = extract_json_from_markdown(res) # Robust JSON Parsing
        
        if json_data and 'ticker' in json_data:
            st.session_state.ticker = json_data['ticker']
            st.markdown(f"✅ 标的确认：**{st.session_state.ticker}**")
            st.session_state.process_status = "ANALYZING"
            st.rerun() # Force rerun to enter analysis phase with state
        else:
            st.error(f"无法识别标的，AI返回：{res}")
            st.stop()

# 3. Analysis Process (Triggered by State)
if st.session_state.process_status == "ANALYZING" and st.session_state.ticker:
    
    ticker = st.session_state.ticker
    
    # --- STEP A: FETCH DATA (Only if missing) ---
    if not st.session_state.market_data:
        with st.status("📡 正在进行全网情报搜集...", expanded=True) as status:
            mkt = fetch_market_data(ticker)
            if mkt['status'] == "OFFLINE":
                st.error("行情数据获取失败")
                st.stop()
            st.session_state.market_data = mkt
            
            queries = {
                "macro": "global macro economy news market trends",
                "meso": f"{ticker} industry competitors market share",
                "micro": f"{ticker} stock news financial reports analysis",
                "pol": "international geopolitics trade war impact"
            }
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {k: executor.submit(search_web, v, "news" if k != "meso" else "general") for k, v in queries.items()}
                for k, f in futures.items():
                    st.session_state.raw_news[k] = f.result()
            
            status.update(label="✅ 初始情报已就绪", state="complete")
    
    # --- STEP B: MEETING (Display Opinions) ---
    mkt = st.session_state.market_data
    news = st.session_state.raw_news
    opinions = {}
    
    st.subheader(f"🗣️ 投研会议 (Round {st.session_state.retry_count + 1})")
    
    # Agents speak
    with st.chat_message("assistant", avatar="🌍"):
        res, _ = call_agent("Macro", SPECIFIC_MODELS["MINIMAX"], "简述宏观环境。", str(news['macro']))
        st.markdown(f"**宏观**: {res}")
        opinions['macro'] = res

    with st.chat_message("assistant", avatar="🏭"):
        res, _ = call_agent("Meso", SPECIFIC_MODELS["MINIMAX"], f"分析 {ticker} 行业。", str(news['meso']))
        st.markdown(f"**行业**: {res}")
        opinions['meso'] = res

    with st.chat_message("assistant", avatar="🔍"):
        res, _ = call_agent("Micro", SPECIFIC_MODELS["MINIMAX"], f"分析 {ticker} 个股。", str(news['micro']))
        st.markdown(f"**个股**: {res}")
        opinions['micro'] = res
    
    with st.chat_message("assistant", avatar="💹"):
        quant_ctx = f"Price:{mkt['price']}, PE:{mkt['pe']}, RSI:{mkt['last_rsi']:.1f}"
        res, _ = call_agent("Finance", SPECIFIC_MODELS["DEEPSEEK"], "评价估值与技术面。", quant_ctx)
        st.markdown(f"**量化**: {res}")
        opinions['quant'] = res

    # --- STEP C: DRAFTING ---
    with st.chat_message("assistant", avatar="📝"):
        st.write("✍️ 正在撰写研报草案...")
        full_ctx = f"Opinions:{json.dumps(opinions, ensure_ascii=False)}\nMarket:{quant_ctx}"
        report_draft, _ = call_agent("Analyst", SPECIFIC_MODELS["DEEPSEEK"], 
                            "写一份结构化研报，包含逻辑、风险和结论。", full_ctx)
        st.markdown(report_draft)

    # --- STEP D: CHIEF REVIEW ---
    with st.chat_message("assistant", avatar="👨‍🔬"):
        st.write("🕵️ **首席研究员 (Kimi)** 正在审核...")
        
        review_prompt = f"""
        你是首席研究员。审查研报。
        1. 若信息严重缺失，输出指令：REWORK: [MACRO/MESO/MICRO]
        2. 若通过，输出最终投资建议。
        研报: {report_draft}
        """
        review_res, thinking = call_agent("Chief", SPECIFIC_MODELS["KIMI"], review_prompt, "开始审核", thinking_needed=True)
        
        if thinking:
            with st.expander("🧠 思考过程", expanded=True):
                st.markdown(f"_{thinking}_")
        
        # Logic for Rework vs Finalize
        if "REWORK:" in review_res and st.session_state.retry_count < 1:
            # REWORK FLOW
            match = re.search(r"REWORK:\s*(\w+)", review_res)
            field = match.group(1).lower() if match else "micro"
            if field not in st.session_state.raw_news: field = "micro"
            
            st.warning(f"🚨 驳回：要求补充 **{field}** 领域信息。正在执行...")
            
            # Supplement Search
            new_query = f"{ticker} {field} deep analysis recent news"
            new_info = search_web(new_query, "general")
            st.session_state.raw_news[field].extend(new_info) # Append Data
            
            st.session_state.retry_count += 1 # Increment Retry
            st.rerun() # Restart analysis with new data
            
        else:
            # SUCCESS FLOW
            st.success("✅ 审核通过")
            st.markdown(f"### 🏆 最终决策\n{review_res}")
            
            # Save final result to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"### 📑 最终研报 ({ticker})\n\n{report_draft}\n\n---\n**🏆 首席决策**: {review_res}", 
                "avatar": "👨‍🔬",
                "thinking": thinking
            })
            st.session_state.process_status = "DONE" # Mark as done

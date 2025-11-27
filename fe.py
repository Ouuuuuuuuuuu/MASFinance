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

# --- MODEL CONFIGURATION ---
MODELS = {
    "ROUTER": "Qwen/Qwen2.5-72B-Instruct",
    "NEWS": "MiniMaxAI/MiniMax-M2",
    "LOGIC": "deepseek-ai/DeepSeek-V3",
    "THINKING": "moonshotai/Kimi-K2-Thinking" 
}

SPECIFIC_MODELS = {
    "DEEPSEEK": "deepseek-ai/DeepSeek-V3", 
    "KIMI": "moonshotai/Kimi-K2-Thinking",
    "MINIMAX": "MiniMaxAI/MiniMax-M2",
    "QWEN": "Qwen/Qwen2.5-72B-Instruct"
}

# --- PAGE SETUP ---
st.set_page_config(
    page_title="MAS 联合研报终端 v3.0",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    .stTextInput > div > div > input { background-color: #f3f4f6; color: #1f2937; }
    .stChatMessage .stChatMessageAvatar { background-color: #e5e7eb; border-radius: 50%; }
    div[data-testid="metric-container"] { background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 10px; border-radius: 8px; }
    
    /* Thinking Process Style */
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

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.subheader("🔑 鉴权设置")
    silicon_flow_key = st.text_input("请输入 SiliconFlow API Key", type="password", help="用于调用模型")
    if not silicon_flow_key:
        st.warning("⚠️ 请输入 API Key 以启动系统")
    st.divider()
    st.caption("Multi-Agent Research System v3.0\nPowered by SiliconFlow")

# --- BACKEND UTILS ---

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
    """Broad search strategy."""
    try:
        tavily = get_tavily_client()
        res = tavily.search(query=query, topic=topic, max_results=5)
        return [f"- {r['title']}: {r['content'][:300]}" for r in res['results']]
    except:
        return ["暂无相关网络搜索数据"]

def call_agent(agent_name, model_id, system_prompt, user_prompt, thinking_needed=False):
    client = get_llm_client()
    if not client: return "请配置 API Key", ""
    
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
            max_tokens=2048 # Increased for thinking
        )
        content = response.choices[0].message.content
        
        # Parse Thinking
        thinking = ""
        if "<thinking>" in content and "</thinking>" in content:
            match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
        
        return content, thinking
    except Exception as e:
        return f"⚠️ {agent_name} Error: {str(e)}", ""

# --- MAIN LOGIC ---

st.title("🏦 MAS 联合研报终端 v3.0")
st.caption(f"混合模型引擎: Qwen (路由) | MiniMax (情报) | DeepSeek (分析) | Kimi (首席研究)")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "首席研究员就位。请下达调研指令（如：分析 特斯拉）。", "avatar": "👨‍🔬"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("thinking"):
            with st.expander("🧠 思考过程 (Thinking Chain)", expanded=False):
                st.markdown(f"_{msg['thinking']}_")

if user_input := st.chat_input("请输入标的..."):
    if not silicon_flow_key:
        st.error("请先配置 SiliconFlow Key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 1. Router
    ticker = None
    with st.chat_message("assistant", avatar="👩‍💼"):
        st.write("🔄 董秘正在立项...")
        res, _ = call_agent("Router", SPECIFIC_MODELS["QWEN"], "提取Yahoo Ticker JSON {'ticker': '...'}", user_input)
        try:
            ticker = json.loads(res.replace("```json","").replace("```",""))['ticker']
            st.markdown(f"✅ 标的确认：**{ticker}**")
        except:
            st.error("无法识别标的")
            st.stop()

    # 2. Data Fetching (Initial)
    mkt = fetch_market_data(ticker)
    if mkt['status'] == "OFFLINE":
        st.error("行情数据获取失败")
        st.stop()

    queries = {
        "macro": "global macro economy news market trends",
        "meso": f"{ticker} industry competitors market share",
        "micro": f"{ticker} stock news financial reports analysis",
        "pol": "international geopolitics trade war impact"
    }
    
    with st.status("📡 正在进行全网情报搜集...", expanded=True) as status:
        raw_news = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {k: executor.submit(search_web, v, "news" if k != "meso" else "general") for k, v in queries.items()}
            for k, f in futures.items():
                raw_news[k] = f.result()
        status.update(label="✅ 初始情报已就绪", state="complete")

    # --- THE PROCESS LOOP (Support 1 Retry) ---
    max_retries = 1
    retry_count = 0
    final_report = ""
    
    while retry_count <= max_retries:
        
        # 3. Meeting (Intelligence Reporting)
        # We collect opinions first
        opinions = {}
        
        # Render Agent Avatars only if it's the first run or specifically requested
        if retry_count == 0:
            st.subheader("🗣️ 投研晨会 (Morning Meeting)")
        else:
            st.subheader("🔄 补充研讨 (Follow-up Meeting)")

        # Macro
        with st.chat_message("assistant", avatar="🌍"):
            res, _ = call_agent("Macro", SPECIFIC_MODELS["MINIMAX"], "你是宏观分析师。简述宏观环境。有什么说什么，确保准确。", str(raw_news['macro']))
            st.markdown(f"**宏观**: {res}")
            opinions['macro'] = res

        # Meso
        with st.chat_message("assistant", avatar="🏭"):
            res, _ = call_agent("Meso", SPECIFIC_MODELS["MINIMAX"], f"你是行业分析师。{ticker} 行业情况如何？相关性低也没关系，说你知道的。", str(raw_news['meso']))
            st.markdown(f"**行业**: {res}")
            opinions['meso'] = res

        # Micro
        with st.chat_message("assistant", avatar="🔍"):
            res, _ = call_agent("Micro", SPECIFIC_MODELS["MINIMAX"], f"你是个股分析师。{ticker} 最近有什么新闻？", str(raw_news['micro']))
            st.markdown(f"**个股**: {res}")
            opinions['micro'] = res

        # Quant & Finance (Quick Check)
        with st.chat_message("assistant", avatar="💹"):
            quant_ctx = f"Price:{mkt['price']}, PE:{mkt['pe']}, RSI:{mkt['last_rsi']:.1f}"
            res, _ = call_agent("Finance", SPECIFIC_MODELS["DEEPSEEK"], "评价估值与技术面状态。", quant_ctx)
            st.markdown(f"**量化财经**: {res}")
            opinions['fin_quant'] = res

        # 4. Analyst Drafting
        with st.chat_message("assistant", avatar="📝"):
            st.write("✍️ 综合分析师正在撰写草案...")
            full_context = f"情报:{json.dumps(opinions, ensure_ascii=False)}\n行情:{quant_ctx}"
            report_draft, _ = call_agent("Analyst", SPECIFIC_MODELS["DEEPSEEK"], 
                                "你是首席分析师。撰写一份简明研报，包含逻辑、风险和结论。", full_context)
            st.markdown(report_draft)

        # 5. Chief Researcher Review (Kimi-Thinking)
        with st.chat_message("assistant", avatar="👨‍🔬"):
            st.write("🕵️ **首席研究员 (Kimi)** 正在深度评估...")
            
            review_prompt = f"""
            你是首席研究员。请审查这份研报和现有情报。
            
            1. 如果你认为某个领域（宏观/行业/个股）的信息严重缺失导致无法判断，请输出指令：REWORK: [FIELD] (例如 REWORK: MACRO)。
            2. 如果信息充足，请进行深度思考，输出最终投资建议。
            
            研报草案:
            {report_draft}
            """
            
            review_res, thinking = call_agent("Chief", SPECIFIC_MODELS["KIMI"], review_prompt, "请开始审核。", thinking_needed=True)
            
            # Show Thinking
            if thinking:
                with st.expander("🧠 首席的思考过程 (点击展开)", expanded=False):
                    st.markdown(f"_{thinking}_")
            
            # Check for Rework
            if "REWORK:" in review_res and retry_count < max_retries:
                # Extract field
                match = re.search(r"REWORK:\s*(\w+)", review_res)
                field = match.group(1).lower() if match else "general"
                
                st.warning(f"🚨 首席驳回：认为 {field} 领域信息不足，要求返工！")
                st.markdown(f"_{review_res}_")
                
                # Action: Search again with broader query
                st.write(f"🔍 正在针对 **{field}** 进行深度补充搜索...")
                new_query = f"{ticker} {field} deep analysis details"
                new_info = search_web(new_query, "general")
                
                # Update context
                if field in raw_news:
                    raw_news[field].extend(new_info)
                else:
                    raw_news['micro'].extend(new_info) # Fallback
                
                retry_count += 1
                time.sleep(1)
                st.rerun() # Rerun logic (simulate loop) - actually in this structure we just continue loop
                continue # Go to next iteration of while loop
                
            else:
                # Final Success
                st.success("✅ 审核通过，最终发布。")
                st.markdown(f"### 🏆 首席最终决策\n\n{review_res}")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"### 📑 最终研报\n\n{report_draft}\n\n---\n**🏆 首席点评**: {review_res}", 
                    "avatar": "👨‍🔬",
                    "thinking": thinking
                })
                break

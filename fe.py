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
    page_title="MAS 联合研报终端 v3.5",
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

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "首席研究员就位。请下达调研指令（如：分析 易点天下）。", "avatar": "👨‍🔬"}]
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
            "status": "ONLINE (AV Backup)",
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
        
    return {"status": "OFFLINE", "error": "Market data unavailable from both YF and AV"}

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
    
    # --- 格式与语言宪法 (Format & Language Constitution) ---
    final_sys_prompt += """
    
    【重要输出指令】
    1. 语言：必须全称使用简体中文 (Simplified Chinese) 回复。严禁使用英文（除非是专有名词代码）。
    2. 格式：禁止使用一级(#)或二级(##)大标题。最大只能使用三级(###)标题。建议多用**加粗**来强调。
    3. 内容：严禁重复输出相同的段落或标题。保持回答精炼、紧凑。
    """

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

st.title("🏦 MAS 联合研报终端 v3.6 (Chinese Fixed)")
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
    if not silicon_flow_key:
        st.error("请先在侧边栏输入 SiliconFlow API Key")
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

    # --- STEP 1: SMART ROUTER (Verification Added) ---
    with st.chat_message("assistant", avatar="👩‍💼"):
        st.write("🔍 董秘正在核实代码...")
        
        # 1. Double Search (English + Chinese)
        # English search is good for tickers, Chinese search ensures we get A-share context
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_en = executor.submit(search_web, f"{user_input} stock ticker Yahoo Finance", "general")
            f_cn = executor.submit(search_web, f"{user_input} 股票代码", "general")
            search_res = f_en.result() + f_cn.result()
        
        search_context = "\n".join(search_res)
        
        # 2. Extract
        router_prompt = f"""
        用户想要分析: "{user_input}"
        
        搜索结果:
        {search_context}
        
        请提取Yahoo Finance Ticker。
        规则：
        1. A股必须是 6 位数字 + .SS (上海) 或 .SZ (深圳)。例如 301171 -> 301171.SZ
        2. 港股是 4 位数字 + .HK
        3. 美股是字母
        4. 务必区分“易点天下(301171)”和“中科润宇(301175)”等相似代码，依靠搜索结果中的公司名匹配。
        
        返回JSON: {{'ticker': '...', 'company_name_in_search': '...'}}
        """
        
        res, _ = call_agent("Router", SPECIFIC_MODELS["QWEN"], "你是董秘。精确提取代码。", router_prompt)
        json_data = extract_json_from_markdown(res)
        
        if json_data and 'ticker' in json_data:
            ticker_candidate = json_data['ticker']
            
            # 3. IDENTITY VERIFICATION (New Step)
            # Fetch real name from YFinance to double check
            try:
                real_info = yf.Ticker(ticker_candidate).info
                real_name = real_info.get('longName', '') or real_info.get('shortName', '')
                
                if real_name:
                    # Let Qwen confirm if "real_name" matches "user_input"
                    verify_prompt = f"""
                    用户输入: "{user_input}"
                    提取代码: "{ticker_candidate}"
                    该代码对应的官方名称: "{real_name}"
                    
                    请判断官方名称是否与用户输入匹配？
                    如果匹配，返回 "YES"。
                    如果不匹配（例如用户搜易点天下，但代码对应中科润宇），返回 "NO"。
                    """
                    verify_res, _ = call_agent("Verifier", SPECIFIC_MODELS["QWEN"], "你是审核员。", verify_prompt)
                    
                    if "NO" in verify_res:
                        st.error(f"⚠️ 警告：代码 {ticker_candidate} 对应公司为 **{real_name}**，似乎与您的输入不符。请尝试输入更准确的全名。")
                        st.stop()
                    else:
                        st.session_state.ticker = ticker_candidate
                        st.markdown(f"✅ 身份核验通过：**{real_name} ({ticker_candidate})**")
                        st.session_state.process_status = "ANALYZING"
                        st.rerun()
                else:
                    # Fallback if YF fails to get name (e.g. network issue), trust LLM but warn
                    st.warning(f"⚠️ 无法从交易所验证代码 {ticker_candidate}，将尝试强行分析...")
                    st.session_state.ticker = ticker_candidate
                    st.session_state.process_status = "ANALYZING"
                    st.rerun()
            except Exception as e:
                st.error(f"代码验证失败: {str(e)}")
                st.stop()
                
        else:
            st.error("无法识别有效代码")
            st.stop()

# 3. Analysis Process
if st.session_state.process_status == "ANALYZING" and st.session_state.ticker:
    
    ticker = st.session_state.ticker
    
    # --- STEP A: FETCH DATA ---
    if not st.session_state.market_data:
        with st.status("📡 正在进行全网情报搜集...", expanded=True) as status:
            mkt = fetch_market_data(ticker)
            st.session_state.market_data = mkt
            
            if mkt['status'] == "OFFLINE":
                st.error("行情数据获取失败 (Yahoo & Alpha Vantage 均不可用)")
            
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
    
    # --- STEP B: MEETING ---
    mkt = st.session_state.market_data
    news = st.session_state.raw_news
    opinions = {}
    
    st.divider()
    
    # Dashboard
    if mkt and mkt['status'] != "OFFLINE":
        st.markdown(f"### 📉 行情看板: {mkt.get('name', ticker)} ({mkt.get('symbol')})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("价格", f"{mkt['price']:.2f}", f"{mkt['change_pct']:.2f}%")
        c2.metric("PE", mkt.get('pe', 'N/A'))
        c3.metric("RSI", f"{mkt.get('last_rsi', 0):.1f}")
        c4.metric("MACD", f"{mkt.get('last_macd', {}).get('hist', 0):.3f}")
        
        if 'history_df' in mkt:
            fig = go.Figure(data=[go.Candlestick(x=mkt['history_df'].index,
                            open=mkt['history_df']['Open'], high=mkt['history_df']['High'],
                            low=mkt['history_df']['Low'], close=mkt['history_df']['Close'])])
            fig.update_layout(height=350, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 暂无实时行情K线")

    st.subheader(f"🗣️ 投研会议 (第 {st.session_state.retry_count + 1} 轮)")
    if st.session_state.retry_count > 0:
        st.info(f"💡 本次会议包含了针对 **{st.session_state.last_rework_field}** 领域的补充情报。")
    
    with st.chat_message("assistant", avatar="🌍"):
        prompt = "简述宏观环境。"
        if st.session_state.last_rework_field == "macro": prompt += " (基于最新补充情报)"
        res, _ = call_agent("Macro", SPECIFIC_MODELS["MINIMAX"], "你是宏观分析师。", f"{prompt}\n情报:{str(news['macro'])}")
        st.markdown(f"**宏观**: {res}")
        opinions['macro'] = res

    with st.chat_message("assistant", avatar="🏭"):
        prompt = f"分析 {ticker} 行业。"
        if st.session_state.last_rework_field == "meso": prompt += " (基于最新补充情报)"
        res, _ = call_agent("Meso", SPECIFIC_MODELS["MINIMAX"], f"你是行业分析师。", f"{prompt}\n情报:{str(news['meso'])}")
        st.markdown(f"**行业**: {res}")
        opinions['meso'] = res

    with st.chat_message("assistant", avatar="🔍"):
        prompt = f"分析 {ticker} 个股。"
        if st.session_state.last_rework_field == "micro": prompt += " (基于最新补充情报)"
        res, _ = call_agent("Micro", SPECIFIC_MODELS["MINIMAX"], f"你是公司研究员。", f"{prompt}\n情报:{str(news['micro'])}")
        st.markdown(f"**个股**: {res}")
        opinions['micro'] = res
    
    if mkt and mkt['status'] != "OFFLINE":
        with st.chat_message("assistant", avatar="💹"):
            quant_ctx = f"Price:{mkt['price']}, PE:{mkt['pe']}, RSI:{mkt.get('last_rsi')}"
            res, _ = call_agent("Finance", SPECIFIC_MODELS["DEEPSEEK"], "你是财务专家。请分析估值与技术面。", quant_ctx)
            st.markdown(f"**量化**: {res}")
            opinions['quant'] = res
    else:
        quant_ctx = "Market Data Offline"

    # --- STEP C: DRAFTING ---
    with st.chat_message("assistant", avatar="📝"):
        st.write("✍️ 正在撰写研报草案...")
        full_ctx = f"Opinions:{json.dumps(opinions, ensure_ascii=False)}\nMarket:{quant_ctx}"
        report_draft, _ = call_agent("Analyst", SPECIFIC_MODELS["DEEPSEEK"], 
                            "你是首席分析师。写一份结构化研报，包含逻辑、风险和结论。", full_ctx)
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
        
        if "REWORK:" in review_res and st.session_state.retry_count < 1:
            match = re.search(r"REWORK:\s*(\w+)", review_res)
            field = match.group(1).lower() if match else "micro"
            # Fuzzy map to keys
            if "macro" in field: field = "macro"
            elif "indus" in field or "meso" in field: field = "meso"
            else: field = "micro"
            
            st.session_state.last_rework_field = field
            st.warning(f"🚨 驳回：要求补充 **{field}** 领域信息。正在执行...")
            
            new_query = f"{ticker} {field} analysis latest news details"
            new_info = search_web(new_query, "general")
            
            st.session_state.raw_news[field].extend(new_info)
            st.session_state.retry_count += 1
            st.rerun()
            
        else:
            st.success("✅ 审核通过")
            st.markdown(f"### 🏆 最终决策\n{review_res}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"### 📑 最终研报 ({ticker})\n\n{report_draft}\n\n---\n**🏆 首席决策**: {review_res}", 
                "avatar": "👨‍🔬", 
                "thinking": thinking
            })
            st.session_state.process_status = "DONE"

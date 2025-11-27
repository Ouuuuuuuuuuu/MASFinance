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
import re
import requests

# --- PAGE SETUP ---
st.set_page_config(
    page_title="MAS 联合研报终端 v4.1",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #1f2937; }
    .report-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .stChatMessage { background-color: transparent; }
    .stChatMessage .stChatMessageAvatar { background-color: #e5e7eb; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e5e7eb; padding: 10px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
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

# --- CONFIGURATION & SECRETS ---
# 修改说明：直接从 st.secrets 读取所有 Keys，不再提供侧边栏输入
try:
    SECRETS = st.secrets["api_keys"]
    silicon_flow_key = SECRETS["silicon_flow"]
    tavily_key = SECRETS["tavily"]
    alpha_vantage_key = SECRETS["alpha_vantage"]
except Exception as e:
    st.error("❌ 启动失败：未检测到完整的 API Keys 配置")
    st.info("""
    请确保项目根目录下的 `.streamlit/secrets.toml` 文件包含以下内容：
    
    ```toml
    [api_keys]
    silicon_flow = "sk-..."
    tavily = "tvly-..."
    alpha_vantage = "..."
    ```
    """)
    st.stop()

# --- MODEL CONFIG ---
SPECIFIC_MODELS = {
    "ROUTER": "Qwen/Qwen2.5-72B-Instruct",
    "VERIFIER": "Qwen/Qwen2.5-72B-Instruct",
    "MACRO": "MiniMaxAI/MiniMax-M2",
    "MESO": "MiniMaxAI/MiniMax-M2",
    "MICRO": "MiniMaxAI/MiniMax-M2",
    "QUANT": "deepseek-ai/DeepSeek-V3",
    "WRITER": "deepseek-ai/DeepSeek-V3",
    "CHIEF": "moonshotai/Kimi-K2-Thinking"
}

# --- STATE INITIALIZATION ---
def init_state():
    defaults = {
        "messages": [{"role": "assistant", "content": "首席研究员就位。请下达调研指令。", "avatar": "👨‍🔬"}],
        "process_status": "IDLE", # IDLE, VERIFYING, ANALYZING, DONE
        "ticker": None,
        "market_data": None,
        "raw_news": {},     # {field: [news1, news2]}
        "opinions": {},     # {field: "analysis text"} -> 用于实现续写不覆盖
        "retry_count": 0,
        "last_rework_field": None,
        "user_query": "",
        "verification_fail": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ 控制台")
    # 移除了所有 Input，仅显示状态
    st.success("✅ API 密钥已加载")
    st.caption("Environment: Protected")

    st.divider()
    if st.button("🗑️ 清空历史 & 重置"):
        st.session_state.clear()
        st.rerun()

# --- UTILS WITH CACHING ---

@st.cache_data(ttl=3600)
def fetch_from_alphavantage(ticker, api_key):
    if not api_key: return None
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=compact"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "Time Series (Daily)" not in data: return None
        
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df = df.rename(columns={"4. close": "Close", "1. open": "Open", "2. high": "High", "3. low": "Low"})
        df = df.astype(float).sort_index()
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_from_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty: return None, None
        return hist, stock.info
    except:
        return None, None

def calculate_technical_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
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

def fetch_market_data(ticker, av_key):
    hist, info = fetch_from_yfinance(ticker)
    source = "YFinance"
    
    if hist is None:
        hist = fetch_from_alphavantage(ticker, av_key)
        info = {}
        source = "AlphaVantage"
    
    if hist is None or hist.empty:
        return {"status": "OFFLINE", "error": "Data unavailable"}
        
    hist = calculate_technical_indicators(hist)
    
    try:
        last_close = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_pct = ((last_close - prev_close) / prev_close) * 100
        
        return {
            "status": f"ONLINE ({source})",
            "symbol": ticker.upper(),
            "name": info.get('longName', ticker) if info else ticker,
            "price": last_close,
            "change_pct": change_pct,
            "pe": info.get('trailingPE', 'N/A') if info else 'N/A',
            "cap": info.get('marketCap', 'N/A') if info else 'N/A',
            "history_df": hist,
            "last_macd": hist['MACD_Hist'].iloc[-1],
            "last_rsi": hist['RSI'].iloc[-1]
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

@st.cache_data(ttl=1800)
def search_web(query, topic="general", _api_key=None):
    if not _api_key: return ["Error: Missing Tavily API Key"]
    try:
        tavily = TavilyClient(api_key=_api_key)
        res = tavily.search(query=query, topic=topic, max_results=5)
        return [f"- {r['title']}: {r['content'][:350]}" for r in res['results']]
    except Exception as e:
        return [f"Search Error: {str(e)}"]

def get_llm_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

def call_agent(agent_name, model_id, system_prompt, user_prompt, thinking_needed=False):
    client = get_llm_client(silicon_flow_key)
    
    final_sys_prompt = system_prompt + """
    \n【输出规范】
    1. 语言：简体中文 (Simplified Chinese)。
    2. 格式：Markdown，禁止使用一级标题(#)，从三级(###)开始。
    3. 风格：专业、客观、金融研报风。
    """
    
    if thinking_needed:
        final_sys_prompt += "\nIMPORTANT: First output thinking process in <thinking>...</thinking>, then final answer."

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

def extract_json_from_markdown(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return None

# --- MAIN LOGIC ---

# 1. Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("thinking"):
            with st.expander("🧠 思考过程", expanded=False):
                st.markdown(f"_{msg['thinking']}_")

# 2. Input
if user_input := st.chat_input("请输入股票名称或代码..."):
    # 检查 Keys 是否存在
    if not (silicon_flow_key and tavily_key):
        st.error("配置错误：缺少必要的 API Key")
        st.stop()
        
    # Reset State for new query
    st.session_state.process_status = "VERIFYING"
    st.session_state.ticker = None
    st.session_state.market_data = None
    st.session_state.raw_news = {}
    st.session_state.opinions = {} # Clear opinions for new stock
    st.session_state.retry_count = 0
    st.session_state.last_rework_field = None
    st.session_state.verification_fail = False
    st.session_state.user_query = user_input
    
    st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "👤"})
    st.rerun()

# 3. VERIFICATION PHASE
if st.session_state.process_status == "VERIFYING":
    with st.chat_message("assistant", avatar="👩‍💼"):
        st.write("🔍 董秘正在核实代码...")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f_en = executor.submit(search_web, f"{st.session_state.user_query} stock ticker Yahoo Finance", "general", tavily_key)
            f_cn = executor.submit(search_web, f"{st.session_state.user_query} 股票代码", "general", tavily_key)
            search_res = f_en.result() + f_cn.result()
            
        search_ctx = "\n".join(search_res)
        
        router_prompt = f"""
        User Query: "{st.session_state.user_query}"
        Search Results: {search_ctx}
        
        Task: Extract the Yahoo Finance Ticker.
        Rules:
        1. A-Share: 6 digits + .SS or .SZ (e.g., 600519.SS)
        2. HK Share: 4 digits + .HK
        3. US Share: Ticker symbol (e.g., TSLA)
        4. Return JSON: {{'ticker': '...', 'company_name': '...'}}
        """
        
        res, _ = call_agent("Router", SPECIFIC_MODELS["ROUTER"], "Extract Ticker JSON.", router_prompt)
        json_data = extract_json_from_markdown(res)
        
        if json_data and 'ticker' in json_data:
            candidate = json_data['ticker']
            candidate_name = json_data.get('company_name', 'Unknown')
            
            verify_prompt = f"""
            User Input: "{st.session_state.user_query}"
            Extracted Ticker: "{candidate}"
            Extracted Name: "{candidate_name}"
            
            Does this ticker likely match the user's intent?
            Return JSON: {{'match': true/false, 'reason': '...'}}
            """
            v_res, _ = call_agent("Verifier", SPECIFIC_MODELS["VERIFIER"], "Verify intent JSON.", verify_prompt)
            v_json = extract_json_from_markdown(v_res)
            
            if v_json and v_json.get('match'):
                st.session_state.ticker = candidate
                st.session_state.process_status = "ANALYZING"
                st.success(f"✅ 锁定标的: {candidate_name} ({candidate})")
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.ticker = candidate
                st.session_state.verification_fail = True
                # 7. 优化提示语：您是不是想找...
                st.warning(f"⚠️ 未完全匹配。您是不是想找：**{candidate_name} ({candidate})**？")
                col1, col2 = st.columns(2)
                if col1.button("✅ 是的，继续分析"):
                    st.session_state.process_status = "ANALYZING"
                    st.rerun()
                if col2.button("❌ 不是，停止"):
                    st.session_state.process_status = "IDLE"
                    st.stop()
        else:
            st.error("❌ 无法识别有效代码，请尝试输入更精确的名称。")
            st.session_state.process_status = "IDLE"

# 4. ANALYSIS PHASE
if st.session_state.process_status == "ANALYZING" and st.session_state.ticker:
    ticker = st.session_state.ticker
    
    # --- FETCH DATA ---
    if not st.session_state.market_data:
        with st.status("📡 正在获取行情与情报...", expanded=True) as status:
            mkt = fetch_market_data(ticker, alpha_vantage_key)
            st.session_state.market_data = mkt
            
            if mkt['status'] == "OFFLINE":
                st.error("行情数据获取失败")
            
            queries = {
                "macro": "global macro economy news market trends",
                "meso": f"{ticker} industry competitors market share",
                "micro": f"{ticker} stock news financial reports analysis",
            }
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {k: executor.submit(search_web, v, "news", tavily_key) for k, v in queries.items()}
                for k, f in futures.items():
                    st.session_state.raw_news[k] = f.result()
            
            status.update(label="✅ 数据就绪", state="complete")

    # --- RENDER DASHBOARD ---
    mkt = st.session_state.market_data
    if mkt and mkt['status'] != "OFFLINE":
        with st.container():
            st.markdown(f"### 📉 {mkt.get('name')} ({mkt.get('symbol')})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("价格", f"{mkt['price']:.2f}", f"{mkt['change_pct']:.2f}%")
            c2.metric("PE", mkt.get('pe'))
            c3.metric("RSI (14)", f"{mkt.get('last_rsi', 0):.1f}")
            c4.metric("MACD", f"{mkt.get('last_macd', 0):.3f}")
    
    st.divider()
    
    # --- AGENT MEETING ---
    news = st.session_state.raw_news
    
    # Helper to render or get cached opinion (实现了“续写不覆盖”逻辑)
    def render_opinion(role, avatar, key, model, prompt_tmpl):
        with st.chat_message("assistant", avatar=avatar):
            is_rework_target = st.session_state.last_rework_field == key
            existing_opinion = st.session_state.opinions.get(key, None)
            
            # 逻辑：
            # 1. 如果已有观点 且 不是当前需要返工的领域 -> 直接展示旧观点 (省钱 & 稳定)
            # 2. 如果是返工领域 -> 传入旧观点，要求"续写/修正"
            # 3. 如果是第一次 -> 初始生成
            
            if existing_opinion and not is_rework_target:
                st.markdown(f"**{role} (已归档)**: {existing_opinion}")
                return existing_opinion
            
            # 构造 Prompt
            current_news = str(news.get(key, ''))
            
            if is_rework_target and existing_opinion:
                # 6. 返工内容继续写，不覆盖
                final_prompt = f"""
                {prompt_tmpl}
                
                【重要】这是你之前的分析：
                "{existing_opinion}"
                
                这是新补充的情报：
                {current_news}
                
                请基于新情报对之前的分析进行**补充和修订**。不要完全推翻，保留有价值的旧观点，将新发现整合进去。
                """
                st.info("🔄 正在基于新情报修订观点...")
            else:
                final_prompt = f"{prompt_tmpl}\n情报:{current_news}"

            res, _ = call_agent(role, model, f"你是{role}分析师。", final_prompt)
            st.markdown(f"**{role}**: {res}")
            
            # Save to session state
            st.session_state.opinions[key] = res
            return res

    st.subheader(f"🗣️ 投研会议 (第 {st.session_state.retry_count + 1} 轮)")
    
    # Render Agents
    render_opinion("Macro", "🌍", "macro", SPECIFIC_MODELS["MACRO"], "简述宏观环境。")
    render_opinion("Industry", "🏭", "meso", SPECIFIC_MODELS["MESO"], f"分析 {ticker} 行业竞争格局。")
    render_opinion("Company", "🔍", "micro", SPECIFIC_MODELS["MICRO"], f"分析 {ticker} 个股基本面。")
    
    # Quant (Always run if market data exists, quick check)
    if mkt and mkt['status'] != "OFFLINE":
        with st.chat_message("assistant", avatar="💹"):
            q_ctx = f"Price:{mkt['price']}, RSI:{mkt.get('last_rsi')}, MACD:{mkt.get('last_macd')}"
            res, _ = call_agent("Quant", SPECIFIC_MODELS["QUANT"], "技术面分析师", f"基于数据评价趋势：{q_ctx}")
            st.markdown(f"**量化**: {res}")
            st.session_state.opinions['quant'] = res

    # --- DRAFTING ---
    with st.chat_message("assistant", avatar="📝"):
        st.write("✍️ 正在撰写草案...")
        # Analyst uses all current opinions (some cached, some updated)
        draft_ctx = f"Opinions: {json.dumps(st.session_state.opinions, ensure_ascii=False)}"
        report_draft, _ = call_agent("Writer", SPECIFIC_MODELS["WRITER"], "首席分析师。整合研报。", draft_ctx)
        st.markdown(report_draft)

    # --- CHIEF REVIEW ---
    with st.chat_message("assistant", avatar="👨‍🔬"):
        st.write("🕵️ 首席研究员审核中...")
        review_prompt = f"研报草案:\n{report_draft}\n\n指令：若信息不足，输出 REWORK: [MACRO/MESO/MICRO]。否则输出最终结论。"
        review_res, thinking = call_agent("Chief", SPECIFIC_MODELS["CHIEF"], "首席研究员。严格审核。", review_prompt, thinking_needed=True)
        
        if thinking:
            with st.expander("🧠 首席思考过程", expanded=True):
                st.markdown(f"_{thinking}_")
        
        # Logic for Rework
        if "REWORK:" in review_res and st.session_state.retry_count < 1:
            match = re.search(r"REWORK:\s*(\w+)", review_res)
            field = match.group(1).lower() if match else "micro"
            
            field_map = {"macro": "macro", "industry": "meso", "meso": "meso", "company": "micro", "micro": "micro"}
            target_key = field_map.get(field, "micro")
            
            st.warning(f"🚨 驳回：需补充 {target_key} 领域信息。正在执行...")
            
            # 5. 智能补充搜索 (Agent 构造关键词)
            st.write(f"🔍 正在针对 {target_key} 进行深度挖掘...")
            keyword_prompt = f"针对股票 {ticker}，目前 {target_key} 领域信息缺失。请生成3个具体的Google搜索关键词用于挖掘该领域的深层信息。只返回关键词，用空格分隔。"
            keywords, _ = call_agent("Searcher", SPECIFIC_MODELS["VERIFIER"], "Search Expert", keyword_prompt)
            
            new_query = f"{ticker} {keywords}"
            new_info = search_web(new_query, "general", tavily_key)
            
            if target_key in st.session_state.raw_news:
                st.session_state.raw_news[target_key].extend(new_info)
            else:
                 st.session_state.raw_news[target_key] = new_info
            
            st.session_state.retry_count += 1
            st.session_state.last_rework_field = target_key
            time.sleep(2)
            st.rerun()
            
        else:
            st.success("✅ 审核通过")
            st.markdown(f"### 🏆 最终决策\n{review_res}")
            
            final_content = f"### 📑 最终研报 ({ticker})\n\n{report_draft}\n\n---\n**🏆 首席决策**: {review_res}"
            st.session_state.messages.append({"role": "assistant", "content": final_content, "avatar": "👨‍🔬", "thinking": thinking})
            st.session_state.process_status = "DONE"
            if st.button("开始新研究"):
                st.session_state.process_status = "IDLE"
                st.rerun()

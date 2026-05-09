import streamlit as st
import requests
import time

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# CUSTOM CSS FOR WHATSAPP-STYLE CHAT
# ============================================================

st.markdown("""
<style>
    .user-bubble {
        background-color: #DCF8C6;
        padding: 12px 16px;
        border-radius: 12px 12px 0px 12px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: #000;
        font-size: 14px;
    }
    .bot-bubble {
        background-color: #FFFFFF;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 0px;
        margin: 8px 0;
        max-width: 80%;
        border: 1px solid #E0E0E0;
        color: #000;
        font-size: 14px;
    }
    .source-tag {
        background-color: #E3F2FD;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        color: #1565C0;
        display: inline-block;
        margin: 2px;
    }
    .latency-tag {
        color: #888;
        font-size: 11px;
        margin-top: 4px;
    }
    .agent-tag {
        background-color: #FFF3E0;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        color: #E65100;
        display: inline-block;
        margin: 2px;
    }
    .header-style {
        text-align: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="header-style">', unsafe_allow_html=True)
st.title("📈 Financial Market Intelligence")
st.caption("RAG Pipeline with Multi-AI Agentic Workflow | Powered by CrewAI & Gemini")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Query Mode",
        options=["rag", "agentic"],
        format_func=lambda x: "🔍 RAG (Fast)" if x == "rag" else "🤖 Agentic (Detailed)",
        help="RAG mode gives quick answers. Agentic mode uses 4 AI agents for comprehensive analysis.",
    )

    st.divider()

    st.markdown("*📊 Agent Pipeline (Agentic Mode):*")
    st.markdown("""
    1. 🔎 Financial Data Retriever
    2. 📊 Financial Analysis Expert
    3. 💼 Portfolio Strategy Advisor
    4. ⚠️ Risk Assessment Specialist
    """)

    st.divider()

    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="FastAPI backend URL",
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()

# ============================================================
# CHAT STATE
# ============================================================

if "chat" not in st.session_state:
    st.session_state.chat = []

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show metadata
        if msg.get("sources"):
            tickers = set(
                s.get("ticker", "") for s in msg["sources"] if s.get("ticker")
            )
            if tickers:
                tags = " ".join(
                    [f'<span class="source-tag">📄 {t}</span>' for t in tickers]
                )
                st.markdown(tags, unsafe_allow_html=True)
        if msg.get("agents"):
            tags = " ".join(
                [f'<span class="agent-tag">🤖 {a}</span>' for a in msg["agents"]]
            )
            st.markdown(tags, unsafe_allow_html=True)
        if msg.get("latency"):
            st.markdown(
                f'<div class="latency-tag">⏱️ {msg["latency"]}s</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input("Ask a financial question...")

if user_input:
    # Add user message
    st.session_state.chat.append({"role": "user", "content": user_input})
    st.markdown(
        f'<div class="user-bubble">🧑 {user_input}</div>',
        unsafe_allow_html=True,
    )

    # Call FastAPI
    with st.spinner("Analyzing..." if mode == "agentic" else "Searching..."):
        try:
            response = requests.post(
                f"{api_url}/query",
                json={"question": user_input, "mode": mode},
                timeout=300,
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer received.")
                sources = data.get("sources", [])
                agents = data.get("agents_used", [])
                latency = data.get("latency", None)

                # Add bot message with metadata
                bot_msg = {
                    "role": "bot",
                    "content": answer,
                    "sources": sources,
                    "agents": agents,
                    "latency": latency,
                }
                st.session_state.chat.append(bot_msg)

                # Display response
                st.markdown(
                    f'<div class="bot-bubble">🤖 {answer}</div>',
                    unsafe_allow_html=True,
                )

                # Show metadata
                if sources:
                    tickers = set(
                        s.get("ticker", "") for s in sources if s.get("ticker")
                    )
                    if tickers:
                        tags = " ".join(
                            [f'<span class="source-tag">📄 {t}</span>' for t in tickers]
                        )
                        st.markdown(tags, unsafe_allow_html=True)
                if agents:
                    tags = " ".join(
                        [f'<span class="agent-tag">🤖 {a}</span>' for a in agents]
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                if latency:
                    st.markdown(
                        f'<div class="latency-tag">⏱️ {latency}s</div>',
                        unsafe_allow_html=True,
                    )
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                st.session_state.chat.append({"role": "bot", "content": error_msg})
                st.error(error_msg)

        except requests.exceptions.ConnectionError:
            err = "Cannot connect to API. Make sure FastAPI is running."
            st.session_state.chat.append({"role": "bot", "content": err})
            st.error(err)
        except requests.exceptions.Timeout:
            err = "Request timed out. Agentic mode can take a few minutes."
            st.session_state.chat.append({"role": "bot", "content": err})
            st.error(err)

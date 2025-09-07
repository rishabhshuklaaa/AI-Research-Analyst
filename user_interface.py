import streamlit as st
import requests
import json
import pandas as pd
import os

# --- Configuration ---
FLASK_API_URL = "http://127.0.0.1:5000"

# --- API Calling Functions ---
def call_api(endpoint, data):
    try:
        response = requests.post(f"{FLASK_API_URL}{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection to backend failed: {e}"}

# --- Custom Display Functions with Source Display ---

def display_swot(result):
    if "error" in result:
        st.error(f"Error generating SWOT: {result.get('raw_output', result['error'])}")
        return
    st.subheader("SWOT Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.success("👍 Strengths")
        for item in result.get("strengths", []): st.markdown(f"- {item}")
        st.warning("👎 Weaknesses")
        for item in result.get("weaknesses", []): st.markdown(f"- {item}")
    with col2:
        st.info("🚀 Opportunities")
        for item in result.get("opportunities", []): st.markdown(f"- {item}")
        st.error("💣 Threats")
        for item in result.get("threats", []): st.markdown(f"- {item}")
    
    # --- FIX: Source display logic is now correctly placed inside the function ---
    if "sources" in result and result["sources"]:
        with st.expander("View Sources"):
            for source in result["sources"]:
                st.caption(source)

def display_promise_tracker(result):
    if "error" in result:
        st.error(f"Error generating comparison: {result.get('raw_output', result['error'])}")
        return
    st.subheader("Promise Tracker Results")
    st.metric(label="Sentiment Change", value=result.get("sentiment_change", "N/A"))
    with st.expander("📝 Summary of Evolution", expanded=True):
        st.write(result.get("summary_of_evolution", "No summary available."))
    col1, col2 = st.columns(2)
    with col1:
        st.info("➕ New Information")
        for item in result.get("new_information", []): st.markdown(f"- {item}")
    with col2:
        st.warning("➖ Dropped Points")
        for item in result.get("dropped_points", []): st.markdown(f"- {item}")
    
    # Also add source display here if your backend provides it
    if "sources" in result and result["sources"]:
        with st.expander("View Sources"):
            for source in result["sources"]:
                st.caption(source)

def display_market_context(result):
    if "error" in result:
        st.error(f"Error fetching context: {result['error']}")
        return
    st.subheader("Market Context Analysis")
    st.metric(label="Overall Market Sentiment", value=result.get("overall_sentiment", "N/A"))
    st.info("🔑 Key Competitor Moves")
    for item in result.get("key_competitor_moves", []): st.markdown(f"- {item}")
    st.warning("📈 Major Industry Trends")
    for item in result.get("major_industry_trends", []): st.markdown(f"- {item}")
    
    if "sources" in result and result["sources"]:
        with st.expander("View Sources"):
            for source in result["sources"]:
                st.caption(source)

def display_memo(result):
    if "error" in result:
        st.error(f"Error generating memo: {result.get('raw_output', result['error'])}")
        return
    st.subheader("Investment Memo")
    with st.expander("📜 Investment Thesis", expanded=True):
        st.info(result.get("investment_thesis", "N/A"))
    st.success("✅ Positive Catalysts")
    for item in result.get("positive_catalysts", []): st.markdown(f"- {item}")
    st.error("🚩 Key Risks")
    for item in result.get("key_risks", []): st.markdown(f"- {item}")
    st.markdown("---")
    st.write(f"**Conclusion:** *{result.get('conclusion', 'N/A')}*")

    if "sources" in result and result["sources"]:
        with st.expander("View Sources"):
            for source in result["sources"]:
                st.caption(source)


# --- STREAMLIT APP ---

st.set_page_config(page_title="AI Analyst Assistant", layout="wide", initial_sidebar_state="expanded")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []

# --- SIDEBAR for Data Input ---
with st.sidebar:
    st.title("AI Analyst Assistant 🦅")
    st.markdown("Your intelligent research partner.")
    
    st.header("1. Start New Session")
    st.markdown("Provide sources to begin your analysis.")
    
    url1 = st.text_input("Primary Document URL", placeholder="e.g., Latest Press Release")
    url2 = st.text_input("Historical Context URL", placeholder="e.g., Last Year's Report")
    url3 = st.text_input("Supporting Analysis URL", placeholder="e.g., News Article")
    
    uploaded_pdfs = st.file_uploader("Upload PDF Reports", type="pdf", accept_multiple_files=True)

    if st.button("Start Analysis Session", type="primary", use_container_width=True):
        with st.spinner("Processing documents... This may take a moment."):
            urls = [url for url in [url1, url2, url3] if url]
            pdf_files_to_upload = []
            if uploaded_pdfs:
                for pdf in uploaded_pdfs:
                    with open(pdf.name, "wb") as f:
                        f.write(pdf.getbuffer())
                    pdf_files_to_upload.append(pdf)

            ingest_result = call_api("/api/ingest", {"urls": urls, "pdfs": [f.name for f in pdf_files_to_upload]})
            if "error" in ingest_result:
                st.error(ingest_result["error"])
            else:
                st.session_state.ingested_sources = urls + [f.name for f in pdf_files_to_upload]
                st.success("Ingestion complete!")
    
    st.markdown("---")
    st.header("2. Session Status")
    if st.session_state.ingested_sources:
        st.success("✅ Ready for Analysis")
        with st.expander("View Ingested Sources"):
            for src in st.session_state.ingested_sources:
                st.caption(src)
    else:
        st.warning("⚪ Waiting for documents...")


# --- MAIN PAGE with Tabs ---
st.header("Analysis Dashboard")

qna_tab, swot_tab, promise_tab, context_tab, chart_tab, memo_tab = st.tabs([
    "💬 Q&A Chat", "📝 SWOT", "🔄 Promise Tracker", "🌐 Market Context", "📊 Visualize Data", "✍️ Final Memo"
])

# Q&A Chat Tab
with qna_tab:
    st.subheader("Conversational Q&A")
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources if they exist in history
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.caption(source)

    # --- FIX: The chat input logic is now correctly placed INSIDE the tab ---
    if prompt := st.chat_input("Ask a question about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_api("/api/qna", {"question": prompt})
                answer = response.get("answer", "Sorry, I couldn't find an answer.")
                sources = response.get("sources", [])
                
                st.markdown(answer)
                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.caption(source)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        })
        st.rerun() # Rerun to update the chat display immediately

# SWOT Analysis Tab
with swot_tab:
    st.subheader("Generate SWOT Analysis")
    swot_topic = st.text_input("Enter a topic to analyze for SWOT", placeholder="e.g., Reliance Industries' Q4 performance")
    if st.button("Generate SWOT", use_container_width=True):
        if swot_topic:
            with st.spinner("Analyzing..."):
                result = call_api("/api/swot", {"topic": swot_topic})
                display_swot(result)
        else:
            st.warning("Please enter a topic.")

# --- FIX: Removed the misplaced source display code from here ---

# Promise Tracker Tab
with promise_tab:
    st.subheader("Track Promises and Narrative Changes")
    compare_topic = st.text_input("Topic to track", placeholder="e.g., Green Energy Strategy")
    source1 = st.text_input("First (older) source URL/PDF", placeholder="Enter full URL or PDF filename")
    source2 = st.text_input("Second (newer) source URL/PDF", placeholder="Enter full URL or PDF filename")
    if st.button("Compare Sources", use_container_width=True):
        if compare_topic and source1 and source2:
            with st.spinner("Comparing narratives..."):
                result = call_api("/api/compare", {"topic": compare_topic, "source1": source1, "source2": source2})
                display_promise_tracker(result)
        else:
            st.warning("Please fill in all fields.")

# Market Context Tab
with context_tab:
    st.subheader("Analyze the Broader Market Context")
    company = st.text_input("Company name", placeholder="e.g., Tata Motors")
    competitors = st.text_input("Competitors", placeholder="e.g., Mahindra, Maruti Suzuki")
    industry = st.text_input("Industry name", placeholder="e.g., Indian Automotive Sector")
    if st.button("Analyze Market", use_container_width=True):
        if company and competitors and industry:
            with st.spinner("Fetching market news and analyzing..."):
                data = {
                    "company": company,
                    "competitors": [c.strip() for c in competitors.split(',')],
                    "industry": industry
                }
                result = call_api("/api/context", data)
                display_market_context(result)
        else:
            st.warning("Please fill in all fields.")

# Visualize Data Tab
with chart_tab:
    st.subheader("Extract & Visualize Numerical Data")
    chart_topic = st.text_input("Topic for data extraction", placeholder="e.g., Q4 Financial Results")
    data_points = st.text_input("Data points to visualize", placeholder="e.g., Revenue, Net Profit")
    if st.button("Generate Chart", use_container_width=True):
        if chart_topic and data_points:
            with st.spinner("Extracting data and creating chart..."):
                data = {
                    "topic": chart_topic,
                    "data_points": [d.strip() for d in data_points.split(',')]
                }
                response = call_api("/api/chart", data)
                st.success(response.get("message", "Chart generation process completed."))
                
                chart_filename = "chart.png"
                if os.path.exists(chart_filename):
                    st.image(chart_filename)
                else:
                    st.warning("Chart file was not found. The backend may have failed to create it.")
        else:
            st.warning("Please fill in all fields.")

# Final Memo Tab
with memo_tab:
    st.subheader("Generate Final Investment Memo")
    memo_topic = st.text_input("Main topic for the memo", placeholder="e.g., Investment case for Reliance Industries")
    if st.button("Generate Memo", type="primary", use_container_width=True):
        if memo_topic:
            with st.spinner("Synthesizing final report..."):
                result = call_api("/api/memo", {"topic": memo_topic})
                display_memo(result)
        else:
            st.warning("Please enter a topic for the memo.")
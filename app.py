"""
DataSense AI - Kali Linux Terminal Interface
A Streamlit application with full hacker/terminal theme
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Import terminal theme
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.terminal_theme import (
    get_terminal_css, 
    get_ascii_banner, 
    get_hacker_emojis,
    get_matrix_rain_html
)

# Page configuration
st.set_page_config(
    page_title="DataSense AI - Terminal",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)

# Matrix rain background
st.markdown(get_matrix_rain_html(), unsafe_allow_html=True)

# Get hacker emojis
emojis = get_hacker_emojis()

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None

if 'cleaned_datasets' not in st.session_state:
    st.session_state.cleaned_datasets = {}

if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main Terminal Header with ASCII Art
st.markdown(f"""
<div class="terminal-header glitch">
    <pre class="ascii-art">{get_ascii_banner()}</pre>
    <h1>⚡ DATASENSE AI - NEURAL ANALYSIS TERMINAL ⚡</h1>
    <p>{emojis['fire']} Advanced Data Intelligence & Machine Learning System {emojis['fire']}</p>
    <p>{emojis['skull']} Powered by Manoj Aberathna | System Status: ACTIVE {emojis['skull']}</p>
</div>
""", unsafe_allow_html=True)

# System Initialization Message
st.markdown(f"""
<div class="terminal-block">
    <p>{emojis['terminal']} <strong>SYSTEM INITIALIZED</strong> | Timestamp: ACTIVE SESSION</p>
    <p>{emojis['unlock']} <strong>ACCESS LEVEL:</strong> ADMINISTRATOR</p>
    <p>{emojis['satellite']} <strong>CONNECTION:</strong> SECURE</p>
</div>
""", unsafe_allow_html=True)

# Welcome Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### {emojis['brain']} NEURAL NETWORK INTERFACE")
    st.markdown(f"""
    <div class="terminal-block">
    <p><strong>DataSense AI</strong> is an advanced cyber intelligence platform that transforms raw data into actionable insights 
    using cutting-edge analytics, machine learning algorithms, and artificial intelligence.</p>
    
    <h4>{emojis['fire']} CORE CAPABILITIES:</h4>
    <ul style="color: #00ffff;">
        <li><strong>{emojis['lightning']} DATA UPLOAD & MANAGEMENT</strong> - Support for CSV, Excel, JSON files (Max: 200MB)</li>
        <li><strong>{emojis['chart']} VISUAL ANALYTICS</strong> - Interactive charts: Bar, Pie, Line, Scatter, Heatmap, Boxplot</li>
        <li><strong>{emojis['target']} DEEP ANALYSIS</strong> - Automated profiling, outlier detection, statistical analysis</li>
        <li><strong>{emojis['wrench']} DATA SANITIZATION</strong> - Handle missing values, duplicates, and outliers</li>
        <li><strong>{emojis['robot']} AUTO ML ENGINE</strong> - Train models for Classification, Regression, Clustering</li>
        <li><strong>{emojis['computer']} AI CHAT INTERFACE</strong> - Natural language queries powered by Gemini AI</li>
        <li><strong>{emojis['brain']} AUTONOMOUS AGENT</strong> - Automated business insights and recommendations</li>
        <li><strong>{emojis['data']} REPORT GENERATION</strong> - Professional PDF reports with complete analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"### {emojis['data']} SYSTEM METRICS")
    
    # Display stats
    num_datasets = len(st.session_state.datasets)
    num_cleaned = len(st.session_state.cleaned_datasets)
    num_models = len(st.session_state.ml_models)
    
    st.markdown(f"""
    <div class="metric-container">
        <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">DATABASE ENTRIES</p>
        <p style="color: #00ffff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{num_datasets}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-container">
        <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">SANITIZED DATA</p>
        <p style="color: #00ffff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{num_cleaned}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-container">
        <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">NEURAL MODELS</p>
        <p style="color: #00ffff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{num_models}</p>
    </div>
    """, unsafe_allow_html=True)

# Quick Start Protocol
st.markdown("---")
st.markdown(f"### {emojis['lightning']} QUICK START PROTOCOL")

cols = st.columns(4)

with cols[0]:
    st.markdown(f"""
    <div class="terminal-block glitch">
        <h4>{emojis['lightning']} STEP 1: UPLOAD</h4>
        <p style="color: #00ffff;">Initialize data stream<br/>(CSV, Excel, JSON)</p>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown(f"""
    <div class="terminal-block glitch">
        <h4>{emojis['chart']} STEP 2: ANALYZE</h4>
        <p style="color: #00ffff;">Visualize and decode<br/>data patterns</p>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="terminal-block glitch">
        <h4>{emojis['wrench']} STEP 3: SANITIZE</h4>
        <p style="color: #00ffff;">Clean and prepare<br/>data matrix</p>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    st.markdown(f"""
    <div class="terminal-block glitch">
        <h4>{emojis['brain']} STEP 4: DEPLOY AI</h4>
        <p style="color: #00ffff;">Generate insights &<br/>ML models</p>
    </div>
    """, unsafe_allow_html=True)

# Current Dataset Info
if st.session_state.current_dataset:
    st.markdown("---")
    st.markdown(f"### {emojis['data']} ACTIVE DATASET")
    df = st.session_state.datasets[st.session_state.current_dataset]
    
    st.markdown(f"""
    <div class="terminal-block">
        <p>{emojis['target']} <strong>DATASET ID:</strong> {st.session_state.current_dataset}</p>
        <p>{emojis['data']} <strong>RECORDS:</strong> {len(df):,} rows | <strong>FIELDS:</strong> {len(df.columns)} columns</p>
        <p>{emojis['gear']} <strong>MEMORY FOOTPRINT:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander(f"{emojis['computer']} DATA PREVIEW - FIRST 10 RECORDS"):
        st.dataframe(df.head(10), use_container_width=True)
else:
    st.markdown(f"""
    <div class="terminal-block" style="border-color: #ff0000;">
        <p style="color: #ff0000;">{emojis['warning']} <strong>NO ACTIVE DATASET</strong></p>
        <p style="color: #00ffff;">Navigate to sidebar {emojis['lightning']} Upload Dataset to initialize data stream</p>
    </div>
    """, unsafe_allow_html=True)

# Command Reference
st.markdown("---")
st.markdown(f"### {emojis['terminal']} SYSTEM COMMANDS")

st.markdown(f"""
<div class="terminal-block">
    <pre style="color: #00ff41; margin: 0;">
┌─[root@datasense]─[~/neural-terminal]
└──╼ $ Available Modules:
    
    {emojis['lightning']} upload_data      - Initialize data stream from external source
    {emojis['chart']} visualize        - Generate visual analytics and graphs
    {emojis['target']} analyze          - Perform deep statistical analysis
    {emojis['wrench']} clean_data       - Sanitize and prepare data matrix
    {emojis['robot']} automl            - Train autonomous ML models
    {emojis['computer']} chat_interface   - Natural language data queries
    {emojis['brain']} ai_agent          - Deploy autonomous AI analyst
    {emojis['data']} generate_report   - Export comprehensive analysis report
    </pre>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #00ff41; padding: 2rem; font-family: 'VT323', monospace;">
    <p style="font-size: 1.2rem;">╔═══════════════════════════════════════════════╗</p>
    <p style="font-size: 1rem;">  DataSense AI v2.0 {emojis['skull']} TERMINAL EDITION  </p>
    <p style="font-size: 0.9rem; color: #00ffff;">  Powered by Streamlit & Gemini AI | © 2024  </p>
    <p style="font-size: 1.2rem;">╚═══════════════════════════════════════════════╝</p>
    <p style="font-size: 0.8rem; color: #00ff41; margin-top: 1rem;">
        {emojis['unlock']} ACCESS GRANTED {emojis['unlock']}
    </p>
</div>
""", unsafe_allow_html=True)

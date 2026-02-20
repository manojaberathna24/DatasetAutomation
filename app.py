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

# Main Terminal Header
st.markdown("""
<div class="terminal-header">
    <h1>DataSense AI - Advanced Analytics</h1>
    <p>Professional Data Intelligence & Machine Learning System</p>
    <p>System Status: ACTIVE</p>
</div>
""", unsafe_allow_html=True)

# System Initialization Message
st.markdown("""
<div class="terminal-block">
    <p><strong>SYSTEM INITIALIZED</strong> | Timestamp: ACTIVE SESSION</p>
    <p><strong>ACCESS LEVEL:</strong> ADMINISTRATOR</p>
    <p><strong>CONNECTION:</strong> SECURE</p>
</div>
""", unsafe_allow_html=True)

# Welcome Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Data Science Workspace")
    st.markdown("""
    <div class="terminal-block">
    <p><strong>DataSense AI</strong> transforms raw data into actionable insights 
    using modern analytics, machine learning, and artificial intelligence.</p>
    
    <h4>CORE CAPABILITIES:</h4>
    <ul>
        <li><strong>DATA UPLOAD & MANAGEMENT</strong> - Support for CSV, Excel, JSON, ZIP, RAR files</li>
        <li><strong>VISUAL ANALYTICS</strong> - Interactive charts: Bar, Pie, Line, Scatter, Heatmap, Boxplot</li>
        <li><strong>DEEP ANALYSIS</strong> - Automated profiling, outlier detection, statistical analysis</li>
        <li><strong>DATA SANITIZATION</strong> - Handle missing values, duplicates, and outliers</li>
        <li><strong>AUTO ML ENGINE</strong> - Train models for Classification, Regression, Clustering</li>
        <li><strong>AI CHAT INTERFACE</strong> - Natural language queries powered by AI Models</li>
        <li><strong>AUTONOMOUS AGENT</strong> - Automated business insights and recommendations</li>
        <li><strong>REPORT GENERATION</strong> - Professional PDF reports with complete analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### System Metrics")
    
    # Display stats
    num_datasets = len(st.session_state.datasets)
    num_cleaned = len(st.session_state.cleaned_datasets)
    num_models = len(st.session_state.ml_models)
    
    st.markdown(f"""
    <div class="metric-container">
        <p class="metric-label" style="color: var(--text-muted); font-size: 0.9rem; margin: 0; font-weight: 500;">DATABASE ENTRIES</p>
        <p class="metric-value" style="color: var(--text-main); font-size: 2.5rem; font-weight: 700; margin: 0.2rem 0;">{num_datasets}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-container " style="margin-top: 1rem;">
        <p class="metric-label" style="color: var(--text-muted); font-size: 0.9rem; margin: 0; font-weight: 500;">SANITIZED DATA</p>
        <p class="metric-value" style="color: var(--text-main); font-size: 2.5rem; font-weight: 700; margin: 0.2rem 0;">{num_cleaned}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-container" style="margin-top: 1rem;">
        <p class="metric-label" style="color: var(--text-muted); font-size: 0.9rem; margin: 0; font-weight: 500;">NEURAL MODELS</p>
        <p class="metric-value" style="color: var(--text-main); font-size: 2.5rem; font-weight: 700; margin: 0.2rem 0;">{num_models}</p>
    </div>
    """, unsafe_allow_html=True)

# Quick Start Protocol
st.markdown("---")
st.markdown("### Quick Start Guide")

cols = st.columns(4)

with cols[0]:
    st.markdown("""
    <div class="terminal-block">
        <h4>STEP 1: UPLOAD</h4>
        <p>Initialize workspace with your dataset (CSV, Excel, JSON, Archive)</p>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("""
    <div class="terminal-block">
        <h4>STEP 2: ANALYZE</h4>
        <p>Visualize and decode data patterns using tools</p>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown("""
    <div class="terminal-block">
        <h4>STEP 3: SANITIZE</h4>
        <p>Clean and prepare the data for better performance</p>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    st.markdown("""
    <div class="terminal-block">
        <h4>STEP 4: DEPLOY AI</h4>
        <p>Generate insights and train automated ML models</p>
    </div>
    """, unsafe_allow_html=True)

# Current Dataset Info
if st.session_state.current_dataset:
    st.markdown("---")
    st.markdown("### Active Workspace")
    df = st.session_state.datasets[st.session_state.current_dataset]
    
    st.markdown(f"""
    <div class="terminal-block">
        <p><strong>DATASET ID:</strong> {st.session_state.current_dataset}</p>
        <p><strong>RECORDS:</strong> {len(df):,} rows | <strong>FIELDS:</strong> {len(df.columns)} columns</p>
        <p><strong>MEMORY:</strong> {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("DATA PREVIEW - FIRST 10 RECORDS"):
        st.dataframe(df.head(10), use_container_width=True)
else:
    st.markdown("""
    <div class="terminal-block stWarning">
        <p><strong>No Workspace Initialized</strong></p>
        <p>Navigate to "Upload Dataset" to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); padding: 2rem; font-family: 'Inter', sans-serif;">
    <p style="font-size: 1.2rem; font-weight: 500;">DataSense AI - Professional Edition</p>
    <p style="font-size: 0.9rem;">Powered by Streamlit, Gemini & OpenRouter</p>
</div>
""", unsafe_allow_html=True)

"""
AI Agent - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_chat import AIChat
from utils.analysis import DataAnalyzer
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
import pandas as pd
import json

st.set_page_config(page_title="Autonomous AI Agent", page_icon="👁️", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown("""
<div class="terminal-header">
    <h1>AUTONOMOUS AI AGENT</h1>
    <p>Intelligent Business Analytics & Insight Generation System</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Check if dataset exists
if 'current_dataset' not in st.session_state or not st.session_state.current_dataset:
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    st.markdown("""
    <div class="terminal-block stWarning">
        <p><strong>NO ACTIVE DATASET</strong></p>
        <p>Upload a dataset first to activate AI agent</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.datasets[st.session_state.current_dataset]

# Initialize AI Chat
if 'ai_chat' not in st.session_state:
    st.session_state.ai_chat = AIChat()

ai_chat = st.session_state.ai_chat
analyzer = DataAnalyzer(df)

# Sidebar
with st.sidebar:
    st.markdown("### Analysis Options")
    
    st.markdown("Click the button below to get a comprehensive AI analysis of your data.")
    
    if st.button("Analyze Dataset", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Columns:** {len(df.columns)}")

# Main content
if st.session_state.get('run_analysis'):
    with st.spinner("AI Agent is analyzing your data... This may take a moment..."):
        insights = ai_chat.get_business_insights(df)
        st.session_state.ai_insights = insights
        st.session_state.run_analysis = False

if 'ai_insights' in st.session_state:
    insights = st.session_state.ai_insights
    
    # Key Insights
    st.markdown("### Key Insights")
    
    if insights.get('key_insights'):
        for i, insight in enumerate(insights['key_insights'], 1):
            st.markdown(f"**{i}.** {insight}")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### Business Recommendations")
    
    if insights.get('recommendations'):
        for i, rec in enumerate(insights['recommendations'], 1):
            st.success(f"**{i}.** {rec}")
    
    st.markdown("---")
    
    # Risks
    st.markdown("### Risk Assessment")
    
    if insights.get('risks'):
        for i, risk in enumerate(insights['risks'], 1):
            st.warning(f"**{i}.** {risk}")
    
    st.markdown("---")
    
    # Trends
    st.markdown("### Trends Analysis")
    
    if insights.get('trends'):
        col1, col2 = st.columns(2)
        mid = len(insights['trends']) // 2
        
        with col1:
            for trend in insights['trends'][:mid]:
                st.info(f"• {trend}")
        
        with col2:
            for trend in insights['trends'][mid:]:
                st.info(f"• {trend}")
    
    st.markdown("---")
    
    # Next Steps
    st.markdown("### Recommended Next Steps")
    
    if insights.get('next_steps'):
        for i, step in enumerate(insights['next_steps'], 1):
            st.markdown(f"**{i}.** {step}")
    
    st.markdown("---")
    
    # Export Insights
    st.markdown("### Export Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download as JSON", use_container_width=True):
            st.download_button(
                label="Download JSON",
                data=json.dumps(insights, indent=2),
                file_name=f"{st.session_state.current_dataset}_insights.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Copy to Clipboard", use_container_width=True):
            insight_text = "# AI Business Insights\n\n"
            insight_text += "## Key Insights\n"
            for i, insight in enumerate(insights.get('key_insights', []), 1):
                insight_text += f"{i}. {insight}\n"
            insight_text += "\n## Recommendations\n"
            for i, rec in enumerate(insights.get('recommendations', []), 1):
                insight_text += f"{i}. {rec}\n"
            
            st.code(insight_text, language="markdown")
    
    with col3:
        if st.button("Re-analyze", use_container_width=True):
            del st.session_state.ai_insights
            st.session_state.run_analysis = True
            st.rerun()

else:
    # Show placeholder
    st.info("Click 'Analyze Dataset' in the sidebar to start AI analysis")
    
    st.markdown("---")
    st.markdown("### What the AI Agent Does")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Analysis Capabilities:**
        - 📊 Comprehensive data profiling
        - 🔍 Pattern and anomaly detection
        - 📈 Trend identification
        - 🎯 Business metric analysis
        """)
    
    with col2:
        st.markdown("""
        **Deliverables:**
        - 💡 Actionable recommendations
        - ⚠️ Risk assessment
        - 🚀 Strategic next steps
        - 📋 Exportable insights report
        """)
    
    st.markdown("---")
    
    # Show basic stats while waiting
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        st.metric("Numeric Columns", numeric_cols)
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    # Basic analysis preview
    st.markdown("---")
    st.markdown("### Quick Analysis")
    
    with st.expander("Column Information"):
        col_types = analyzer.get_column_types()
        col_df = pd.DataFrame([
            {
                'Column': col,
                'Type': info['type'],
                'Unique': info['unique'],
                'Missing %': f"{info['null_percentage']:.1f}%"
            }
            for col, info in col_types.items()
        ])
        st.dataframe(col_df, use_container_width=True)
    
    with st.expander("Summary Statistics"):
        summary = analyzer.get_summary_statistics()
        if not summary.empty:
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No numeric columns for summary statistics")

st.markdown("---")
st.markdown("""
### Tips for Best Results

1. **Clean Data First**: Consider cleaning your data before running AI analysis
2. **Specific Questions**: The more structured your data, the better the insights
3. **Business Context**: AI provides general insights - apply your domain knowledge
4. **Iterative Process**: Use insights to guide further data exploration
""")

"""
Chat with Data Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_chat import AIChat
from utils.visualization import DataVisualizer
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
import pandas as pd

st.set_page_config(page_title="AI Chat Interface", page_icon="💻", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown("""
<div class="terminal-header">
    <h1>AI CHAT INTERFACE</h1>
    <p>Natural Language Data Query System</p>
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
        <p>Upload a dataset first to initialize chat interface</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.datasets[st.session_state.current_dataset]

# Initialize AI Chat
if 'ai_chat' not in st.session_state:
    st.session_state.ai_chat = AIChat()

ai_chat = st.session_state.ai_chat

# Sidebar
with st.sidebar:
    st.markdown("### Suggested Questions")
    
    suggestions = [
        "How many rows are in the dataset?",
        "What are the column names?",
        "Show me the first 10 rows",
        "What is the average of [column]?",
        "Which column has the most missing values?",
        "Show me a summary of the data",
        "What are the unique values in [column]?",
        "Plot a histogram of [column]"
    ]
    
    for suggestion in suggestions:
        if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
            st.session_state.current_question = suggestion
    
    st.markdown("---")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Columns:** {len(df.columns)}")

# Chat interface
st.markdown("### Chat")

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for i, message in enumerate(st.session_state.chat_history):
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.write(message['content'])
    else:
        with st.chat_message("assistant"):
            st.write(message['answer'])
            
            # Show code if available
            if message.get('code'):
                with st.expander("Generated Code"):
                    st.code(message['code'], language='python')
            
            # Show chart if available
            if message.get('chart'):
                st.plotly_chart(message['chart'], use_container_width=True)

# Input box
question = st.chat_input("Ask a question about your data...")

# Handle pre-filled suggestion
if 'current_question' in st.session_state and st.session_state.current_question:
    question = st.session_state.current_question
    st.session_state.current_question = None

if question:
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': question
    })
    
    # Get AI response
    with st.spinner("Thinking..."):
        response = ai_chat.query_data(df, question)
    
    # Create chart if suggested
    chart = None
    if response.get('chart_type') and response['chart_type'] != 'none':
        viz = DataVisualizer(df)
        
        try:
            if response['chart_type'] == 'bar' and 'data' in response:
                # Use response data if available
                pass
            elif response['chart_type'] == 'pie':
                # Try to create pie chart from first column
                if len(df.columns) > 0:
                    chart = viz.create_pie_chart(df.columns[0])
            # Add more chart types as needed
        except:
            pass
    
    # Add assistant response to history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'answer': response.get('answer', 'Sorry, I could not process that question.'),
        'code': response.get('code'),
        'chart': chart
    })
    
    st.rerun()

# Quick Stats Panel
st.markdown("---")
st.markdown("### Quick Dataset Stats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Rows", f"{len(df):,}")

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    missing = df.isnull().sum().sum()
    st.metric("Missing Values", f"{missing:,}")

with col4:
    duplicates = len(df[df.duplicated()])
    st.metric("Duplicates", f"{duplicates:,}")

# Data preview
with st.expander("Quick Data Preview"):
    st.dataframe(df.head(10), use_container_width=True)

# Column info
with st.expander("Column Information"):
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null': df.count(),
        'Unique': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

# Instructions
st.markdown("---")
st.markdown("""
### How to Use

1. **Ask Questions**: Type your question in natural language in the chat box
2. **Get Answers**: The AI will analyze your data and provide answers
3. **View Code**: Expand the code section to see the Python code generated
4. **Visualize**: The AI may generate charts based on your questions
5. **Suggestions**: Click the suggested questions in the sidebar for quick queries

**Example Questions:**
- "What is the correlation between column A and column B?"
- "Show me the top 10 values in column X"
- "What percentage of data is missing in column Y?"
- "Plot a scatter chart of sales vs profit"
- "Which category has the highest average revenue?"
""")

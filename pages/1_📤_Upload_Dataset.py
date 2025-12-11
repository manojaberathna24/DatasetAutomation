"""
Dataset Upload Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from utils.terminal_theme import get_terminal_css, get_hacker_emojis, create_terminal_progress_bar
import pandas as pd

st.set_page_config(page_title="Upload Dataset", page_icon="⚡", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)

# Get hacker emojis
emojis = get_hacker_emojis()

# Initialize data loader
loader = DataLoader()

# Terminal Header
st.markdown(f"""
<div class="terminal-header">
    <h1>{emojis['lightning']} DATA UPLOAD TERMINAL {emojis['lightning']}</h1>
    <p>{emojis['satellite']} Initialize Data Stream | Maximum File Size: 200MB</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# File uploader
st.markdown(f"### {emojis['data']} UPLOAD DATA SOURCE")
st.markdown(f"""
<div class="terminal-block">
    <p>{emojis['computer']} <strong>SUPPORTED FORMATS:</strong> CSV | EXCEL (xlsx, xls) | JSON</p>
    <p>{emojis['fire']} <strong>MAX SIZE:</strong> 200 MB | <strong>STATUS:</strong> READY</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    f"{emojis['unlock']} SELECT FILE TO UPLOAD",
    type=['csv', 'xlsx', 'xls', 'json'],
    help="Upload your dataset file (CSV, Excel, or JSON format)"
)

if uploaded_file is not None:
    with st.spinner(f"{emojis['gear']} PROCESSING DATA STREAM..."):
        # Load the file
        df = loader.load_file(uploaded_file)
        
        if df is not None:
            # Get file info
            file_info = loader.get_file_info(df, uploaded_file.name)
            
            # Display success message
            st.markdown(f"""
            <div class="terminal-block" style="border-color: #00ff41;">
                <p style="color: #00ff41;">{emojis['fire']} <strong>UPLOAD SUCCESSFUL:</strong> {uploaded_file.name}</p>
                <p style="color: #00ffff;">{emojis['unlock']} Data stream initialized and ready for analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display file information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">{emojis['data']} RECORDS</p>
                    <p style="color: #00ffff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{file_info['rows']:,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">{emojis['target']} FIELDS</p>
                    <p style="color: #00ffff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{file_info['columns']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">{emojis['gear']} SIZE</p>
                    <p style="color: #00ffff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{file_info['size_mb']:.2f}MB</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                missing = sum(file_info['missing_values'].values())
                color = "#ff0000" if missing > 0 else "#00ff41"
                st.markdown(f"""
                <div class="metric-container">
                    <p style="color: #00ff41; font-size: 0.9rem; margin: 0;">{emojis['warning']} MISSING</p>
                    <p style="color: {color}; font-size: 2rem; font-weight: 700; margin: 0.2rem 0;">{missing:,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Preview data
            st.markdown(f"### {emojis['computer']} DATA PREVIEW")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Column information
            st.markdown(f"### {emojis['data']} FIELD ANALYSIS")
            
            col_df = pd.DataFrame({
                'Field Name': file_info['column_names'],
                'Data Type': [file_info['dtypes'][col] for col in file_info['column_names']],
                'Missing Values': [file_info['missing_values'][col] for col in file_info['column_names']],
                'Missing %': [f"{(file_info['missing_values'][col] / file_info['rows']) * 100:.1f}%" 
                             for col in file_info['column_names']]
            })
            
            st.dataframe(col_df, use_container_width=True)
            
            st.markdown("---")
            
            # Save to session state
            col1, col2 = st.columns([3, 1])
            
            with col1:
                dataset_name = st.text_input(
                    f"{emojis['terminal']} DATASET IDENTIFIER",
                    value=uploaded_file.name.rsplit('.', 1)[0],
                    help="Enter a unique identifier for this dataset"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"{emojis['unlock']} SAVE TO DATABASE", type="primary", use_container_width=True):
                    # Save to session state
                    st.session_state.datasets[dataset_name] = df
                    st.session_state.current_dataset = dataset_name
                    
                    # Save to storage (GCS or local)
                    filepath = loader.save_to_storage(df, dataset_name)
                    
                    if filepath:
                        st.markdown(f"""
                        <div class="terminal-block" style="border-color: #00ff41;">
                            <p style="color: #00ff41;">{emojis['fire']} <strong>DATABASE UPDATE SUCCESSFUL</strong></p>
                            <p style="color: #00ffff;">{emojis['data']} Dataset ID: <strong>{dataset_name}</strong></p>
                            <p style="color: #00ffff;">{emojis['gear']} Location: {filepath}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.snow()
                    else:
                        st.markdown(f"""
                        <div class="terminal-block" style="border-color: #ff0000;">
                            <p style="color: #ff0000;">{emojis['warning']} WARNING: Dataset saved to session only</p>
                            <p style="color: #00ffff;">Persistent storage unavailable</p>
                        </div>
                        """, unsafe_allow_html=True)

# Show existing datasets
st.markdown("---")
st.markdown(f"### {emojis['data']} DATABASE INVENTORY")

if st.session_state.datasets:
    for name, data in st.session_state.datasets.items():
        with st.expander(f"{emojis['target']} {name}"):
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(f"{emojis['data']} Records", f"{len(data):,}")
            col2.metric(f"{emojis['target']} Fields", len(data.columns))
            col3.metric(f"{emojis['gear']} Size", f"{data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            is_current = name == st.session_state.current_dataset
            status = f"{emojis['fire']} ACTIVE" if is_current else f"{emojis['lock']} INACTIVE"
            col4.markdown(f"**Status:** {status}")
            
            # Action buttons
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if not is_current:
                    if st.button(f"{emojis['unlock']} ACTIVATE", key=f"activate_{name}"):
                        st.session_state.current_dataset = name
                        st.rerun()
            
            with col_b:
                if st.button(f"{emojis['computer']} PREVIEW", key=f"preview_{name}"):
                    st.dataframe(data.head(), use_container_width=True)
            
            with col_c:
                if st.button(f"{emojis['danger']} DELETE", key=f"delete_{name}"):
                    del st.session_state.datasets[name]
                    if st.session_state.current_dataset == name:
                        st.session_state.current_dataset = None
                    st.rerun()
else:
    st.markdown(f"""
    <div class="terminal-block" style="border-color: #ff0000;">
        <p style="color: #ff0000;">{emojis['warning']} <strong>DATABASE EMPTY</strong></p>
        <p style="color: #00ffff;">Upload a file above to initialize data stream</p>
    </div>
    """, unsafe_allow_html=True)

# Instructions
st.markdown("---")
st.markdown(f"### {emojis['terminal']} OPERATION PROTOCOL")

st.markdown(f"""
<div class="terminal-block">
    <pre style="color: #00ff41; margin: 0;">
┌─[root@datasense]─[~/upload-terminal]
└──╼ $ Operation Sequence:

    {emojis['lightning']} STEP 1: Select file using upload interface
    {emojis['computer']} STEP 2: Review data preview and field analysis
    {emojis['unlock']} STEP 3: Assign unique dataset identifier
    {emojis['fire']} STEP 4: Save to database
    {emojis['target']} STEP 5: Navigate to analysis modules via sidebar

    SUPPORTED FORMATS:
    • CSV (.csv)
    • Excel (.xlsx, .xls)
    • JSON (.json)

    MAXIMUM FILE SIZE: 200 MB
    </pre>
</div>
""", unsafe_allow_html=True)

"""
Enhanced Streamlit App Styling - Light Pastel Theme
"""
import streamlit as st

# Apply light pastel color CSS
st.markdown("""
<style>
    /* Light Pastel Color Scheme */
    :root {
        --primary-green: #81c784;
        --light-green: #c8e6c9;
        --primary-blue: #64b5f6;
        --light-blue: #bbdefb;
        --primary-purple: #ba68c8;
        --light-purple: #e1bee7;
        --cream: #fff9e6;
        --light-gray: #f5f5f5;
        --medium-gray: #e0e0e0;
        --dark-gray: #757575;
        --white: #ffffff;
        --border-color: #e0e0e0;
        --text-color: #424242;
    }
    
    /* Remove default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container - Light cream background */
    .main {
        background-color: var(--cream);
    }
    
    /* Headers - Light green and blue accents */
    h1 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-bottom: 3px solid var(--primary-green);
        padding-bottom: 12px;
        margin-bottom: 24px;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: var(--text-color) !important;
        font-weight: 500 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-top: 24px;
        border-left: 4px solid var(--primary-blue);
        padding-left: 12px;
    }
    
    h3 {
        color: var(--dark-gray) !important;
        font-weight: 500 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric cards - Soft pastel backgrounds */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        color: var(--text-color);
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px;
        color: var(--primary-green);
        font-weight: 500;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--light-blue) 0%, var(--light-green) 100%);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buttons - Light green/blue gradient */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--primary-blue) 100%);
        color: var(--white);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-purple) 100%);
        font-weight: 600;
    }
    
    .stButton>button[kind="secondary"] {
        background: var(--white);
        color: var(--primary-blue);
        border: 2px solid var(--primary-blue);
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: var(--light-blue);
    }
    
    /* Download buttons - Light purple */
    .stDownloadButton>button {
        background: linear-gradient(135deg, var(--light-purple) 0%, var(--light-blue) 100%);
        color: var(--text-color);
        border: 2px solid var(--primary-purple);
        border-radius: 8px;
        font-weight: 500;
        padding: 10px 20px;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--primary-blue) 100%);
        color: var(--white);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar - Light green background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--light-green) 0%, var(--light-blue) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
       color: var(--text-color) !important;
    }
    
    /* Alerts - Soft pastel backgrounds */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid var(--primary-blue);
        background-color: var(--light-blue);
        padding: 16px;
        color: var(--text-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stSuccess {
        border-left-color: var(--primary-green);
        background-color: var(--light-green);
    }
    
    .stInfo {
        border-left-color: var(--primary-blue);
        background-color: var(--light-blue);
    }
    
    .stWarning {
        border-left-color: #ffb74d;
        background-color: #fff3e0;
    }
    
    .stError {
        border-left-color: #e57373;
        background-color: #ffebee;
    }
    
    /* Tabs - Soft colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: var(--light-gray);
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--white);
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        color: var(--dark-gray);
        border: 1px solid var(--border-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--primary-blue) 100%);
        color: var(--white);
        border: none;
    }
    
    /* Expander - Light backgrounds */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, var(--light-blue) 0%, var(--light-purple) 100%);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-weight: 500;
        color: var(--text-color);
        padding: 12px 16px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-purple) 100%);
        color: var(--white);
    }
    
    /* Dataframe - Light headers */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--primary-blue) 100%) !important;
        color: var(--white) !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: var(--light-blue);
    }
    
    .dataframe tbody tr:hover {
        background-color: var(--light-green);
    }
    
    /* Input fields - Light borders and focus */
    .stSelectbox>div>div, 
    .stMultiSelect>div>div,
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: var(--white);
    }
    
    .stSelectbox>div>div:focus-within,
    .stMultiSelect>div>div:focus-within,
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 2px var(--light-blue);
    }
    
    /* Slider - Gradient */
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, var(--primary-green) 0%, var(--primary-blue) 100%);
    }
    
    .stSlider>div>div>div {
        background-color: var(--light-gray);
    }
    
    /* Radio buttons */
    .stRadio>div {
        background-color: var(--white);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Checkbox */
    .stCheckbox {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* File uploader - Light gradient */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--primary-blue);
        border-radius: 12px;
        background: linear-gradient(135deg, var(--light-blue) 0%, var(--light-purple) 100%);
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-green);
        background: linear-gradient(135deg, var(--light-green) 0%, var(--light-blue) 100%);
    }
    
    /* Progress bars - Gradient */
    .stProgress>div>div>div>div {
        background: linear-gradient(90deg, var(--primary-green) 0%, var(--primary-blue) 100%);
    }
    
    .stProgress>div>div {
        background-color: var(--light-gray);
    }
    
    /* Divider */
    hr {
        border-color: var(--border-color);
        margin: 24px 0;
    }
    
    /* Code blocks */
    code {
        background-color: var(--light-purple);
        color: var(--text-color);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        border: 1px solid var(--border-color);
    }
    
    /* Links - Blue */
    .stMarkdown a {
        color: var(--primary-blue);
        text-decoration: none;
        font-weight: 500;
    }
    
    .stMarkdown a:hover {
        color: var(--primary-purple);
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

"""
Kali Linux Terminal Theme Manager
Provides centralized styling, animations, and effects for the hacker interface
"""

"""
Professional Dark Theme Manager
Provides centralized styling for a sleek, modern, and professional data analysis interface
"""

def get_terminal_css():
    """Returns comprehensive CSS for professional dark theme"""
    return """
    <style>
        /* Import Modern Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Color Palette */
        :root {
            --primary: #3b82f6;      /* Professional blue */
            --primary-hover: #2563eb;
            --bg-color: #0f172a;     /* Slate 900 background */
            --panel-bg: #1e293b;     /* Slate 800 cards */
            --text-main: #f8fafc;    /* Slate 50 text */
            --text-muted: #94a3b8;   /* Slate 400 text */
            --border-color: #334155; /* Slate 700 border */
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #0ea5e9;
        }
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Main Background */
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-main);
        }
        
        /* Headers */
        .terminal-header {
            background-color: var(--panel-bg);
            padding: 2rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        .terminal-header h1 {
            color: var(--text-main);
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.025em;
        }
        
        .terminal-header p {
            color: var(--text-muted);
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            font-weight: 400;
        }
        
        /* Content Blocks */
        .terminal-block {
            background-color: var(--panel-bg);
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--primary);
            border-radius: 6px;
            margin: 1rem 0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--primary) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        }
        
        .stButton>button:hover {
            background-color: var(--primary-hover) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--panel-bg) !important;
            border-right: 1px solid var(--border-color);
        }
        
        [data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--text-main) !important;
            font-size: 1.6rem !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }
        
        .metric-container {
            background-color: var(--panel-bg);
            padding: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            margin-bottom: 0.5rem;
        }
        
        /* Data Tables */
        .dataframe {
            border: 1px solid var(--border-color) !important;
            border-radius: 6px;
            background-color: var(--bg-color) !important;
        }
        
        .dataframe th {
            background-color: var(--panel-bg) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border-color) !important;
            font-weight: 600 !important;
            padding: 0.75rem !important;
        }
        
        .dataframe td {
            color: var(--text-muted) !important;
            border: 1px solid var(--border-color) !important;
            padding: 0.75rem !important;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background-color: var(--panel-bg);
            border: 1px dashed var(--border-color);
            border-radius: 8px;
            padding: 2rem;
            transition: border-color 0.2s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary);
        }
        
        /* Success/Info/Warning Messages */
        .stSuccess {
            background-color: rgba(16, 185, 129, 0.1) !important;
            border-left: 4px solid var(--success) !important;
            color: var(--text-main) !important;
            padding: 1rem;
            border-radius: 0 6px 6px 0;
        }
        
        .stInfo {
            background-color: rgba(14, 165, 233, 0.1) !important;
            border-left: 4px solid var(--info) !important;
            color: var(--text-main) !important;
            padding: 1rem;
            border-radius: 0 6px 6px 0;
        }
        
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1) !important;
            border-left: 4px solid var(--warning) !important;
            color: var(--text-main) !important;
            padding: 1rem;
            border-radius: 0 6px 6px 0;
        }
        
        .stError {
            background-color: rgba(239, 68, 68, 0.1) !important;
            border-left: 4px solid var(--danger) !important;
            color: var(--text-main) !important;
            padding: 1rem;
            border-radius: 0 6px 6px 0;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--panel-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-main) !important;
            border-radius: 6px;
            font-weight: 500;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #273549 !important;
        }
        
        /* Inputs */
        .stTextInput input, .stSelectbox select {
            background-color: var(--bg-color) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-main) !important;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
        }
        
        .stTextInput input:focus, .stSelectbox select:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background-color: var(--primary) !important;
        }
        
        /* Markdown Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-main) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
        }
        
        /* Paragraphs */
        p {
            color: var(--text-muted) !important;
        }
        
        /* Links */
        a {
            color: var(--primary) !important;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Code Blocks */
        code {
            background-color: var(--bg-color) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border-color);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
            font-size: 0.875em;
        }
        
        /* Horizontal Rule */
        hr {
            border-color: var(--border-color) !important;
            margin: 2rem 0;
        }
    </style>
    """

def get_typing_animation_js():
    """Returns empty JS as animations are removed in professional theme"""
    return ""

def get_ascii_banner():
    """Returns empty banner to keep it professional"""
    return ""

def get_hacker_emojis():
    """Returns dictionary of empty strings to remove emojis throughout the app"""
    return {
        'skull': '',
        'fire': '',
        'lightning': '',
        'alien': '',
        'target': '',
        'lock': '',
        'unlock': '',
        'key': '',
        'warning': '',
        'danger': '',
        'computer': '',
        'terminal': '',
        'data': '',
        'chart': '',
        'brain': '',
        'robot': '',
        'gear': '',
        'wrench': '',
        'shield': '',
        'satellite': ''
    }

def create_terminal_progress_bar(progress, width=50):
    """
    Creates a standard Unicode progress bar for the professional theme
    """
    filled = int(width * progress)
    empty = width - filled
    bar = '■' * filled + '□' * empty
    percentage = int(progress * 100)
    return f"[{bar}] {percentage}%"

def get_matrix_rain_html():
    """Returns empty string to remove matrix rain effect"""
    return ""

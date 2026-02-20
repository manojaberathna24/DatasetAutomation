"""
Professional animations and effects for Streamlit
"""
import streamlit as st
import time

def show_success_animation(message="Operation completed successfully"):
    """Show professional success animation"""
    st.markdown(f"""
    <div class="success-animation">
        <div class="checkmark-circle">
            <div class="checkmark"></div>
        </div>
        <p class="success-message">{message}</p>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes drawCheck {{
            0% {{ stroke-dashoffset: 100; }}
            100% {{ stroke-dashoffset: 0; }}
        }}
        
        @keyframes scaleIn {{
            0% {{ transform: scale(0); }}
            50% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
        
        .success-animation {{
            text-align: center;
            padding: 20px;
            animation: fadeIn 0.5s ease-out;
        }}
        
        .checkmark-circle {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 3px solid #1a1a1a;
            margin: 0 auto 20px;
            position: relative;
            animation: scaleIn 0.4s ease-out;
        }}
        
        .checkmark {{
            width: 40px;
            height: 20px;
            border-left: 4px solid #1a1a1a;
            border-bottom: 4px solid #1a1a1a;
            transform: rotate(-45deg);
            position: absolute;
            top: 25px;
            left: 17px;
            animation: drawCheck 0.3s 0.3s ease-out forwards;
        }}
        
        .success-message {{
            font-size: 18px;
            color: #1a1a1a;
            font-weight: 500;
            margin: 0;
            animation: fadeIn 0.5s 0.3s ease-out both;
        }}
    </style>
    """, unsafe_allow_html=True)
    time.sleep(2)  # Show for 2 seconds

def show_loading_bar(message="Processing..."):
    """Show professional loading bar"""
    st.markdown(f"""
    <div class="loading-container">
        <p class="loading-message">{message}</p>
        <div class="loading-bar">
            <div class="loading-progress"></div>
        </div>
    </div>
    <style>
        @keyframes loading {{
            0% {{ width: 0%; }}
            100% {{ width: 100%; }}
        }}
        
        .loading-container {{
            text-align: center;
            padding: 20px;
        }}
        
        .loading-message {{
            font-size: 16px;
            color: #4a4a4a;
            margin-bottom: 15px;
            font-weight: 500;
        }}
        
        .loading-bar {{
            width: 100%;
            height: 4px;
            background-color: #e0e0e0;
            border-radius: 2px;
            overflow: hidden;
        }}
        
        .loading-progress {{
            height: 100%;
            background-color: #1a1a1a;
            animation: loading 2s ease-in-out infinite;
        }}
    </style>
    """, unsafe_allow_html=True)

def fade_in_element():
    """Add fade-in animation to elements"""
    st.markdown("""
    <style>
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .element-container {{
            animation: fadeInUp 0.6s ease-out;
        }}
        
        [data-testid="stMetric"] {{
            animation: fadeInUp 0.5s ease-out;
        }}
        
        .stDataFrame {{
            animation: fadeInUp 0.6s ease-out;
        }}
    </style>
    """, unsafe_allow_html=True)

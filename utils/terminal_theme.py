"""
Kali Linux Terminal Theme Manager
Provides centralized styling, animations, and effects for the hacker interface
"""

def get_terminal_css():
    """Returns comprehensive CSS for Kali Linux terminal theme"""
    return """
    <style>
        /* Import Terminal Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=VT323&family=Courier+Prime&display=swap');
        
        /* Color Palette */
        :root {
            --matrix-green: #00ff41;
            --matrix-cyan: #00ffff;
            --matrix-red: #ff0000;
            --terminal-bg: #0a0a0a;
            --terminal-border: #00ff41;
            --glow-color: rgba(0, 255, 65, 0.5);
        }
        
        /* Global Styles */
        * {
            font-family: 'Courier Prime', 'Courier New', 'Consolas', 'Monaco', monospace !important;
        }
        
        /* Main Background */
        .stApp {
            background: #0a0a0a;
            background-image: 
                repeating-linear-gradient(
                    0deg,
                    rgba(0, 255, 65, 0.03) 0px,
                    transparent 1px,
                    transparent 2px,
                    rgba(0, 255, 65, 0.03) 3px
                );
            color: var(--matrix-green);
        }
        
        /* Scanline Effect */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.15),
                rgba(0, 0, 0, 0.15) 1px,
                transparent 1px,
                transparent 2px
            );
            pointer-events: none;
            z-index: 1000;
            animation: scanline 8s linear infinite;
        }
        
        @keyframes scanline {
            0% { transform: translateY(0); }
            100% { transform: translateY(10px); }
        }
        
        /* CRT Screen Effect */
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(ellipse at center, transparent 0%, rgba(0, 0, 0, 0.3) 100%);
            pointer-events: none;
            z-index: 999;
        }
        
        /* Terminal Header */
        .terminal-header {
            background: linear-gradient(135deg, #001a00 0%, #003300 100%);
            padding: 2rem;
            border: 2px solid var(--matrix-green);
            border-radius: 5px;
            margin-bottom: 2rem;
            box-shadow: 
                0 0 20px rgba(0, 255, 65, 0.3),
                inset 0 0 20px rgba(0, 255, 65, 0.1);
            position: relative;
            overflow: hidden;
            animation: fadeIn 1s ease-in;
        }
        
        .terminal-header::before {
            content: "";
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, var(--matrix-green), var(--matrix-cyan), var(--matrix-green));
            z-index: -1;
            filter: blur(10px);
            opacity: 0.3;
            animation: pulse 3s ease-in-out infinite;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.6; }
        }
        
        .terminal-header h1 {
            color: var(--matrix-green);
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 
                0 0 10px var(--glow-color),
                0 0 20px var(--glow-color),
                0 0 30px var(--glow-color);
            font-family: 'VT323', monospace !important;
            letter-spacing: 3px;
        }
        
        .terminal-header p {
            color: var(--matrix-cyan);
            font-size: 1.2rem;
            margin: 0.5rem 0 0 0;
            text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
        }
        
        /* ASCII Art */
        .ascii-art {
            color: var(--matrix-green);
            font-family: 'VT323', monospace !important;
            font-size: 0.8rem;
            line-height: 1.2;
            text-shadow: 0 0 5px var(--glow-color);
            white-space: pre;
        }
        
        /* Terminal Command Blocks */
        .terminal-block {
            background: rgba(0, 20, 0, 0.8);
            padding: 1.5rem;
            border: 1px solid var(--matrix-green);
            border-left: 4px solid var(--matrix-green);
            border-radius: 3px;
            margin: 1rem 0;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
            position: relative;
        }
        
        .terminal-block::before {
            content: "▶";
            position: absolute;
            left: 0.5rem;
            top: 0.5rem;
            color: var(--matrix-green);
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        .terminal-block:hover {
            border-color: var(--matrix-cyan);
            box-shadow: 
                0 0 20px rgba(0, 255, 255, 0.4),
                inset 0 0 10px rgba(0, 255, 65, 0.1);
            transform: translateX(5px);
        }
        
        /* Glitch Effect */
        .glitch {
            position: relative;
            animation: glitch 3s infinite;
        }
        
        @keyframes glitch {
            0%, 90%, 100% { transform: translate(0); }
            91% { transform: translate(-2px, 2px); }
            92% { transform: translate(2px, -2px); }
            93% { transform: translate(-2px, -2px); }
            94% { transform: translate(2px, 2px); }
        }
        
        .glitch:hover {
            animation: glitch-hover 0.3s infinite;
        }
        
        @keyframes glitch-hover {
            0%, 100% { 
                text-shadow: 
                    -2px 0 var(--matrix-red),
                    2px 0 var(--matrix-cyan);
            }
            50% { 
                text-shadow: 
                    2px 0 var(--matrix-red),
                    -2px 0 var(--matrix-cyan);
            }
        }
        
        /* Typing Animation */
        .typing {
            overflow: hidden;
            border-right: 3px solid var(--matrix-green);
            white-space: nowrap;
            animation: typing 3s steps(40) 1s forwards, blink-caret 0.75s step-end infinite;
            max-width: fit-content;
        }
        
        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }
        
        @keyframes blink-caret {
            from, to { border-color: transparent; }
            50% { border-color: var(--matrix-green); }
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #001a00 0%, #003300 100%) !important;
            color: var(--matrix-green) !important;
            border: 2px solid var(--matrix-green) !important;
            border-radius: 3px !important;
            padding: 0.5rem 2rem !important;
            font-weight: 600 !important;
            font-family: 'Courier Prime', monospace !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.3) !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .stButton>button:hover {
            background: var(--matrix-green) !important;
            color: #000 !important;
            box-shadow: 
                0 0 20px rgba(0, 255, 65, 0.6),
                inset 0 0 10px rgba(0, 0, 0, 0.5) !important;
            transform: translateY(-2px);
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0a0a 0%, #001a00 100%) !important;
            border-right: 2px solid var(--matrix-green);
            box-shadow: 0 0 20px rgba(0, 255, 65, 0.2);
        }
        
        [data-testid="stSidebar"] * {
            color: var(--matrix-green) !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--matrix-cyan) !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--matrix-green) !important;
            font-weight: 600 !important;
        }
        
        .metric-container {
            background: rgba(0, 20, 0, 0.5);
            padding: 1rem;
            border: 1px solid var(--matrix-green);
            border-radius: 3px;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
        }
        
        /* Data Tables */
        .dataframe {
            border: 1px solid var(--matrix-green) !important;
            border-radius: 3px;
            background: rgba(0, 20, 0, 0.3) !important;
        }
        
        .dataframe th {
            background: rgba(0, 50, 0, 0.8) !important;
            color: var(--matrix-green) !important;
            border: 1px solid var(--matrix-green) !important;
            font-weight: 700 !important;
        }
        
        .dataframe td {
            color: var(--matrix-cyan) !important;
            border: 1px solid rgba(0, 255, 65, 0.2) !important;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(0, 20, 0, 0.5);
            border: 2px dashed var(--matrix-green);
            border-radius: 3px;
            padding: 2rem;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: var(--matrix-cyan);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        }
        
        /* Success/Info/Warning Messages */
        .stSuccess {
            background: rgba(0, 50, 0, 0.5) !important;
            border-left: 4px solid var(--matrix-green) !important;
            color: var(--matrix-green) !important;
            border-radius: 3px;
        }
        
        .stInfo {
            background: rgba(0, 50, 50, 0.5) !important;
            border-left: 4px solid var(--matrix-cyan) !important;
            color: var(--matrix-cyan) !important;
            border-radius: 3px;
        }
        
        .stWarning {
            background: rgba(50, 0, 0, 0.5) !important;
            border-left: 4px solid var(--matrix-red) !important;
            color: var(--matrix-red) !important;
            border-radius: 3px;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(0, 20, 0, 0.5) !important;
            border: 1px solid var(--matrix-green) !important;
            color: var(--matrix-green) !important;
            border-radius: 3px;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(0, 30, 0, 0.7) !important;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
        }
        
        /* Text Input */
        .stTextInput input {
            background: rgba(0, 20, 0, 0.8) !important;
            border: 1px solid var(--matrix-green) !important;
            color: var(--matrix-green) !important;
            border-radius: 3px;
        }
        
        .stTextInput input:focus {
            border-color: var(--matrix-cyan) !important;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5) !important;
        }
        
        /* Select Box */
        .stSelectbox select {
            background: rgba(0, 20, 0, 0.8) !important;
            border: 1px solid var(--matrix-green) !important;
            color: var(--matrix-green) !important;
            border-radius: 3px;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background-color: var(--matrix-green) !important;
            box-shadow: 0 0 10px var(--glow-color);
        }
        
        /* ScrollBar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0a0a0a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--matrix-green);
            border-radius: 5px;
            box-shadow: 0 0 5px var(--glow-color);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--matrix-cyan);
        }
        
        /* Matrix Rain Background */
        .matrix-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.1;
            pointer-events: none;
        }
        
        /* Markdown Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--matrix-green) !important;
            text-shadow: 0 0 10px var(--glow-color);
            font-family: 'VT323', monospace !important;
            letter-spacing: 2px;
        }
        
        /* Paragraphs */
        p {
            color: var(--matrix-cyan) !important;
        }
        
        /* Links */
        a {
            color: var(--matrix-cyan) !important;
            text-decoration: none;
            border-bottom: 1px solid var(--matrix-cyan);
            transition: all 0.3s ease;
        }
        
        a:hover {
            color: var(--matrix-green) !important;
            border-bottom-color: var(--matrix-green);
            text-shadow: 0 0 5px var(--glow-color);
        }
        
        /* Code Blocks */
        code {
            background: rgba(0, 20, 0, 0.8) !important;
            color: var(--matrix-green) !important;
            border: 1px solid var(--matrix-green);
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier Prime', monospace !important;
        }
        
        /* Horizontal Rule */
        hr {
            border-color: var(--matrix-green) !important;
            opacity: 0.3;
        }
    </style>
    """

def get_typing_animation_js():
    """Returns JavaScript for typing animation effect"""
    return """
    <script>
        function typeWriter(element, text, speed = 50) {
            let i = 0;
            element.innerHTML = '';
            function type() {
                if (i < text.length) {
                    element.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(type, speed);
                }
            }
            type();
        }
        
        // Auto-apply to elements with typing class
        document.addEventListener('DOMContentLoaded', function() {
            const typingElements = document.querySelectorAll('.typing-text');
            typingElements.forEach(el => {
                const text = el.textContent;
                typeWriter(el, text);
            });
        });
    </script>
    """

def get_ascii_banner():
    """Returns ASCII art banner for terminal theme"""
    return """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗███╗   ██╗║
    ║   ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝████╗  ██║║
    ║   ██║  ██║███████║   ██║   ███████║███████╗█████╗  ██╔██╗ ██║║
    ║   ██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝  ██║╚██╗██║║
    ║   ██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗██║ ╚████║║
    ║   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝║
    ║                                                               ║
    ║                         SENSE AI                              ║
    ║                  [NEURAL ANALYSIS SYSTEM]                     ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

def get_hacker_emojis():
    """Returns dictionary of hacker-themed emojis"""
    return {
        'skull': '💀',
        'fire': '🔥',
        'lightning': '⚡',
        'alien': '👾',
        'target': '🎯',
        'lock': '🔒',
        'unlock': '🔓',
        'key': '🔑',
        'warning': '⚠️',
        'danger': '☠️',
        'computer': '💻',
        'terminal': '🖥️',
        'data': '📊',
        'chart': '📈',
        'brain': '🧠',
        'robot': '🤖',
        'gear': '⚙️',
        'wrench': '🔧',
        'shield': '🛡️',
        'satellite': '🛰️'
    }

def create_terminal_progress_bar(progress, width=50):
    """
    Creates an ASCII progress bar
    Args:
        progress: Float between 0 and 1
        width: Character width of the bar
    Returns:
        String representing the progress bar
    """
    filled = int(width * progress)
    empty = width - filled
    bar = '█' * filled + '░' * empty
    percentage = int(progress * 100)
    return f"[{bar}] {percentage}%"

def get_matrix_rain_html():
    """Returns HTML/JavaScript for matrix rain effect"""
    return """
    <canvas id="matrix-rain" style="position: fixed; top: 0; left: 0; z-index: -1; opacity: 0.1;"></canvas>
    <script>
        const canvas = document.getElementById('matrix-rain');
        const ctx = canvas.getContext('2d');
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
        const fontSize = 14;
        const columns = canvas.width / fontSize;
        const drops = Array(Math.floor(columns)).fill(1);
        
        function draw() {
            ctx.fillStyle = 'rgba(10, 10, 10, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#00ff41';
            ctx.font = fontSize + 'px monospace';
            
            for (let i = 0; i < drops.length; i++) {
                const text = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                
                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        }
        
        setInterval(draw, 33);
        
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
    """

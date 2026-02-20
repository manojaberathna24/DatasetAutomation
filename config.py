"""
Configuration settings for DataSense AI
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Gemini AI and OpenRouter Fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite-preview:free")

# Google Cloud Storage
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# File Settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20000"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "csv,xlsx,xls,json,zip,rar").split(",")

# Local Storage (fallback if GCS not configured)
LOCAL_STORAGE_DIR = "data_storage"
UPLOAD_DIR = os.path.join(LOCAL_STORAGE_DIR, "uploads")
CLEANED_DIR = os.path.join(LOCAL_STORAGE_DIR, "cleaned")
MODELS_DIR = os.path.join(LOCAL_STORAGE_DIR, "models")
REPORTS_DIR = os.path.join(LOCAL_STORAGE_DIR, "reports")

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, CLEANED_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Chart Color Schemes
COLOR_SCHEMES = {
    "vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"],
    "professional": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
    "dark": ["#BB86FC", "#03DAC6", "#CF6679", "#018786", "#3700B3"],
    "ocean": ["#006994", "#0892A5", "#1AC9E6", "#6DD3CE", "#A6E1FA"],
    "sunset": ["#FF9B85", "#FF6F69", "#FEC8C8", "#F4B6C2", "#FFD1DC"]
}

# ML Model Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Outlier Detection
OUTLIER_Z_THRESHOLD = 3
OUTLIER_IQR_MULTIPLIER = 1.5

# App Theme
THEME_CONFIG = {
    "primaryColor": "#FF6B6B",
    "backgroundColor": "#0E1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#FAFAFA",
    "font": "sans-serif"
}

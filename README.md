# 🧠 DataSense AI

**DataSense AI** is a comprehensive Streamlit-based data analysis platform that combines data visualization, automated cleaning, machine learning, and AI-powered insights into a single, intuitive web application.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 📤 Dataset Upload & Management
- Support for CSV, Excel (xlsx, xls), and JSON files
- File size limit: 200MB
- Google Cloud Storage integration with local fallback
- Dataset preview and metadata display
- Multi-dataset management

### 📊 Interactive Visualizations
- **8 Chart Types**: Bar, Pie, Line, Histogram, Scatter, Heatmap, Boxplot, Violin
- Dynamic axis selection and data aggregation
- Customizable color schemes (Vibrant, Professional, Dark, Ocean, Sunset)
- Interactive Plotly charts with zoom, pan, and export
- Chart download as PNG/HTML

### 🔍 Comprehensive Data Analysis
- Automatic column type detection (Numeric, Categorical, Text, DateTime)
- Missing value analysis with visualization
- Outlier detection using IQR and Z-score methods
- Summary statistics and correlation matrix
- Duplicate row detection
- Exportable analysis reports (JSON)

### 🧹 Intelligent Data Cleaning
- **Missing Values**: Drop, fill (mean/median/mode), forward/backward fill, custom value
- **Duplicates**: Identify and remove duplicates
- **Outliers**: Remove, cap (winsorize), or mark outliers
- Before/after preview
- Cleaning operation log
- Export cleaned datasets (CSV/Excel)

### 🤖 AutoML - Automated Machine Learning
- **Classification**: Logistic Regression, Random Forest, XGBoost, SVM
- **Regression**: Linear, Ridge, Random Forest, XGBoost
- **Clustering**: K-Means, DBSCAN, Hierarchical
- Automated model comparison and selection
- Performance metrics and visualizations:
  - Confusion matrices
  - ROC curves
  - Feature importance
  - Residual plots
  - Cluster visualizations
- Model export (.pkl format)

### 💬 Chat with Data (AI-Powered)
- Natural language data querying using Google Gemini AI
- Automatic code generation for pandas operations
- Chart generation from questions
- Chat history and suggested questions
- Fallback mode when API is not configured

### 🎯 AI Business Intelligence Agent
- Comprehensive automated analysis
- Key insights extraction
- Business recommendations
- Risk assessment
- Trend identification
- Next steps suggestions
- Exportable insights (JSON)

### 📄 Professional PDF Reports
- Customizable report sections
- Dataset overview and statistics
- Column information tables
- ML model results
- AI-generated insights
- Professional formatting with custom styling

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
cd Desktop/data
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env  # Windows
   ```

2. Edit `.env` and add your credentials:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GCS_BUCKET_NAME=your_bucket_name_here
   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
   ```

**Note:** The application will work without these configurations but with limited features:
- Without Gemini API: Chat and AI Agent will use fallback mode
- Without GCS: Files will be stored locally

## 📖 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Quick Start Guide

1. **Upload Dataset**
   - Navigate to "📤 Upload Dataset"
   - Select your CSV/Excel/JSON file
   - Review the preview and save the dataset

2. **Explore & Visualize**
   - Go to "📊 Visualize" to create charts
   - Choose chart type and configure axes
   - Customize colors and download charts

3. **Analyze Data**
   - Visit "🔍 Analyze" for automated analysis
   - Review column types, missing values, outliers
   - Generate correlation matrices

4. **Clean Data**
   - Open "🧹 Clean Data" to prepare your data
   - Handle missing values, duplicates, outliers
   - Preview and export cleaned dataset

5. **Build ML Models**
   - Navigate to "🤖 AutoML"
   - Select task (Classification/Regression/Clustering)
   - Choose target and features
   - Train and compare multiple models
   - Download the best model

6. **Chat with Data**
   - Go to "💬 Chat with Data"
   - Ask questions in natural language
   - Get AI-powered answers and charts

7. **Get AI Insights**
   - Visit "🎯 AI Agent"
   - Click "Analyze Dataset"
   - Review insights, recommendations, and risks

8. **Generate Reports**
   - Open "📄 Generate Report"
   - Select sections to include
   - Generate and download PDF

## 🔧 Configuration

### File Settings (`config.py`)
- `MAX_FILE_SIZE_MB`: Maximum upload size (default: 200MB)
- `ALLOWED_EXTENSIONS`: Supported file types
- `COLOR_SCHEMES`: Chart color palettes

### ML Settings
- `RANDOM_STATE`: Random seed for reproducibility (42)
- `TEST_SIZE`: Train/test split ratio (0.2)
- `CV_FOLDS`: Cross-validation folds (5)
- `OUTLIER_Z_THRESHOLD`: Z-score threshold (3)
- `OUTLIER_IQR_MULTIPLIER`: IQR multiplier (1.5)

## 📁 Project Structure
```
data/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── pages/                     # Streamlit pages
│   ├── 1_📤_Upload_Dataset.py
│   ├── 2_📊_Visualize.py
│   ├── 3_🔍_Analyze.py
│   ├── 4_🧹_Clean_Data.py
│   ├── 5_🤖_AutoML.py
│   ├── 6_💬_Chat_with_Data.py
│   ├── 7_🎯_AI_Agent.py
│   └── 8_📄_Generate_Report.py
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading & GCS integration
│   ├── analysis.py           # Data analysis functions
│   ├── visualization.py      # Chart generation
│   ├── cleaning.py           # Data cleaning utilities
│   ├── ml_models.py          # ML pipeline
│   ├── ai_chat.py            # Gemini AI integration
│   └── pdf_generator.py      # PDF report creation
└── data_storage/             # Local storage (auto-created)
    ├── uploads/
    ├── cleaned/
    ├── models/
    └── reports/
```

## 🔑 Getting API Keys

### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### Google Cloud Storage (Optional)
1. Create a [Google Cloud Project](https://console.cloud.google.com/)
2. Enable Cloud Storage API
3. Create a storage bucket
4. Create a service account and download the JSON key
5. Update `.env` with bucket name and credentials path

## 💡 Tips & Best Practices

1. **Data Quality**: Clean your data before running ML models for better results
2. **Feature Selection**: Use domain knowledge to select relevant features
3. **API Limits**: Gemini API has usage quotas - use wisely
4. **Large Datasets**: For files >100MB, consider using only necessary columns
5. **Model Comparison**: Always compare multiple models in AutoML
6. **Insights**: Combine AI insights with your domain expertise

## 🐛 Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Streamlit Not Found
```bash
pip install streamlit
```

### Google Cloud Authentication Error
- Verify your service account JSON file path
- Check file permissions
- Ensure GCS API is enabled in your project

### Gemini API Errors
- Verify API key in `.env` file
- Check API quota limits
- Ensure internet connection

### PDF Generation Issues
```bash
pip install --upgrade reportlab fpdf2
```

## 📊 Supported Data Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | .csv | UTF-8 encoding recommended |
| Excel | .xlsx, .xls | Multiple sheets not supported |
| JSON | .json | Must be in records format |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created with ❤️ using Streamlit and Google Gemini AI

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **Google Gemini** - For AI-powered insights
- **Plotly** - For interactive visualizations
- **scikit-learn** - For machine learning capabilities
- **XGBoost** - For gradient boosting models

## 📞 Support

For issues, questions, or suggestions:
- Check the [Troubleshooting](#-troubleshooting) section
- Review the code documentation
- Check Streamlit documentation at [docs.streamlit.io](https://docs.streamlit.io)

---

**DataSense AI** - Transform your data into actionable insights! 🚀

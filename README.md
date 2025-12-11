# DataSense AI - Neural Analysis Terminal

A powerful Streamlit-based data analysis and machine learning platform with an advanced cyberpunk/hacker-themed interface. DataSense AI transforms raw data into actionable insights using cutting-edge analytics, machine learning algorithms, and artificial intelligence powered by Google Gemini AI.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Contributing](#contributing)
- [License](#license)

## Introduction

DataSense AI is an advanced cyber intelligence platform designed for data analysts, data scientists, and business professionals who need to:

- Analyze large datasets quickly and efficiently
- Generate insightful visualizations without writing code
- Build machine learning models with minimal effort
- Extract business insights using AI-powered chat interface
- Generate comprehensive PDF reports for stakeholders
- Clean and prepare data for analysis

The application features a unique cyberpunk/hacker-themed interface inspired by Kali Linux terminal aesthetics, complete with matrix-style animations and terminal-like interactions.

## Technologies Used

### Core Framework
- **Streamlit 1.29.0** - Main web application framework for building interactive dashboards

### Data Processing & Analysis
- **Pandas 2.1.4** - Data manipulation and analysis
- **NumPy 1.26.2** - Numerical computing and array operations
- **SciPy 1.11.4** - Scientific computing and statistical functions
- **OpenPyXL 3.1.2** - Excel file (.xlsx) reading and writing
- **xlrd 2.0.1** - Excel file (.xls) support

### Data Visualization
- **Plotly 5.18.0** - Interactive charts and dashboards
- **Matplotlib 3.8.2** - Static plotting library
- **Seaborn 0.13.0** - Statistical data visualization

### Machine Learning
- **Scikit-learn 1.3.2** - Classic ML algorithms (classification, regression, clustering)
- **XGBoost 2.0.3** - Gradient boosting framework
- **LightGBM 4.1.0** - Gradient boosting machine learning framework
- **Imbalanced-learn 0.11.0** - Handling imbalanced datasets

### Time Series & Forecasting
- **Prophet 1.1.5** - Time series forecasting

### AI & Natural Language Processing
- **Google Generative AI 0.3.2** - Gemini AI integration for intelligent chat and insights

### Cloud Storage
- **Google Cloud Storage 2.14.0** - Cloud storage integration (optional)

### Report Generation
- **ReportLab 4.0.7** - PDF generation library
- **FPDF2 2.7.6** - Alternative PDF generation

### Utilities
- **Python-dotenv 1.0.0** - Environment variable management
- **Pillow 10.1.0** - Image processing
- **Joblib 1.3.2** - Model persistence and pipeline caching

## Features

### 1. Data Upload & Management
- Support for multiple file formats: CSV, Excel (XLSX/XLS), JSON
- Maximum file size: 200MB
- Automatic data type detection
- Session-based dataset management

### 2. Visual Analytics
- Interactive chart types:
  - Bar Charts
  - Pie Charts
  - Line Charts
  - Scatter Plots
  - Heatmaps
  - Box Plots
  - Correlation Matrices

### 3. Deep Analysis
- Automated data profiling
- Statistical summaries
- Outlier detection
- Correlation analysis
- Missing value analysis

### 4. Data Cleaning & Preparation
- Handle missing values (drop, fill, interpolate)
- Remove duplicate records
- Outlier detection and removal
- Data type conversion
- Column renaming and selection

### 5. AutoML Engine
- Automated machine learning model training
- Supported problem types:
  - Classification (Binary & Multi-class)
  - Regression
  - Clustering
- Multiple algorithms:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - SVM
  - K-Means
  - And more...
- Automatic model evaluation and metrics

### 6. AI Chat Interface
- Natural language queries about your data
- Powered by Google Gemini AI
- Context-aware responses
- Data insights and recommendations

### 7. Autonomous AI Agent
- Automated business insights generation
- Pattern recognition
- Trend analysis
- Actionable recommendations

### 8. Professional Report Generation
- Comprehensive PDF reports
- Includes:
  - Executive summary
  - Data statistics
  - Visualizations
  - Missing value analysis
  - Correlation insights
  - Key findings

## Prerequisites

Before installing DataSense AI, ensure you have:

- **Python 3.8 or higher** installed on your system
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Google Gemini API Key** (required for AI features)
- **Google Cloud Service Account** (optional, for cloud storage features)

## Installation

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/yourusername/datasense-ai.git
cd datasense-ai
```

If you don't have a Git repository URL, you can download the project as a ZIP file and extract it.

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment helps isolate project dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install all the technologies listed in the [Technologies Used](#technologies-used) section.

### Step 4: Set Up Environment Variables

See the [Configuration](#configuration) section below for detailed instructions on setting up your credentials.

## Configuration

### Setting Up Your .env File

The application uses environment variables for configuration. Follow these steps:

#### Step 1: Copy the Example File

Copy the `.env.example` file to create your own `.env` file:

```bash
# On Windows
copy .env.example .env

# On macOS/Linux
cp .env.example .env
```

#### Step 2: Obtain Your Google Gemini API Key

The Gemini API key is **required** for AI-powered features (Chat, AI Agent, Insights).

1. **Visit Google AI Studio:**
   - Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
   - Or visit [https://aistudio.google.com/](https://aistudio.google.com/)

2. **Sign in with your Google Account:**
   - Use your existing Google account or create a new one

3. **Create an API Key:**
   - Click on "Get API Key" or "Create API Key"
   - Click "Create API Key in new project" (or select an existing project)
   - Copy the generated API key

4. **Add to .env file:**
   - Open the `.env` file in a text editor
   - Replace `your_gemini_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567
   ```

**Important Notes:**
- Keep your API key secret and never commit it to version control
- The Gemini API has a free tier with generous quotas
- Some features may be limited without a valid API key

#### Step 3: Configure Google Cloud Storage (Optional)

Google Cloud Storage is **optional** and only needed if you want to store datasets in the cloud.

**If you don't need cloud storage, you can skip this step.**

To enable cloud storage:

1. **Create a Google Cloud Project:**
   - Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Google Cloud Storage API:**
   - In the Cloud Console, go to "APIs & Services" > "Library"
   - Search for "Cloud Storage API"
   - Click "Enable"

3. **Create a Storage Bucket:**
   - Go to "Cloud Storage" > "Buckets"
   - Click "Create Bucket"
   - Choose a unique name for your bucket
   - Select location and storage class
   - Click "Create"

4. **Create a Service Account:**
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name and description
   - Grant it "Storage Object Admin" role
   - Click "Done"

5. **Generate Service Account Key:**
   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the JSON key file
   - Save it securely in your project directory

6. **Update .env file:**
   ```
   GCS_BUCKET_NAME=your-bucket-name
   GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
   ```

#### Step 4: Configure Application Settings (Optional)

You can customize these settings in your `.env` file:

```
# Maximum file upload size in MB (default: 200)
MAX_FILE_SIZE_MB=200

# Allowed file extensions (default: csv,xlsx,xls,json)
ALLOWED_EXTENSIONS=csv,xlsx,xls,json
```

### Final .env File Example

Your completed `.env` file should look like this:

```
# DataSense AI Configuration

# Google Gemini AI API Key (REQUIRED)
GEMINI_API_KEY=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567

# Google Cloud Storage Configuration (OPTIONAL)
GCS_BUCKET_NAME=my-datasense-bucket
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account-key.json

# Application Settings (OPTIONAL)
MAX_FILE_SIZE_MB=200
ALLOWED_EXTENSIONS=csv,xlsx,xls,json
```

## Running the Application

### Start the Application

After completing installation and configuration, run the application:

```bash
streamlit run app.py
```

### Access the Application

Once the application starts, you'll see output like:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.100:8501
```

Open your web browser and navigate to:
```
http://localhost:8501
```

### Stopping the Application

To stop the application:
- Press `Ctrl + C` in the terminal where the application is running

### Troubleshooting

**Issue: Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Issue: Module not found errors**
```bash
pip install -r requirements.txt --upgrade
```

**Issue: Gemini API errors**
- Verify your API key is correct in the `.env` file
- Check your internet connection
- Ensure you haven't exceeded API quota limits

## Project Structure

```
datasense-ai/
│
├── app.py                          # Main application entry point
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .env                            # Your actual environment variables (not in git)
│
├── pages/                          # Streamlit pages (multi-page app)
│   ├── 1_Upload_Dataset.py        # Data upload functionality
│   ├── 2_Visualize.py              # Data visualization charts
│   ├── 3_Analyze.py                # Statistical analysis
│   ├── 4_Clean_Data.py             # Data cleaning utilities
│   ├── 5_AutoML.py                 # Machine learning model training
│   ├── 6_Chat_with_Data.py         # AI chat interface
│   ├── 7_AI_Agent.py               # Autonomous AI insights
│   └── 8_Generate_Report.py        # PDF report generation
│
├── utils/                          # Utility modules
│   ├── terminal_theme.py           # Cyberpunk theme styling
│   ├── data_loader.py              # Data loading functions
│   ├── visualization.py            # Chart generation
│   ├── ml_models.py                # ML model implementations
│   ├── ai_chat.py                  # Gemini AI integration
│   └── report_generator.py         # PDF report creation
│
├── data_storage/                   # Local data storage (created at runtime)
│   ├── datasets/                   # Uploaded datasets
│   ├── models/                     # Trained ML models
│   └── reports/                    # Generated reports
│
└── README.md                       # This file
```

## Usage Guide

### 1. Upload Your Dataset
- Navigate to "Upload Dataset" in the sidebar
- Upload a CSV, Excel, or JSON file
- Review the data preview and basic statistics

### 2. Visualize Your Data
- Go to "Visualize" page
- Select chart type (bar, line, pie, scatter, etc.)
- Choose columns for X and Y axes
- Customize colors and labels
- Download or share visualizations

### 3. Analyze Your Data
- Navigate to "Analyze" page
- View statistical summaries
- Check correlation matrices
- Identify outliers and patterns
- Review missing value analysis

### 4. Clean Your Data
- Go to "Clean Data" page
- Handle missing values (drop, fill, interpolate)
- Remove duplicates
- Filter outliers
- Save cleaned dataset

### 5. Train ML Models
- Navigate to "AutoML" page
- Select problem type (classification/regression/clustering)
- Choose target variable
- Select algorithm
- Train model and view results
- Download trained model

### 6. Chat with Your Data
- Go to "Chat with Data" page
- Ask questions in natural language
- Get AI-powered insights and answers
- View conversation history

### 7. Get AI Insights
- Navigate to "AI Agent" page
- Let AI analyze your data automatically
- Review generated insights and recommendations
- Export findings

### 8. Generate Reports
- Go to "Generate Report" page
- Customize report sections
- Generate professional PDF report
- Download and share with stakeholders

## Deploying to Streamlit Cloud

### Prerequisites

Before deploying to Streamlit Cloud, ensure you have:
- A GitHub account
- Your code pushed to a GitHub repository
- Google Gemini API key ready

### Step 1: Prepare Your Repository

1. **Push your code to GitHub:**
   ```bash
   cd c:\Users\Manoj Aberathna\Desktop\data
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/datasense-ai.git
   git push -u origin main
   ```

2. **Ensure these files are in your repository:**
   - `requirements.txt` - Python dependencies
   - `.python-version` - Specifies Python 3.11.9
   - `app.py` - Main application file
   - `.env.example` - Template for environment variables

3. **Important: Do NOT commit `.env` file**
   - Add `.env` to your `.gitignore` file
   - This file contains sensitive API keys

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in with your GitHub account

2. **Create New App:**
   - Click "New app"
   - Select your repository: `yourusername/datasense-ai`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets (Environment Variables):**
   - Before the app fully deploys, click on "Advanced settings"
   - Or go to your app settings after deployment
   - Click "Secrets"
   - Add your environment variables in TOML format:
   
   ```toml
   # Secrets for DataSense AI
   GEMINI_API_KEY = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
   
   # Optional: Google Cloud Storage
   GCS_BUCKET_NAME = "your-bucket-name"
   GOOGLE_APPLICATION_CREDENTIALS = "path/to/service-account-key.json"
   
   # Optional: Application Settings
   MAX_FILE_SIZE_MB = "200"
   ALLOWED_EXTENSIONS = "csv,xlsx,xls,json"
   ```

4. **Save and Deploy:**
   - Click "Save"
   - Streamlit Cloud will automatically deploy your app
   - Wait 2-5 minutes for the build to complete

### Step 3: Access Your Deployed App

Once deployment is successful, you'll receive a URL like:
```
https://yourusername-datasense-ai-app-xyz123.streamlit.app/
```

Share this URL with anyone who needs access to your DataSense AI application!

### Updating Your Deployed App

Any changes you push to your GitHub repository will automatically trigger a redeploy:

```bash
git add .
git commit -m "Update feature"
git push
```

Streamlit Cloud will detect the changes and redeploy automatically.

### Troubleshooting Deployment Issues

**Issue: Dependencies fail to install**
- **Solution:** The `.python-version` file ensures Python 3.11.9 is used. If you still have issues, try updating package versions in `requirements.txt`

**Issue: Missing environment variables**
- **Solution:** Double-check your Secrets in Streamlit Cloud settings match the required variable names

**Issue: App crashes on startup**
- **Solution:** Check the logs in Streamlit Cloud for error messages. Common issues:
  - Missing or invalid `GEMINI_API_KEY`
  - Import errors (verify all dependencies are in `requirements.txt`)
  - File path issues (use relative paths, not absolute Windows paths)

**Issue: File upload not working**
- **Solution:** Streamlit Cloud has memory and storage limits. Keep uploaded files under the `MAX_FILE_SIZE_MB` limit

**Issue: Slow performance**
- **Solution:** Streamlit Cloud free tier has resource limits. Consider:
  - Caching frequently used data with `@st.cache_data`
  - Optimizing large datasets
  - Upgrading to a paid Streamlit Cloud plan

### Managing Multiple Environments

**Development (Local):**
- Use `.env` file for local development
- Test changes before pushing to GitHub

**Production (Streamlit Cloud):**
- Use Streamlit Secrets for environment variables
- Monitor app logs for issues
- Set up branch-based deployments (e.g., `main` for production, `dev` for testing)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**DataSense AI v2.0 - Terminal Edition**

Powered by Streamlit & Gemini AI | Developed by Manoj Aberathna | 2024

For issues, questions, or feature requests, please open an issue on GitHub.

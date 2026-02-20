
Try using streamlit :-  https://datasence.streamlit.app/

Demo Video :-  https://drive.google.com/file/d/18jKZhAWmm4ZazScZtk1EhxNgsXEndZPk/view?usp=drive_link



# DataSense AI - Neural Analysis Terminal

A powerful Streamlit-based data analysis and machine learning platform with an advanced cyberpunk/hacker-themed interface. DataSense AI transforms raw data into actionable insights using cutting-edge analytics, machine learning algorithms, and artificial intelligence powered by Google Gemini AI.


![Alt text](img.jpg.JPG)



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
git clone https://github.com/manojaberathna24/DatasetAutomation.git
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


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**DataSense AI v2.0 - Terminal Edition**

Powered by Streamlit & Gemini AI | Developed by Manoj Aberathna | 2025

For issues, questions, or feature requests, please open an issue on GitHub.

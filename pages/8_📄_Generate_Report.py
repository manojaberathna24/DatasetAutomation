"""
Report Generation Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_generator import PDFReportGenerator, create_analysis_report
from utils.analysis import DataAnalyzer
from utils.visualization import DataVisualizer
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
from config import REPORTS_DIR
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Report Generator", page_icon="📊", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown("""
<div class="terminal-header">
    <h1>REPORT GENERATION WORKSPACE</h1>
    <p>Professional PDF Report Compilation System</p>
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
        <p>Upload a dataset first to generate reports</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Auto-detect and use cleaned dataset if available
base_name = st.session_state.current_dataset
cleaned_name = f"{base_name}_cleaned"

# Check if cleaned version exists
if 'cleaned_datasets' in st.session_state and cleaned_name in st.session_state.cleaned_datasets:
    df = st.session_state.cleaned_datasets[cleaned_name]
    using_cleaned = True
    dataset_display_name = cleaned_name
    st.success(f"Using cleaned dataset: {cleaned_name}")
    st.info("This dataset has been cleaned - duplicates removed and missing values filled")
else:
    df = st.session_state.datasets[base_name]
    using_cleaned = False
    dataset_display_name = base_name

analyzer = DataAnalyzer(df)
viz = DataVisualizer(df)

# Sidebar
with st.sidebar:
    st.markdown("### Report Sections")
    
    include_sections = {
        'overview': st.checkbox("Dataset Overview", value=True),
        'columns': st.checkbox("Column Information", value=True),
        'missing': st.checkbox("Missing Values Analysis", value=True),
        'statistics': st.checkbox("Summary Statistics", value=True),
        'correlation': st.checkbox("Correlation Analysis", value=False),
        'visualizations': st.checkbox("Visualizations", value=False),
        'ml_results': st.checkbox("ML Model Results", value=False),
        'ai_insights': st.checkbox("AI Insights", value=False)
    }
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")

# Main content
st.markdown("### Report Configuration")

col1, col2 = st.columns(2)

with col1:
    report_title = st.text_input(
        "Report Title",
        value=f"Data Analysis Report - {st.session_state.current_dataset}"
    )

with col2:
    report_author = st.text_input(
        "Author Name",
        value="DataSense AI"
    )

# Auto-cleaning section
st.markdown("---")
st.markdown("### Data Cleaning & Quality")

auto_clean = st.checkbox("Auto-Clean Data Before Report", value=False, 
                         help="Automatically remove duplicates and fill missing values")

if auto_clean:
    st.info("Auto-Cleaning will:")
    st.write("• Remove duplicate rows")
    st.write("• Drop columns that are 100% missing")
    st.write("• Fill missing numeric values with median")
    st.write("• Fill missing text values with mode")
    
    if st.button("Clean Data Now", type="primary"):
        from utils.cleaning import DataCleaner
        
        with st.spinner("Cleaning data..."):
            cleaner = DataCleaner(df)
            
            # Remove duplicates
            before_rows = len(df)
            cleaner.remove_duplicates(keep='first')
            
            # Drop completely empty columns first
            cleaner.drop_empty_columns()
            
            # Fill remaining missing values
            # Need to get updated columns after dropping empty ones
            current_df = cleaner.get_cleaned_df()
            numeric_cols = current_df.select_dtypes(include=['number']).columns.tolist()
            text_cols = current_df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                cleaner.handle_missing_values(strategy='fill_median', columns=numeric_cols)
            if text_cols:
                cleaner.handle_missing_values(strategy='fill_mode', columns=text_cols)
            
            # Get cleaned data
            cleaned_df = cleaner.get_cleaned_df()
            summary = cleaner.get_cleaning_summary()
            
            # Store in session
            st.session_state.cleaned_report_data = cleaned_df
            st.session_state.clean_summary = summary
            
            # Show results
            st.success(f"Data cleaned successfully!")
            
            # Before/After Comparison
            st.markdown("### Before vs After Cleaning")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", len(cleaned_df), 
                       delta=f"{len(cleaned_df) - before_rows}")
            col2.metric("Missing Values", cleaned_df.isnull().sum().sum(),
                       delta=f"{cleaned_df.isnull().sum().sum() - df.isnull().sum().sum()}")
            col3.metric("Duplicates", len(cleaned_df[cleaned_df.duplicated()]),
                       delta=f"{len(cleaned_df[cleaned_df.duplicated()]) - len(df[df.duplicated()])}")
            col4.metric("Data Quality", "Clean", delta="Improved")
            
            # Side by side preview
            st.markdown("#### Data Preview Comparison")
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.markdown("**Original Data**")
                st.dataframe(df.head(10), use_container_width=True, height=300)
            
            with col_after:
                st.markdown("**Cleaned Data**")
                st.dataframe(cleaned_df.head(10), use_container_width=True, height=300)
            
            # Download cleaned data
            st.markdown("### Download Cleaned Dataset")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = cleaned_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"{st.session_state.current_dataset}_cleaned.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    cleaned_df.to_excel(writer, index=False)
                st.download_button(
                    "Download Excel",
                    buffer.getvalue(),
                    f"{st.session_state.current_dataset}_cleaned.xlsx",
                    use_container_width=True
                )
            
            with col3:
                json_str = cleaned_df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"{st.session_state.current_dataset}_cleaned.json",
                    "application/json",
                    use_container_width=True
                )

# Use cleaned data if available
if 'cleaned_report_data' in st.session_state and auto_clean:
    df_to_analyze = st.session_state.cleaned_report_data
    st.info("📊 Using cleaned dataset for report generation")
    analyzer = DataAnalyzer(df_to_analyze)
    viz = DataVisualizer(df_to_analyze)
else:
    df_to_analyze = df

# Preview sections
st.markdown("---")
st.markdown("### Report Preview")

if include_sections['overview']:
    with st.expander("Dataset Overview", expanded=True):
        # Show data quality status
        if using_cleaned:
            st.markdown("**Data Status:** CLEANED (No duplicates, Missing values handled)")
        else:
            st.markdown("**Data Status:** ORIGINAL")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        col4.metric("Missing Values", df.isnull().sum().sum())

if include_sections['columns']:
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

if include_sections['missing']:
    with st.expander("Missing Values Analysis"):
        missing_df = analyzer.analyze_missing_values()
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found!")

if include_sections['statistics']:
    with st.expander("Summary Statistics"):
        summary = analyzer.get_summary_statistics()
        if not summary.empty:
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No numeric columns for statistics")

if include_sections['correlation']:
    with st.expander("Correlation Analysis"):
        corr_matrix = analyzer.get_correlation_matrix()
        if not corr_matrix.empty:
            fig = viz.create_heatmap()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation")

if include_sections['visualizations']:
    with st.expander("Visualizations"):
        st.info("Visualizations will be included in the PDF report")
        st.write("Charts that will be generated:")
        st.write("- Distribution histograms for numeric columns")
        st.write("- Correlation heatmap")
        st.write("- Missing values bar chart")

if include_sections['ml_results']:
    with st.expander("ML Model Results"):
        if 'ml_results' in st.session_state:
            results = st.session_state.ml_results
            st.write(f"**Task Type:** {st.session_state.ml_task}")
            st.write(f"**Models Trained:** {len(results)}")
            
            # Show best model
            if st.session_state.ml_task == "Classification":
                best = max(results.items(), key=lambda x: x[1]['accuracy'])
                st.write(f"**Best Model:** {best[0]} (Accuracy: {best[1]['accuracy']:.4f})")
            elif st.session_state.ml_task == "Regression":
                best = max(results.items(), key=lambda x: x[1]['r2'])
                st.write(f"**Best Model:** {best[0]} (R²: {best[1]['r2']:.4f})")
        else:
            st.info("No ML models trained yet. Train models in the AutoML page first.")

if include_sections['ai_insights']:
    with st.expander("AI Insights"):
        if 'ai_insights' in st.session_state:
            insights = st.session_state.ai_insights
            
            st.markdown("**Key Insights:**")
            for i, insight in enumerate(insights.get('key_insights', []), 1):
                st.write(f"{i}. {insight}")
            
            st.markdown("**Recommendations:**")
            for i, rec in enumerate(insights.get('recommendations', []), 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No AI insights generated yet. Use the AI Agent page first.")

# Generate Report
st.markdown("---")
st.markdown("### Generate Report")

if st.button("Generate PDF Report", type="primary", use_container_width=True):
    with st.spinner("Generating professional PDF report... This may take a moment..."):
        try:
            # Create report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_suffix = "_cleaned" if using_cleaned else "_original"
            filename = f"{base_name}{status_suffix}_report_{timestamp}.pdf"
            filepath = os.path.join(REPORTS_DIR, filename)
            
            # Initialize PDF generator
            pdf = PDFReportGenerator(filepath)
            
            # Title page
            title_text = report_title
            if using_cleaned:
                title_text += " (Cleaned Dataset)"
            
            pdf.add_title_page(
                title_text,
                f"Generated by {report_author}"
            )
            
            # Dataset Overview
            if include_sections['overview']:
                pdf.add_section("Dataset Overview")
                
                # Add data status
                if using_cleaned:
                    pdf.add_section("", "Data Status: CLEANED - Duplicates removed, Missing values handled")
                else:
                    pdf.add_section("", "Data Status: ORIGINAL - No cleaning applied")
                
                pdf.add_key_metrics({
                    "Dataset Name": dataset_display_name,
                    "Total Rows": f"{len(df):,}",
                    "Total Columns": len(df.columns),
                    "Memory Usage (MB)": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f}",
                    "Missing Values": df.isnull().sum().sum(),
                    "Duplicate Rows": len(df[df.duplicated()])
                })
            
            # Column Information
            if include_sections['columns']:
                pdf.add_section("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Unique': [df[col].nunique() for col in df.columns]
                })
                pdf.add_dataframe(col_info.head(20))
            
            # Missing Values
            if include_sections['missing']:
                pdf.add_section("Missing Values Analysis")
                missing_df = analyzer.analyze_missing_values()
                if len(missing_df) > 0:
                    pdf.add_dataframe(missing_df)
                else:
                    pdf.add_section("", "No missing values detected in the dataset.")
            
            # Summary Statistics
            if include_sections['statistics']:
                pdf.add_section("Summary Statistics")
                summary = analyzer.get_summary_statistics()
                if not summary.empty:
                    pdf.add_dataframe(summary)
            
            # ML Results
            if include_sections['ml_results'] and 'ml_results' in st.session_state:
                pdf.add_section("Machine Learning Results")
                results = st.session_state.ml_results
                
                results_df = pd.DataFrame([
                    {'Model': name, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
                    for name, metrics in results.items()
                ])
                pdf.add_dataframe(results_df)
            
            # AI Insights
            if include_sections['ai_insights'] and 'ai_insights' in st.session_state:
                insights = st.session_state.ai_insights
                
                pdf.add_section("AI-Generated Insights")
                if insights.get('key_insights'):
                    pdf.add_section("Key Insights", insights['key_insights'])
                
                if insights.get('recommendations'):
                    pdf.add_section("Recommendations", insights['recommendations'])
            
            # Build PDF
            if pdf.build():
                st.success("Report generated successfully!")
                
                # Download button
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f.read(),
                        file_name=filename,
                        mime="application/pdf",
                        type="primary"
                    )
            else:
                st.error("Failed to generate PDF report")
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

st.markdown("---")
st.markdown("""
### Report Features

Your PDF report will include:
- **Professional Formatting**: Clean, readable layout with custom styling
- **Comprehensive Tables**: All data tables with proper formatting
- **Key Metrics**: Important statistics and findings
- **Section Organization**: Clear structure for easy navigation
- **Timestamp**: Generation date and time for record-keeping

**Note:** Charts and visualizations require additional plotting libraries. The current version includes tables and text-based analysis.
""")

"""
Data Analysis Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analysis import DataAnalyzer
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="Deep Analysis", page_icon="👾", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown("""
<div class="terminal-header">
    <h1>DEEP ANALYSIS WORKSPACE</h1>
    <p>Statistical Analysis & Pattern Detection System</p>
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
        <p>Upload a dataset first to initialize analysis terminal</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Use cleaned dataset if available, otherwise use original
dataset_name = st.session_state.current_dataset
if 'cleaned_datasets' in st.session_state:
    cleaned_name = f"{dataset_name}_cleaned"
    if cleaned_name in st.session_state.cleaned_datasets:
        df = st.session_state.cleaned_datasets[cleaned_name]
        st.success(f"✅ Analyzing cleaned dataset: **{cleaned_name}**")
    else:
        df = st.session_state.datasets[dataset_name]
else:
    df = st.session_state.datasets[dataset_name]

# Initialize analyzer
analyzer = DataAnalyzer(df)
st.session_state.current_figures = []

# Sidebar
with st.sidebar:
    st.markdown("### ANALYSIS CONFIGURATION")
    
    analysis_options = st.multiselect(
        "Select Analysis Modules",
        ["Basic Info", "Column Types", "Missing Values", "Summary Statistics", 
         "Outliers", "Duplicates", "Correlation"],
        default=["Basic Info", "Missing Values"]
    )
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Records:** {len(df):,}")
    st.markdown(f"**Fields:** {len(df.columns)}")

# Basic Information
if "Basic Info" in analysis_options:
    st.markdown("### 📋 Basic Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    col4.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    
    st.markdown("---")

# Column Types
if "Column Types" in analysis_options:
    st.markdown("### 📊 Column Type Analysis")
    
    column_types = analyzer.get_column_types()
    
    # Create summary DataFrame
    type_df = pd.DataFrame([
        {
            'Column': col,
            'Type': info['type'],
            'Data Type': info['dtype'],
            'Unique Values': info['unique'],
            'Missing Count': info['null_count'],
            'Missing %': f"{info['null_percentage']:.1f}%"
        }
        for col, info in column_types.items()
    ])
    
    st.dataframe(type_df, use_container_width=True)
    
    # Type distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        type_counts = type_df['Type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="Column Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
        if 'current_figures' in st.session_state:
            st.session_state.current_figures.append(fig)
    
    with col2:
        dtype_counts = type_df['Data Type'].value_counts()
        fig = px.bar(x=dtype_counts.index, y=dtype_counts.values, 
                     title="Data Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
        if 'current_figures' in st.session_state:
            st.session_state.current_figures.append(fig)
    
    st.markdown("---")

# Missing Values
if "Missing Values" in analysis_options:
    st.markdown("### ⚠️ Missing Values Analysis")
    
    missing_df = analyzer.analyze_missing_values()
    
    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(missing_df, x='Column', y='Missing %', 
                     title="Missing Values by Column",
                     color='Missing %',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        if 'current_figures' in st.session_state:
            st.session_state.current_figures.append(fig)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    st.markdown("---")

# Summary Statistics
if "Summary Statistics" in analysis_options:
    st.markdown("### 📈 Summary Statistics")
    
    summary = analyzer.get_summary_statistics()
    
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("View Distribution", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                if 'current_figures' in st.session_state:
                    st.session_state.current_figures.append(fig)
            
            with col2:
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                if 'current_figures' in st.session_state:
                    st.session_state.current_figures.append(fig)
    else:
        st.info("No numeric columns found for summary statistics.")
    
    st.markdown("---")

# Outliers
if "Outliers" in analysis_options:
    st.markdown("### 🎯 Outlier Detection")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select Column for Outlier Detection", numeric_cols, key="outlier_col")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**IQR Method**")
            outlier_indices, iqr_info = analyzer.detect_outliers_iqr(selected_col)
            
            st.write(f"Lower Bound: {iqr_info.get('lower_bound', 0):.2f}")
            st.write(f"Upper Bound: {iqr_info.get('upper_bound', 0):.2f}")
            st.write(f"Outliers: {iqr_info.get('outlier_count', 0)} ({iqr_info.get('outlier_percentage', 0):.1f}%)")
        
        with col2:
            st.markdown("**Z-Score Method**")
            outlier_indices_z, z_info = analyzer.detect_outliers_zscore(selected_col)
            
            st.write(f"Mean: {z_info.get('mean', 0):.2f}")
            st.write(f"Std Dev: {z_info.get('std', 0):.2f}")
            st.write(f"Outliers: {z_info.get('outlier_count', 0)} ({z_info.get('outlier_percentage', 0):.1f}%)")
        
        # Box plot with outliers highlighted
        fig = px.box(df, y=selected_col, title=f"Outliers in {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
        if 'current_figures' in st.session_state:
            st.session_state.current_figures.append(fig)
    else:
        st.info("No numeric columns available for outlier detection.")
    
    st.markdown("---")

# Duplicates
if "Duplicates" in analysis_options:
    st.markdown("### 🔄 Duplicate Analysis")
    
    duplicates, dup_info = analyzer.detect_duplicates()
    
    col1, col2 = st.columns(2)
    
    col1.metric("Duplicate Rows", dup_info['duplicate_count'])
    col2.metric("Duplicate %", f"{dup_info['duplicate_percentage']:.2f}%")
    
    if dup_info['duplicate_count'] > 0:
        with st.expander("View Duplicate Rows"):
            st.dataframe(duplicates.head(50), use_container_width=True)
    else:
        st.success("✅ No duplicate rows found!")
    
    st.markdown("---")

# Correlation
if "Correlation" in analysis_options:
    st.markdown("### 🔗 Correlation Analysis")
    
    corr_matrix = analyzer.get_correlation_matrix()
    
    if not corr_matrix.empty:
        # Heatmap
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            height=600,
            width=800
        )
        
        st.plotly_chart(fig, use_container_width=True)
        if 'current_figures' in st.session_state:
            st.session_state.current_figures.append(fig)
        
        # Strong correlations
        st.markdown("**Strong Correlations (> 0.7 or < -0.7)**")
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': f"{corr_val:.3f}"
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("No strong correlations found (threshold: 0.7)")
    else:
        st.info("Not enough numeric columns for correlation analysis.")
    
    st.markdown("---")

# Export Analysis
st.markdown("### 📥 Export Analysis")

# Generate report data once
if st.button("📊 Generate Analysis Report", type="primary", use_container_width=True):
    st.session_state.analysis_report = analyzer.generate_full_report()
    st.session_state.selected_sections = analysis_options  # Store selected sections
    st.success("✅ Analysis report generated!")

if 'analysis_report' in st.session_state:
    report = st.session_state.analysis_report
    selected_sections = st.session_state.get('selected_sections', analysis_options)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON Export
        st.markdown("**📄 JSON Format**")
        st.info("Structured data for programmatic use")
        st.download_button(
            label="💾 Download JSON",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"{st.session_state.current_dataset}_analysis.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # PDF Export
        st.markdown("**📑 PDF Format**")
        st.info(f"Includes: {', '.join(selected_sections)}")
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            from io import BytesIO
            from datetime import datetime
            
            # Create PDF
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter,
                                  leftMargin=50, rightMargin=50,
                                  topMargin=50, bottomMargin=50)
            elements = []
            styles = getSampleStyleSheet()
            
            # Custom styles with colors
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#A23B72'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Title
            title = Paragraph(f"<b>Data Analysis Report</b><br/>{st.session_state.current_dataset}", title_style)
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Date
            date_text = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
            elements.append(date_text)
            
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>Note on Charts:</b> The standard PDF includes text and tables. Use the 'Download PDF with all images' button to include charts (generation may take a moment).", styles['Normal']))
            elements.append(Spacer(1, 10))
            
            elements.append(Paragraph(f"Selected Sections: {', '.join(selected_sections)}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Basic Info (if selected)
            if "Basic Info" in selected_sections:
                elements.append(Paragraph("<b>Basic Information</b>", heading_style))
                basic_data = [
                    ['Metric', 'Value'],
                    ['Total Rows', f"{report.get('basic_info', {}).get('rows', 0):,}"],
                    ['Total Columns', f"{report.get('basic_info', {}).get('columns', 0)}"],
                    ['Memory Usage', f"{report.get('basic_info', {}).get('memory_mb', 0):.2f} MB"],
                    ['Missing Cells', f"{report.get('basic_info', {}).get('missing_cells', 0):,}"]
                ]
                t = Table(basic_data, colWidths=[250, 250])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                elements.append(t)
                elements.append(Spacer(1, 20))
            
            # Column Types (if selected)
            if "Column Types" in selected_sections and 'column_types' in report:
                elements.append(Paragraph("<b>Column Types</b>", heading_style))
                col_data = [['Column', 'Type', 'Unique', 'Missing %']]
                for col, info in list(report['column_types'].items())[:20]:
                    col_data.append([
                        col[:30],  # Truncate long names
                        info.get('type', ''),
                        str(info.get('unique', '')),
                        f"{info.get('null_percentage', 0):.1f}%"
                    ])
                t = Table(col_data, colWidths=[150, 100, 100, 100])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                elements.append(t)
                elements.append(Spacer(1, 20))
            
            # Missing Values (if selected)
            if "Missing Values" in selected_sections and 'missing_values' in report:
                elements.append(Paragraph("<b>Missing Values Analysis</b>", heading_style))
                if report['missing_values']:
                    missing_data = [['Column', 'Count', 'Percentage', 'Suggested Action']]
                    for item in report['missing_values'][:15]:
                        # Handle percentage - it might be string or float
                        pct_value = item.get('Missing %', '0%')
                        if isinstance(pct_value, str):
                            pct = float(pct_value.replace('%', ''))
                        else:
                            pct = float(pct_value)
                        
                        if pct > 50:
                            suggestion = "Drop column"
                        elif pct > 20:
                            suggestion = "Use median/mode fill"
                        else:
                            suggestion = "Forward fill or mean fill"
                        
                        missing_data.append([
                            str(item.get('Column', ''))[:25],
                            str(item.get('Missing Count', '')),
                            str(pct_value),
                            suggestion
                        ])
                    t = Table(missing_data, colWidths=[120, 80, 80, 150])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F18F01')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    elements.append(t)
                else:
                    elements.append(Paragraph("✅ No missing values found!", styles['Normal']))
                elements.append(Spacer(1, 20))
            
            # Summary Statistics (if selected)
            if "Summary Statistics" in selected_sections and 'summary_statistics' in report:
                elements.append(Paragraph("<b>Summary Statistics</b>", heading_style))
                elements.append(Paragraph("See JSON export for detailed statistics", styles['Normal']))
                elements.append(Spacer(1, 20))
            
            # Outliers (if selected)
            if "Outliers" in selected_sections:
                elements.append(Paragraph("<b>Outlier Detection</b>", heading_style))
                elements.append(Paragraph("Use IQR or Z-Score methods in the Analyze page for detailed outlier analysis", styles['Normal']))
                elements.append(Spacer(1, 20))
            
            # Duplicates (if selected)
            if "Duplicates" in selected_sections and 'duplicates' in report:
                dup_info = report.get('duplicates', {})
                elements.append(Paragraph("<b>Duplicate Analysis</b>", heading_style))
                dup_data = [
                    ['Metric', 'Value'],
                    ['Duplicate Rows', f"{dup_info.get('count', 0):,}"],
                    ['Percentage', f"{dup_info.get('percentage', 0):.2f}%"]
                ]
                t = Table(dup_data, colWidths=[250, 250])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#C73E1D')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                elements.append(t)
                elements.append(Spacer(1, 20))
            
            # Build PDF
            # Build PDF without images (Standard)
            doc_std = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
            doc_std.build(elements)
            pdf_bytes_std = buffer.getvalue()
            
            # --- START BUILD PDF WITH IMAGES ---
            buffer_img = BytesIO()
            doc_img = SimpleDocTemplate(buffer_img, pagesize=letter, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
            elements_img = list(elements) # copy the elements
            
            if "current_figures" in st.session_state and len(st.session_state.current_figures) > 0:
                from reportlab.platypus import Image as RLImage
                import kaleido
                elements_img.append(PageBreak())
                elements_img.append(Paragraph("<b>Generated Charts</b>", heading_style))
                elements_img.append(Spacer(1, 20))
                for idx, fig in enumerate(st.session_state.current_figures):
                    try:
                        img_bytes = fig.to_image(format="png", width=800, height=500, scale=1.5)
                        img_stream = BytesIO(img_bytes)
                        rl_img = RLImage(img_stream, width=450, height=280)
                        elements_img.append(rl_img)
                        elements_img.append(Spacer(1, 15))
                    except Exception as img_e:
                        pass
            
            doc_img.build(elements_img)
            pdf_bytes_img = buffer_img.getvalue()
            # --- END BUILD PDF WITH IMAGES ---

            col2_inner1, col2_inner2 = st.columns(2)
            
            with col2_inner1:
                st.download_button(
                    label="📑 Download PDF",
                    data=pdf_bytes_std,
                    file_name=f"{st.session_state.current_dataset}_analysis.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2_inner2:
                st.download_button(
                    label="📸 Download PDF with all images",
                    data=pdf_bytes_img,
                    file_name=f"{st.session_state.current_dataset}_analysis_with_charts.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="secondary"
                )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            st.info("PDF export requires reportlab. Make sure all dependencies are installed.")



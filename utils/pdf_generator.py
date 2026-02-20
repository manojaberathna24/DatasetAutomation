"""
PDF Report Generation
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import streamlit as st
from datetime import datetime
import os

class PDFReportGenerator:
    """Generate professional PDF reports"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.doc = SimpleDocTemplate(filepath, pagesize=letter)
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom styles for the PDF"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Body style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            spaceAfter=12
        ))
    
    def add_title_page(self, title, subtitle=None):
        """Add title page to report"""
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        
        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['Heading3']))
        
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        ))
        self.story.append(PageBreak())
    
    def add_section(self, title, content=None):
        """Add a section with title and content"""
        self.story.append(Paragraph(title, self.styles['CustomHeading']))
        if content:
            if isinstance(content, str):
                self.story.append(Paragraph(content, self.styles['CustomBody']))
            elif isinstance(content, list):
                for item in content:
                    self.story.append(Paragraph(f"• {item}", self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_dataframe(self, df, max_rows=20):
        """Add pandas DataFrame as a table"""
        # Limit rows if too many
        if len(df) > max_rows:
            display_df = df.head(max_rows)
            truncated = True
        else:
            display_df = df
            truncated = False
        
        # Prepare data
        data = [display_df.columns.tolist()] + display_df.values.tolist()
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(table)
        
        if truncated:
            self.story.append(Paragraph(
                f"<i>Showing {max_rows} of {len(df)} rows</i>",
                self.styles['Normal']
            ))
        
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_chart_from_plotly(self, fig, width=6*inch, height=4*inch):
        """Add a Plotly chart to the PDF"""
        try:
            # Convert Plotly figure to image
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_buffer = BytesIO(img_bytes)
            
            # Add to PDF
            img = Image(img_buffer, width=width, height=height)
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            st.warning(f"Could not add chart to PDF: {str(e)}")
    
    def add_matplotlib_chart(self, fig, width=6*inch, height=4*inch):
        """Add a matplotlib chart to the PDF"""
        try:
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            
            img = Image(img_buffer, width=width, height=height)
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            st.warning(f"Could not add chart to PDF: {str(e)}")
    
    def add_key_metrics(self, metrics_dict):
        """Add key metrics in a formatted way"""
        data = [[k, str(v)] for k, v in metrics_dict.items()]
        
        table = Table(data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8e8e8')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 12)
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
    
    def build(self):
        """Build the PDF document"""
        try:
            self.doc.build(self.story)
            return True
        except Exception as e:
            st.error(f"Error building PDF: {str(e)}")
            return False


def create_analysis_report(df, analysis_results, filepath):
    """
    Create a comprehensive analysis report
    
    Args:
        df: pandas DataFrame
        analysis_results: dict with analysis results
        filepath: output PDF filepath
    """
    pdf = PDFReportGenerator(filepath)
    
    # Title page
    pdf.add_title_page(
        "DataSense AI - Data Analysis Report",
        "Comprehensive Dataset Analysis"
    )
    
    # Dataset Overview
    pdf.add_section("Dataset Overview")
    pdf.add_key_metrics({
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Memory Usage (MB)": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f}",
        "Missing Values": df.isnull().sum().sum()
    })
    
    # Column Information
    pdf.add_section("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    pdf.add_dataframe(col_info)
    
    # Summary Statistics
    if len(df.select_dtypes(include=['number']).columns) > 0:
        pdf.add_section("Summary Statistics")
        pdf.add_dataframe(df.describe())
    
    # Build PDF
    return pdf.build()

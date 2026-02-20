"""
Data visualization utilities using Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from config import COLOR_SCHEMES

class DataVisualizer:
    """Create interactive visualizations"""
    
    def __init__(self, df, color_scheme='vibrant'):
        self.df = df
        self.colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['vibrant'])
    
    def create_bar_chart(self, x_col, y_col=None, agg_func='count', title=None, orientation='v'):
        """Create bar chart with aggregation"""
        try:
            if y_col is None or agg_func == 'count':
                # Simple count bar chart
                data = self.df[x_col].value_counts().reset_index()
                data.columns = [x_col, 'Count']
                fig = px.bar(
                    data, 
                    x=x_col if orientation == 'v' else 'Count',
                    y='Count' if orientation == 'v' else x_col,
                    title=title or f'Count by {x_col}',
                    color_discrete_sequence=self.colors,
                    orientation=orientation
                )
            else:
                # Aggregated bar chart
                if agg_func == 'sum':
                    data = self.df.groupby(x_col)[y_col].sum().reset_index()
                elif agg_func == 'mean':
                    data = self.df.groupby(x_col)[y_col].mean().reset_index()
                elif agg_func == 'median':
                    data = self.df.groupby(x_col)[y_col].median().reset_index()
                elif agg_func == 'max':
                    data = self.df.groupby(x_col)[y_col].max().reset_index()
                elif agg_func == 'min':
                    data = self.df.groupby(x_col)[y_col].min().reset_index()
                else:
                    data = self.df.groupby(x_col)[y_col].count().reset_index()
                
                fig = px.bar(
                    data,
                    x=x_col if orientation == 'v' else y_col,
                    y=y_col if orientation == 'v' else x_col,
                    title=title or f'{agg_func.title()} of {y_col} by {x_col}',
                    color_discrete_sequence=self.colors,
                    orientation=orientation
                )
            
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def create_pie_chart(self, column, title=None, top_n=10):
        """Create pie chart"""
        try:
            data = self.df[column].value_counts().head(top_n)
            
            fig = px.pie(
                values=data.values,
                names=data.index,
                title=title or f'Distribution of {column}',
                color_discrete_sequence=self.colors
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            return None
    
    def create_line_chart(self, x_col, y_col, color_col=None, title=None):
        """Create line chart"""
        try:
            fig = px.line(
                self.df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title or f'{y_col} over {x_col}',
                color_discrete_sequence=self.colors
            )
            
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating line chart: {str(e)}")
            return None
    
    def create_histogram(self, column, bins=30, title=None):
        """Create histogram"""
        try:
            fig = px.histogram(
                self.df,
                x=column,
                nbins=bins,
                title=title or f'Distribution of {column}',
                color_discrete_sequence=self.colors
            )
            
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating histogram: {str(e)}")
            return None
    
    def create_scatter_plot(self, x_col, y_col, color_col=None, size_col=None, title=None):
        """Create scatter plot"""
        try:
            plot_df = self.df.copy()
            if size_col:
                # Plotly crashes if size column contains NaNs
                plot_df[size_col] = plot_df[size_col].fillna(0)
                
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                title=title or f'{y_col} vs {x_col}',
                color_discrete_sequence=self.colors
            )
            
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
            return None
    
    def create_heatmap(self, title=None):
        """Create correlation heatmap"""
        try:
            numeric_df = self.df.select_dtypes(include=['number'])
            
            if numeric_df.empty:
                st.warning("No numeric columns for correlation heatmap")
                return None
            
            corr_matrix = numeric_df.corr()
            
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
                title=title or 'Correlation Heatmap',
                template='plotly_dark',
                height=600,
                width=800
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return None
    
    def create_box_plot(self, y_col, x_col=None, title=None):
        """Create box plot"""
        try:
            fig = px.box(
                self.df,
                x=x_col,
                y=y_col,
                title=title or f'Box Plot of {y_col}',
                color=x_col,
                color_discrete_sequence=self.colors
            )
            
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")
            return None
    
    def create_violin_plot(self, y_col, x_col=None, title=None):
        """Create violin plot"""
        try:
            fig = px.violin(
                self.df,
                x=x_col,
                y=y_col,
                title=title or f'Violin Plot of {y_col}',
                color=x_col,
                color_discrete_sequence=self.colors,
                box=True
            )
            
            fig.update_layout(template='plotly_dark', height=500)
            return fig
        except Exception as e:
            st.error(f"Error creating violin plot: {str(e)}")
            return None

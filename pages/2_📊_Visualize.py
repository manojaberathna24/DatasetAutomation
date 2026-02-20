"""
Data Visualization Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import DataVisualizer
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
import pandas as pd

st.set_page_config(page_title="Visual Analytics", page_icon="📈", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown("""
<div class="terminal-header">
    <h1>VISUAL ANALYTICS WORKSPACE</h1>
    <p>Advanced Data Visualization & Pattern Recognition</p>
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
        <p>Upload a dataset first to initialize visualization terminal</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.datasets[st.session_state.current_dataset]

# Sidebar for chart selection
with st.sidebar:
    st.markdown("### CHART CONFIGURATION")
    
    chart_type = st.selectbox(
        "Chart Type",
        ["Bar Chart", "Pie Chart", "Line Chart", "Histogram", 
         "Scatter Plot", "Heatmap", "Box Plot", "Violin Plot"]
    )
    
    color_scheme = st.selectbox(
        "Color Scheme",
        ["vibrant", "professional", "dark", "ocean", "sunset"]
    )
    
    st.markdown("---")
    st.markdown(f"**Current Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown(f"**Columns:** {len(df.columns)}")

# Initialize visualizer
viz = DataVisualizer(df, color_scheme=color_scheme)

# Chart creation based on type
st.markdown(f"### {chart_type}")

if chart_type == "Bar Chart":
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("X-axis (Category)", df.columns)
        orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
    
    with col2:
        use_aggregation = st.checkbox("Use Aggregation", value=False)
        
        if use_aggregation:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            y_col = st.selectbox("Y-axis (Numeric)", numeric_cols)
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "median", "count", "max", "min"])
        else:
            y_col = None
            agg_func = "count"
    
    custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary"):
        fig = viz.create_bar_chart(
            x_col, y_col, agg_func, 
            title=custom_title if custom_title else None,
            orientation='h' if orientation == "Horizontal" else 'v'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            
            # Download options
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # HTML download
                html_str = fig.to_html()
                st.download_button(
                    label="📄 Download HTML",
                    data=html_str,
                    file_name=f"{chart_type.replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with col_d2:
                # PNG download
                st.info("💡 Tip: To download as a PNG image, hover over the chart and click the **Camera (📷)** icon in the top right menu!", icon="ℹ️")


elif chart_type == "Pie Chart":
    col1, col2 = st.columns(2)
    
    with col1:
        column = st.selectbox("Column", df.columns)
        top_n = st.slider("Top N Categories", 5, 20, 10)
    
    with col2:
        custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="pie_gen"):
        fig = viz.create_pie_chart(
            column, 
            title=custom_title if custom_title else None,
            top_n=top_n
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            
            # Download options
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                st.info("💡 Tip: To download as a PNG image, hover over the chart and click the **Camera (📷)** icon in the top right menu!", icon="ℹ️")

elif chart_type == "Line Chart":
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.select_dtypes(include=['number']).columns)
    
    with col2:
        color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
        custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="line_gen"):
        fig = viz.create_line_chart(
            x_col, y_col, color_col,
            title=custom_title if custom_title else None
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                try:
                    st.download_button("🖼️ Download PNG", fig.to_image(format="png", width=1200, height=800), f"{chart_type.replace(' ', '_')}.png", "image/png", use_container_width=True)
                except:
                    st.info("Install kaleido for PNG export: pip install kaleido")

elif chart_type == "Histogram":
    col1, col2 = st.columns(2)
    
    with col1:
        column = st.selectbox("Column", df.select_dtypes(include=['number']).columns)
        bins = st.slider("Number of Bins", 10, 100, 30)
    
    with col2:
        custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="hist_gen"):
        fig = viz.create_histogram(
            column, bins,
            title=custom_title if custom_title else None
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                try:
                    st.download_button("🖼️ Download PNG", fig.to_image(format="png", width=1200, height=800), f"{chart_type.replace(' ', '_')}.png", "image/png", use_container_width=True)
                except:
                    st.info("Install kaleido for PNG export")

elif chart_type == "Scatter Plot":
    col1, col2 = st.columns(2)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
    
    with col2:
        color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
        size_col = st.selectbox("Size by (optional)", [None] + numeric_cols)
        custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="scatter_gen"):
        fig = viz.create_scatter_plot(
            x_col, y_col, color_col, size_col,
            title=custom_title if custom_title else None
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                try:
                    st.download_button("🖼️ Download PNG", fig.to_image(format="png", width=1200, height=800), f"{chart_type.replace(' ', '_')}.png", "image/png", use_container_width=True)
                except:
                    st.info("Install kaleido for PNG export")

elif chart_type == "Heatmap":
    st.info("Correlation heatmap for numeric columns")
    custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="heatmap_gen"):
        fig = viz.create_heatmap(
            title=custom_title if custom_title else None
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                try:
                    st.download_button("🖼️ Download PNG", fig.to_image(format="png", width=1200, height=800), f"{chart_type.replace(' ', '_')}.png", "image/png", use_container_width=True)
                except:
                    st.info("Install kaleido for PNG export")

elif chart_type == "Box Plot":
    col1, col2 = st.columns(2)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    with col1:
        y_col = st.selectbox("Y-axis (Numeric)", numeric_cols)
        x_col = st.selectbox("Group by (optional)", [None] + df.columns.tolist())
    
    with col2:
        custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="box_gen"):
        fig = viz.create_box_plot(
            y_col, x_col,
            title=custom_title if custom_title else None
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                try:
                    st.download_button("🖼️ Download PNG", fig.to_image(format="png", width=1200, height=800), f"{chart_type.replace(' ', '_')}.png", "image/png", use_container_width=True)
                except:
                    st.info("Install kaleido for PNG export")

elif chart_type == "Violin Plot":
    col1, col2 = st.columns(2)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    with col1:
        y_col = st.selectbox("Y-axis (Numeric)", numeric_cols)
        x_col = st.selectbox("Group by (optional)", [None] + df.columns.tolist())
    
    with col2:
        custom_title = st.text_input("Custom Title (optional)")
    
    if st.button("Generate Chart", type="primary", key="violin_gen"):
        fig = viz.create_violin_plot(
            y_col, x_col,
            title=custom_title if custom_title else None
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
            st.markdown("### 📥 Download Chart")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download HTML", fig.to_html(), f"{chart_type.replace(' ', '_')}.html", "text/html", use_container_width=True)
            with col_d2:
                try:
                    st.download_button("🖼️ Download PNG", fig.to_image(format="png", width=1200, height=800), f"{chart_type.replace(' ', '_')}.png", "image/png", use_container_width=True)
                except:
                    st.info("Install kaleido for PNG export")

# Chart Gallery
st.markdown("---")
st.markdown("### 📚 Chart Gallery")
st.markdown("Quick examples of different chart types:")

with st.expander("📊 View Chart Examples"):
    cols = st.columns(2)
    
    # Example charts if numeric columns exist
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) > 0:
        with cols[0]:
            st.markdown("**Histogram Example**")
            fig = viz.create_histogram(numeric_cols[0])
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})
        
        with cols[1]:
            if len(numeric_cols) > 1:
                st.markdown("**Correlation Heatmap**")
                fig = viz.create_heatmap()
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'format': 'png', 'filename': 'chart_export', 'height': 800, 'width': 1200, 'scale': 2}, 'displayModeBar': True})

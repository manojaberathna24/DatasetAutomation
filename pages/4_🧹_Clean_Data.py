"""
Data Cleaning Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cleaning import DataCleaner
from utils.analysis import DataAnalyzer
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
import pandas as pd
from config import CLEANED_DIR

st.set_page_config(page_title="Data Sanitization", page_icon="🔧", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown("""
<div class="terminal-header">
    <h1>DATA SANITIZATION WORKSPACE</h1>
    <p>Advanced Data Cleaning & Optimization System</p>
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
        <p>Upload a dataset first to initialize cleaning terminal</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.datasets[st.session_state.current_dataset]

# Initialize cleaner and analyzer
if 'cleaner' not in st.session_state or st.session_state.get('cleaner_dataset') != st.session_state.current_dataset:
    st.session_state.cleaner = DataCleaner(df)
    st.session_state.cleaner_dataset = st.session_state.current_dataset

cleaner = st.session_state.cleaner
analyzer = DataAnalyzer(df)

# Sidebar
with st.sidebar:
    st.markdown("### Cleaning Options")
    
    if st.button("🔄 Reset to Original", type="secondary"):
        cleaner.reset()
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Original Rows:** {len(df):,}")
    st.markdown(f"**Current Rows:** {len(cleaner.get_cleaned_df()):,}")

# Issue Detection
st.markdown("### 🔍 Detected Issues")

col1, col2, col3 = st.columns(3)

missing_count = df.isnull().sum().sum()
dup_count = len(df[df.duplicated()])
numeric_cols = df.select_dtypes(include=['number']).columns
outlier_count = 0

for col in numeric_cols:
    _, info = analyzer.detect_outliers_iqr(col)
    outlier_count += info.get('outlier_count', 0)

with col1:
    st.metric("⚠️ Missing Values", f"{missing_count:,}", 
             delta=f"{(missing_count / df.size * 100):.1f}% of total cells")

with col2:
    st.metric("🔄 Duplicate Rows", f"{dup_count:,}",
             delta=f"{(dup_count / len(df) * 100):.1f}% of total rows")

with col3:
    st.metric("🎯 Potential Outliers", f"{outlier_count:,}",
             delta="Based on IQR method")

st.markdown("---")

# AI Cleaning Assistant
st.markdown("### 🤖 AI Cleaning Assistant")
with st.expander("💡 Smart Cleaning Recommendations", expanded=True):
    hints = []
    
    # Missing values hints
    if missing_count > 0:
        missing_cols = df.columns[df.isnull().any()]
        hints.append(f"**Missing Values Detected:**")
        for col in list(missing_cols)[:5]:
            pct = (df[col].isnull().sum() / len(df)) * 100
            if pct > 50:
                hints.append(f"  • ⚠️ **{col}**: {pct:.1f}% missing → ⭐ Consider dropping this column (too many missing values)")
            elif pct > 20:
                hints.append(f"  • 📊 **{col}**: {pct:.1f}% missing → ⭐ Use median/mode fill for numeric/categorical data")
            else:
                hints.append(f"  • ✓ **{col}**: {pct:.1f}% missing → ⭐ Forward fill or mean/median fill recommended")
        hints.append("")
    
    # Duplicates hints
    if dup_count > 0:
        dup_pct = (dup_count / len(df)) * 100
        if dup_pct > 10:
            hints.append(f"**Duplicates:** Found {dup_count:,} duplicate rows ({dup_pct:.1f}%)")
            hints.append(f"  ⭐ **Recommendation:** High duplicate rate - remove duplicates, keep first occurrence")
        else:
            hints.append(f"**Duplicates:** Found {dup_count:,} duplicate rows ({dup_pct:.1f}%)")
            hints.append(f"  ⭐ **Recommendation:** Remove duplicates to improve data quality")
        hints.append("")
    
    # Outliers hints
    if outlier_count > 0:
        hints.append(f"**Outliers:** Detected {outlier_count:,} potential outliers across numeric columns")
        hints.append(f"  ⭐ **Recommendations:**")
        hints.append(f"    • Cap outliers (winsorize) to reduce extreme values without losing data")
        hints.append(f"    • Remove outliers if they represent errors or anomalies")
        hints.append(f"    • Mark outliers for further investigation")
        hints.append("")
    
    # Data quality score
    total_cells = df.size
    issues = missing_count + (dup_count * len(df.columns))
    quality_score = max(0, 100 - (issues / total_cells * 100))
    
    st.markdown(f"**📊 Data Quality Score:** {quality_score:.1f}/100")
    if quality_score >= 90:
        st.success("✅ Excellent data quality!")
    elif quality_score >= 70:
        st.info("ℹ️ Good data quality, minor cleaning recommended")
    else:
        st.warning("⚠️ Poor data quality, cleaning highly recommended")
    
    st.markdown("")
    for hint in hints:
        st.markdown(hint)
    
    if not hints:
        st.success("✅ No issues detected! Your data looks clean.")

st.markdown("---")

# Cleaning Operations
tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Duplicates", "Outliers", "Preview & Export"])

# Tab 1: Missing Values
with tab1:
    st.markdown("### ⚠️ Handle Missing Values")
    
    missing_df = analyzer.analyze_missing_values()
    
    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Cleaning Strategy",
                ["drop_rows", "drop_columns", "fill_mean", "fill_median", 
                 "fill_mode", "forward_fill", "backward_fill", "fill_custom"]
            )
        
        with col2:
            columns_with_missing = df.columns[df.isnull().any()].tolist()
            selected_cols = st.multiselect(
                "Select Columns (leave empty for all)",
                columns_with_missing
            )
        
        if strategy == "fill_custom":
            fill_value = st.text_input("Custom Fill Value")
        else:
            fill_value = None
        
        if st.button("🧹 Apply Cleaning", type="primary", key="missing_clean"):
            with st.spinner("Cleaning..."):
                success = cleaner.handle_missing_values(
                    strategy=strategy,
                    columns=selected_cols if selected_cols else None,
                    fill_value=fill_value
                )
                
                if success:
                    st.success("✅ Missing values cleaned successfully!")
                    st.rerun()
    else:
        st.success("✅ No missing values found!")

# Tab 2: Duplicates
with tab2:
    st.markdown("### 🔄 Handle Duplicates")
    
    duplicates, dup_info = analyzer.detect_duplicates()
    
    col1, col2 = st.columns(2)
    col1.metric("Duplicate Rows", dup_info['duplicate_count'])
    col2.metric("Percentage", f"{dup_info['duplicate_percentage']:.2f}%")
    
    if dup_info['duplicate_count'] > 0:
        with st.expander("View Duplicates"):
            st.dataframe(duplicates.head(20), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            subset_cols = st.multiselect(
                "Consider columns (leave empty for all)",
                df.columns.tolist()
            )
        
        with col2:
            keep = st.selectbox(
                "Keep which duplicate?",
                ["first", "last", False],
                format_func=lambda x: "First occurrence" if x == "first" else 
                             "Last occurrence" if x == "last" else "Remove all duplicates"
            )
        
        if st.button("🗑️ Remove Duplicates", type="primary", key="dup_clean"):
            with st.spinner("Removing duplicates..."):
                success = cleaner.remove_duplicates(
                    subset=subset_cols if subset_cols else None,
                    keep=keep
                )
                
                if success:
                    st.success("✅ Duplicates removed successfully!")
                    st.rerun()
    else:
        st.success("✅ No duplicates found!")

# Tab 3: Outliers
with tab3:
    st.markdown("### 🎯 Handle Outliers")
    
    if len(numeric_cols) > 0:
        method = st.radio("Detection Method", ["IQR", "Z-Score"], horizontal=True)
        
        selected_cols_outlier = st.multiselect(
            "Select Numeric Columns",
            numeric_cols.tolist(),
            default=numeric_cols.tolist()[:3] if len(numeric_cols) >= 3 else numeric_cols.tolist()
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            action = st.selectbox(
                "Action",
                ["remove", "cap", "mark"],
                format_func=lambda x: "Remove outliers" if x == "remove" else
                             "Cap outliers (winsorize)" if x == "cap" else
                             "Mark outliers (add column)"
            )
        
        with col2:
            if method == "IQR":
                multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
                threshold = None
            else:
                threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
                multiplier = None
        
        # Show outlier preview
        if selected_cols_outlier:
            st.markdown("**Outlier Preview:**")
            
            for col in selected_cols_outlier[:3]:  # Show first 3
                if method == "IQR":
                    outliers, info = analyzer.detect_outliers_iqr(col)
                else:
                    outliers, info = analyzer.detect_outliers_zscore(col, threshold)
                
                st.write(f"**{col}**: {info.get('outlier_count', 0)} outliers ({info.get('outlier_percentage', 0):.1f}%)")
        
        if st.button("🧹 Apply Outlier Handling", type="primary", key="outlier_clean"):
            with st.spinner("Handling outliers..."):
                if method == "IQR":
                    success = cleaner.handle_outliers_iqr(
                        selected_cols_outlier,
                        action=action,
                        multiplier=multiplier
                    )
                else:
                    success = cleaner.handle_outliers_zscore(
                        selected_cols_outlier,
                        action=action,
                        threshold=threshold
                    )
                
                if success:
                    st.success("✅ Outliers handled successfully!")
                    st.rerun()
    else:
        st.info("No numeric columns available for outlier detection.")

# Tab 4: Preview & Export
with tab4:
    st.markdown("### 👀 Before & After Comparison")
    
    cleaned_df = cleaner.get_cleaned_df()
    summary = cleaner.get_cleaning_summary()
    
    # Comparison metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{summary['cleaned_rows']:,}", 
                 delta=f"{summary['cleaned_rows'] - summary['original_rows']:,}")
    with col2:
        st.metric("Columns", summary['cleaned_columns'],
                 delta=f"{summary['cleaned_columns'] - summary['original_columns']}")
    with col3:
        missing_before = df.isnull().sum().sum()
        missing_after = cleaned_df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_after:,}",
                 delta=f"{missing_after - missing_before:,}")
    with col4:
        dup_before = len(df[df.duplicated()])
        dup_after = len(cleaned_df[cleaned_df.duplicated()])
        st.metric("Duplicates", f"{dup_after:,}",
                 delta=f"{dup_after - dup_before:,}")
    
    st.markdown("---")
    
    # Side-by-side comparison
    st.markdown("### 📊 Data Preview Comparison")
    
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.markdown("**🔴 Before Cleaning**")
        st.dataframe(df.head(10), use_container_width=True, height=400)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    with col_after:
        st.markdown("**🟢 After Cleaning**")
        st.dataframe(cleaned_df.head(10), use_container_width=True, height=400)
        st.caption(f"Shape: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns")
    
    st.markdown("---")
    
    # Operations log
    st.markdown("### 📝 Cleaning Operations Log")
    if summary['operations']:
        for i, op in enumerate(summary['operations'], 1):
            st.write(f"{i}. ✓ {op}")
    else:
        st.info("No cleaning operations applied yet.")
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 💾 Export Cleaned Data")
    st.info("Download your cleaned dataset in multiple formats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📄 CSV Format**")
        st.caption("Universal format for data analysis")
        csv = cleaned_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{st.session_state.current_dataset}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        st.markdown("**📊 Excel Format**")
        st.caption("For Microsoft Excel")
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            cleaned_df.to_excel(writer, index=False)
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f"{st.session_state.current_dataset}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
    
    with col3:
        st.markdown("**📋 JSON Format**")
        st.caption("For APIs and web apps")
        json_str = cleaned_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{st.session_state.current_dataset}_cleaned.json",
            mime="application/json",
            use_container_width=True,
            type="primary"
        )
    
    st.markdown("---")
    
    # Save to session
    st.markdown("### 💾 Save to Session")
    if st.button("💿 Save Cleaned Dataset for Further Analysis", use_container_width=True):
        cleaned_name = f"{st.session_state.current_dataset}_cleaned"
        st.session_state.datasets[cleaned_name] = cleaned_df
        if 'cleaned_datasets' not in st.session_state:
            st.session_state.cleaned_datasets = {}
        st.session_state.cleaned_datasets[cleaned_name] = cleaned_df
        st.success(f"✅ Saved as '{cleaned_name}' - Now available in all analysis pages!")


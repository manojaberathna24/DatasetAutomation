"""
AutoML Page - Terminal Interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ml_models import MLPipeline
from utils.terminal_theme import get_terminal_css, get_hacker_emojis
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import MODELS_DIR

st.set_page_config(page_title="AutoML Engine", page_icon="🎯", layout="wide")

# Apply terminal theme
st.markdown(get_terminal_css(), unsafe_allow_html=True)
emojis = get_hacker_emojis()

st.markdown(f"""
<div class="terminal-header">
    <h1>{emojis['robot']} AUTOML ENGINE {emojis['robot']}</h1>
    <p>{emojis['fire']} Automated Machine Learning & Model Training System</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Check if dataset exists
if not st.session_state.current_dataset:
    st.markdown(f"""
    <div class="terminal-block" style="border-color: #ff0000;">
        <p style="color: #ff0000;">{emojis['warning']} <strong>NO ACTIVE DATASET</strong></p>
        <p style="color: #00ffff;">Upload a dataset first to initialize AutoML engine</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.datasets[st.session_state.current_dataset]

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 ML Task Selection")
    
    task_type = st.selectbox(
        "Task Type",
        ["Classification", "Regression", "Clustering"]
    )
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Rows:** {len(df):,}")

# Help Section
st.markdown("### ℹ️ What is AutoML?")
with st.expander("📚 Learn about AutoML and how to use it", expanded=False):
    st.markdown("""
    **AutoML (Automated Machine Learning)** trains multiple machine learning models automatically and selects the best one.
    
    ### 🎯 Task Types Explained
    
    **1. Classification** - Predict categories/classes
    - Example: Email spam detection (spam vs. not spam)
    - Example: Customer churn prediction (will leave vs. will stay)
    - Example: Disease diagnosis (positive vs. negative)
    - **Use when:** Your target column has categories (Yes/No, High/Medium/Low, etc.)
    
    **2. Regression** - Predict numeric values
    - Example: House price prediction
    - Example: Sales forecasting
    - Example: Temperature prediction
    - **Use when:** Your target column has numbers you want to predict
    
    **3. Clustering** - Group similar items together
    - Example: Customer segmentation
    - Example: Product categorization
    - Example: Anomaly detection
    - **Use when:** You want to find natural groups in your data (no target column needed)
    
    ### 🚀 How to Use
    1. Select your task type (Classification/Regression/Clustering)
    2. Choose target column (what to predict) - not needed for clustering
    3. Select feature columns (what to use for prediction)
    4. Click "Train Models" - sit back while we train multiple algorithms
    5. Review model comparison and download the best model
    
    ### 📊 Metrics Explained
    - **Accuracy**: % of correct predictions (higher is better)
    - **Precision**: Of predicted positives, how many are correct
    - **Recall**: Of actual positives, how many we found
    - **F1 Score**: Balance between precision and recall
    - **R² Score**: How well model explains variance (1.0 = perfect)
    - **RMSE**: Average prediction error (lower is better)
    """)

st.markdown("---")

if task_type in ["Classification", "Regression"]:
    st.markdown(f"### {task_type} Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target selection
        target_column = st.selectbox(
            "🎯 Target Column (What to predict)",
            df.columns.tolist()
        )
    
    with col2:
        # Feature selection
        available_features = [col for col in df.columns if col != target_column]
        feature_columns = st.multiselect(
            "📊 Feature Columns (Leave empty for all)",
            available_features,
            default=[]
        )
        
        if not feature_columns:
            feature_columns = available_features
    
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("🚀 Train Models", type="primary", key="train_ml"):
        with st.spinner("🧠 Training multiple models... This may take a minute..."):
            # Initialize pipeline
            pipeline = MLPipeline(df, task_type=task_type.lower())
            
            # Prepare data
            X_train, X_test, y_train, y_test, features = pipeline.prepare_data(
                target_column, feature_columns, test_size
            )
            
            if X_train is not None:
                # Train models
                if task_type == "Classification":
                    results = pipeline.train_classification_models(X_train, X_test, y_train, y_test)
                else:
                    results = pipeline.train_regression_models(X_train, X_test, y_train, y_test)
                
                # Store in session
                st.session_state.ml_results = results
                st.session_state.ml_pipeline = pipeline
                st.session_state.ml_task = task_type
                
                st.success("✅ Models trained successfully!")
                st.rerun()
    
    # Display results if available
    if 'ml_results' in st.session_state and st.session_state.get('ml_task') == task_type:
        results = st.session_state.ml_results
        pipeline = st.session_state.ml_pipeline
        
        st.markdown("---")
        st.markdown("### 📊 Model Comparison")
        
        if task_type == "Classification":
            # Create comparison DataFrame
            comp_df = pd.DataFrame([
                {
                    'Model': name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1 Score': f"{metrics['f1']:.4f}"
                }
                for name, metrics in results.items()
            ])
            
            st.dataframe(comp_df, use_container_width=True)
            
            # Best model
            best_name, best_model, best_metrics = pipeline.get_best_model()
            st.success(f"🏆 Best Model: **{best_name}** (Accuracy: {best_metrics['accuracy']:.4f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig = px.bar(
                    x=list(results.keys()),
                    y=[metrics['accuracy'] for metrics in results.values()],
                    title="Model Accuracy Comparison",
                    labels={'x': 'Model', 'y': 'Accuracy'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confusion Matrix for best model
                cm = best_metrics['confusion_matrix']
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    text=cm,
                    texttemplate='%{text}',
                    colorscale='Blues'
                ))
                fig.update_layout(
                    title=f"Confusion Matrix - {best_name}",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            with st.expander("📋 Detailed Classification Report"):
                st.text(best_metrics['classification_report'])
            
            # ROC Curve if binary
            if 'roc_auc' in best_metrics:
                st.markdown("### 📈 ROC Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=best_metrics['fpr'],
                    y=best_metrics['tpr'],
                    mode='lines',
                    name=f'ROC (AUC = {best_metrics["roc_auc"]:.3f})'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(dash='dash')
                ))
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Regression
            # Create comparison DataFrame
            comp_df = pd.DataFrame([
                {
                    'Model': name,
                    'R² Score': f"{metrics['r2']:.4f}",
                    'RMSE': f"{metrics['rmse']:.4f}",
                    'MAE': f"{metrics['mae']:.4f}",
                    'MSE': f"{metrics['mse']:.4f}"
                }
                for name, metrics in results.items()
            ])
            
            st.dataframe(comp_df, use_container_width=True)
            
            # Best model
            best_name, best_model, best_metrics = pipeline.get_best_model()
            st.success(f"🏆 Best Model: **{best_name}** (R² Score: {best_metrics['r2']:.4f})")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # R² comparison
                fig = px.bar(
                    x=list(results.keys()),
                    y=[metrics['r2'] for metrics in results.values()],
                    title="Model R² Score Comparison",
                    labels={'x': 'Model', 'y': 'R² Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # RMSE comparison
                fig = px.bar(
                    x=list(results.keys()),
                    y=[metrics['rmse'] for metrics in results.values()],
                    title="Model RMSE Comparison",
                    labels={'x': 'Model', 'y': 'RMSE'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            st.markdown("### 📉 Residual Plot (Best Model)")
            residuals = best_metrics['actual'] - best_metrics['predictions']
            fig = px.scatter(
                x=best_metrics['predictions'],
                y=residuals,
                title="Residual Plot",
                labels={'x': 'Predicted Values', 'y': 'Residuals'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export model
        st.markdown("---")
        st.markdown("### 💾 Export Model")
        
        model_to_export = st.selectbox("Select Model to Export", list(results.keys()))
        
        if st.button("📥 Download Model (.pkl)", key="download_model"):
            import joblib
            import tempfile
            
            filepath = os.path.join(MODELS_DIR, f"{model_to_export}.pkl")
            pipeline.save_model(model_to_export, filepath)
            
            with open(filepath, 'rb') as f:
                st.download_button(
                    label="Download Model File",
                    data=f.read(),
                    file_name=f"{model_to_export}_{st.session_state.current_dataset}.pkl",
                    mime="application/octet-stream"
                )

else:  # Clustering
    st.markdown("### 🎯 Clustering Analysis")
    
    # Feature selection
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Clustering requires at least 2 numeric columns")
        st.stop()
    
    feature_columns = st.multiselect(
        "📊 Select Features for Clustering",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    
    if st.button("🚀 Run Clustering", type="primary", key="train_cluster"):
        with st.spinner("🧠 Running clustering algorithms..."):
            # Prepare data
            X = df[feature_columns].copy()
            X = X.fillna(X.mean())
            
            # Initialize pipeline
            pipeline = MLPipeline(df, task_type='clustering')
            X_scaled, _, _, _, _ = pipeline.prepare_data(None, feature_columns)
            
            # Train clustering models
            results = pipeline.train_clustering_models(X_scaled, n_clusters)
            
            # Store in session
            st.session_state.cluster_results = results
            st.session_state.cluster_pipeline = pipeline
            st.session_state.cluster_features = feature_columns
            
            st.success("✅ Clustering completed!")
            st.rerun()
    
    # Display results
    if 'cluster_results' in st.session_state:
        results = st.session_state.cluster_results
        features = st.session_state.cluster_features
        
        st.markdown("---")
        st.markdown("### 📊 Clustering Results")
        
        # Comparison
        comp_df = pd.DataFrame([
            {
                'Algorithm': name,
                'Clusters Found': metrics['n_clusters'],
                'Silhouette Score': f"{metrics['silhouette_score']:.4f}"
            }
            for name, metrics in results.items()
        ])
        
        st.dataframe(comp_df, use_container_width=True)
        
        # Visualize clusters
        for name, metrics in results.items():
            with st.expander(f"📌 {name} - Visualization"):
                labels = metrics['labels']
                
                # Add labels to dataframe
                viz_df = df[features].copy()
                viz_df['Cluster'] = labels
                
                # 2D visualization (first 2 features)
                if len(features) >= 2:
                    fig = px.scatter(
                        viz_df,
                        x=features[0],
                        y=features[1],
                        color='Cluster',
                        title=f"{name} Clusters",
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster sizes
                st.markdown("**Cluster Sizes:**")
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title="Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)

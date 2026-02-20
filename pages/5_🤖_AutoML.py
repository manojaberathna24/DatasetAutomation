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

st.markdown("""
<div class="terminal-header">
    <h1>AUTOML ENGINE</h1>
    <p>Automated Machine Learning & Model Training System</p>
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
        <p>Upload a dataset first to initialize AutoML engine</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.datasets[st.session_state.current_dataset]

# Sidebar
with st.sidebar:
    st.markdown("### ML Task Selection")
    
    task_type = st.selectbox(
        "Task Type",
        ["Classification", "Regression", "Clustering"]
    )
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {st.session_state.current_dataset}")
    st.markdown(f"**Rows:** {len(df):,}")

# Help Section
st.markdown("### What is AutoML?")
with st.expander("Learn about AutoML and how to use it", expanded=False):
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
            "Target Column (What to predict)",
            df.columns.tolist()
        )
    
    with col2:
        # Feature selection
        available_features = [col for col in df.columns if col != target_column]
        feature_columns = st.multiselect(
            "Feature Columns (Leave empty for all)",
            available_features,
            default=[]
        )
        
        if not feature_columns:
            feature_columns = available_features
    
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Train Models", type="primary", key="train_ml"):
        with st.spinner("Training multiple models... This may take a minute..."):
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
                st.session_state.train_features = features
                st.session_state.train_target = target_column
                
                st.success("Models trained successfully!")
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
            if best_name is None:
                st.error("❌ No models could be trained. Check the logs for errors.")
                st.stop()
            st.success(f"Best Model: **{best_name}** (Accuracy: {best_metrics['accuracy']:.4f})")
            
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
            if best_name is None:
                st.error("❌ No models could be trained. Check the logs for errors.")
                st.stop()
            st.success(f"Best Model: **{best_name}** (R² Score: {best_metrics['r2']:.4f})")
            
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
        model_to_export = st.selectbox("Select Model to Export / Test", list(results.keys()))
        
        col1, col2 = st.columns(2)
        import joblib
        import pickle
        import tempfile
        
        # 1. Download PKL
        filepath_pkl = os.path.join(MODELS_DIR, f"{model_to_export}.pkl")
        pipeline.save_model(model_to_export, filepath_pkl)
        with open(filepath_pkl, 'rb') as f:
            col1.download_button(
                label="📥 Download Model (.pkl)",
                data=f.read(),
                file_name=f"{model_to_export}_{st.session_state.current_dataset}.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            
        # 2. Download JOBLIB
        filepath_joblib = os.path.join(MODELS_DIR, f"{model_to_export}.joblib")
        # Save explicitly as joblib
        best_model = pipeline.models.get(model_to_export)
        if best_model:
            joblib.dump(best_model, filepath_joblib)
            with open(filepath_joblib, 'rb') as f:
                col2.download_button(
                    label="📥 Download Model (.joblib)",
                    data=f.read(),
                    file_name=f"{model_to_export}_{st.session_state.current_dataset}.joblib",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        st.markdown("---")
        



    st.markdown("---")
    st.markdown("### 🧪 Test the Model Live")
    st.info("Enter values for the features below to get a live prediction. You can either test the model you just trained, or upload a previously exported model!")
    
    # Check if a model exists in memory
    memory_model_exists = 'ml_results' in st.session_state and st.session_state.get('ml_task') == task_type
    
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_model_file = st.file_uploader("📥 Upload a .pkl or .joblib model", type=['pkl', 'joblib'])
    
    custom_model_dict = None
    custom_model_name = ""
    
    if uploaded_model_file is not None:
        import tempfile
        import joblib
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_model_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_model_file.getvalue())
            tmp_filepath = tmp_file.name
            
        try:
            loaded_data = joblib.load(tmp_filepath)
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                custom_model_dict = loaded_data
                custom_model_name = uploaded_model_file.name
                
                # Check for legacy models
                if 'features' not in custom_model_dict or not custom_model_dict['features']:
                    st.warning("⚠️ This model was exported with an older version and is missing feature metadata. Prediction might fail if the current dataset doesn't match.")
                else:
                    st.success(f"✅ Successfully loaded custom model: {custom_model_name}")
            else:
                st.error("Invalid model file format. Expected a dictionary containing 'model', 'scaler', etc.")
        except Exception as e:
            st.error(f"Error loading uploaded model: {e}")
            
    # Determine which model to test
    active_test_model = None
    active_test_name = ""
    features_to_test = []
    target_name = ""
    use_custom_pipeline = False
    
    if custom_model_dict is not None:
        active_test_model = custom_model_dict['model']
        active_test_name = custom_model_name
        features_to_test = custom_model_dict.get('features', [])
        target_name = custom_model_dict.get('target', '')
        use_custom_pipeline = True
    elif memory_model_exists:
        try:
            model_to_export = st.session_state.get('last_model_to_export', list(st.session_state.ml_results.keys())[0])
            active_test_model = st.session_state.ml_pipeline.models.get(model_to_export)
            active_test_name = model_to_export
            features_to_test = st.session_state.get('train_features', [])
            target_name = st.session_state.get('train_target', '')
        except Exception:
            pass

    if not features_to_test:
        # Fallback for models missing feature lists
        train_target = target_name if target_name else st.session_state.get('train_target', '')
        
        # If we still don't know the target but we are loading a custom model, we can try to guess it
        # by checking if there's a label_encoder named 'target'
        if not train_target and custom_model_dict is not None:
            encoders = custom_model_dict.get('label_encoders', {})
            # It's hard to guess the exact column name if 'target' is the only key,
            # but usually, the last column in the dataset is the target
            # Let's check common target names in this dataset or use the last column
            common_targets = ['Loan_Status', 'Status', 'Target', 'Class', 'y']
            for ct in common_targets:
                if ct in df.columns:
                    train_target = ct
                    break
            if not train_target:
                train_target = df.columns[-1]

        
        features_to_test = [c for c in df.columns if c != train_target]
        
        # Another safety check: If the model has a known number of features, we should only take that many
        if active_test_model is not None and hasattr(active_test_model, 'n_features_in_'):
            expected_n = active_test_model.n_features_in_
            if len(features_to_test) > expected_n:
                # Truncate to expected number
                features_to_test = features_to_test[:expected_n]
                
        # Remove target if it accidentally ended up in the features list
        if 'Loan_Status' in features_to_test:
             features_to_test.remove('Loan_Status')

    if active_test_model is not None:
        with st.form("test_model_form"):
            st.markdown(f"**📝 Enter Data for {active_test_name}**")
            test_inputs = {}
            
            f_cols = st.columns(2)
            for i, feat in enumerate(features_to_test):
                col_idx = i % 2
                if feat not in df.columns:
                    continue  # Safety check
                    
                feat_type = df[feat].dtype
                clean_feat_data = df[feat].dropna()
                example_val = clean_feat_data.iloc[0] if not clean_feat_data.empty else ""
                
                with f_cols[col_idx]:
                    if pd.api.types.is_numeric_dtype(feat_type):
                        min_val = float(clean_feat_data.min()) if not clean_feat_data.empty else 0.0
                        max_val = float(clean_feat_data.max()) if not clean_feat_data.empty else 100.0
                        mean_val = float(clean_feat_data.mean()) if not clean_feat_data.empty else 0.0
                        
                        # Only use slider if range is reasonable and not ID-like
                        if min_val != max_val and len(clean_feat_data.unique()) > 2:
                            test_inputs[feat] = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=mean_val)
                        else:
                            test_inputs[feat] = st.number_input(f"{feat}", value=mean_val)
                    else:
                        unique_vals = clean_feat_data.unique().tolist()
                        # Use horizontal radio buttons for binary choices (like Gender: Male/Female)
                        if len(unique_vals) <= 3:
                            test_inputs[feat] = st.radio(f"{feat}", options=unique_vals, horizontal=True)
                        else:
                            test_inputs[feat] = st.selectbox(f"{feat}", options=unique_vals)
            
            submit_test = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)
            
        if submit_test:
            with st.spinner("Predicting..."):
                try:
                    input_df = pd.DataFrame([test_inputs])
                    
                    if use_custom_pipeline:
                        # Prepare data manually using custom_model_dict elements
                        X_input = input_df.copy()
                        custom_encoders = custom_model_dict.get('label_encoders', {})
                        custom_scaler = custom_model_dict.get('scaler')
                        
                        for col in X_input.select_dtypes(include=['object']).columns:
                            if col in custom_encoders:
                                le = custom_encoders[col]
                                try:
                                    X_input[col] = le.transform(X_input[col].astype(str))
                                except ValueError:
                                    X_input[col] = 0
                            else:
                                X_input[col] = 0
                        
                        X_input = X_input.fillna(0)
                        if custom_scaler:
                            X_input = custom_scaler.transform(X_input)
                            
                        prediction = active_test_model.predict(X_input)
                    else:
                        # Use active session pipeline
                        X_input, _, _, _, _ = st.session_state.ml_pipeline.prepare_data(target_name, features_to_test, is_training=False, input_data=input_df)
                        prediction = active_test_model.predict(X_input)
                    
                    # Inverse transform target label if it was encoded
                    result = prediction[0]
                    if use_custom_pipeline:
                        encoders = custom_model_dict.get('label_encoders', {})
                        if 'target' in encoders:
                            try:
                                result = encoders['target'].inverse_transform([result])[0]
                            except: pass
                    else:
                        if 'target' in st.session_state.ml_pipeline.label_encoders:
                            try:
                                result = st.session_state.ml_pipeline.label_encoders['target'].inverse_transform([result])[0]
                            except: pass
                            
                    st.success(f"### 🎉 Result: {result}")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Make sure the model was trained on these exact features, or the uploaded model matches this dataset.")
    elif not memory_model_exists:
        st.info("Upload a model above to begin testing, or train a new model first!")

else:  # Clustering
    st.markdown("### 🎯 Clustering Analysis")
    
    # Feature selection
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Clustering requires at least 2 numeric columns")
        st.stop()
    
    feature_columns = st.multiselect(
        "Select Features for Clustering",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    
    if st.button("Run Clustering", type="primary", key="train_cluster"):
        with st.spinner("Running clustering algorithms..."):
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

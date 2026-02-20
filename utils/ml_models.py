"""
Machine Learning model utilities
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             mean_squared_error, r2_score, mean_absolute_error)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import joblib
import streamlit as st

class MLPipeline:
    """Machine Learning pipeline for training and evaluating models"""
    
    def __init__(self, df, task_type='classification'):
        self.df = df
        self.task_type = task_type
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def prepare_data(self, target_column, feature_columns=None, test_size=0.2, is_training=True, input_data=None):
        """Prepare data for training or prediction"""
        try:
            # Select features
            if feature_columns is None:
                feature_columns = [col for col in self.df.columns if col != target_column]
            
            if is_training:
                X = self.df[feature_columns].copy()
            else:
                X = input_data[feature_columns].copy()
            
            # Handle categorical features
            for col in X.select_dtypes(include=['object']).columns:
                if is_training:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen labels by filling with the most common or a default (prevent crash)
                        try:
                            X[col] = le.transform(X[col].astype(str))
                        except ValueError:
                            # If a completely new category is introduced in testing, fallback to 0
                            X[col] = 0
                    else:
                        X[col] = 0
            
            # Handle missing values
            if is_training:
                self.feature_means = X.mean()
                X = X.fillna(self.feature_means)
            else:
                if hasattr(self, 'feature_means'):
                    X = X.fillna(self.feature_means)
                else:
                    X = X.fillna(0)
            
            if not is_training:
                X_scaled = self.scaler.transform(X)
                return X_scaled, None, None, None, feature_columns
                
            if self.task_type in ['classification', 'regression']:
                y = self.df[target_column]
                
                # Encode target for classification
                if self.task_type == 'classification' and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    self.label_encoders['target'] = le
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                
                return X_train, X_test, y_train, y_test, feature_columns
            
            else:  # clustering
                X = self.scaler.fit_transform(X)
                return X, None, None, None, feature_columns
                
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None, None, None, None
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train multiple classification models"""
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        for name, model in models_to_train.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, zero_division=0)
                }
                
                # ROC curve for binary classification
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                    metrics['roc_auc'] = auc(fpr, tpr)
                    metrics['fpr'] = fpr
                    metrics['tpr'] = tpr
                
                self.models[name] = model
                self.results[name] = metrics
                
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
        
        return self.results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train multiple regression models"""
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'XGBoost': XGBRegressor(random_state=42)
        }
        
        for name, model in models_to_train.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                self.models[name] = model
                self.results[name] = metrics
                
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
        
        return self.results
    
    def train_clustering_models(self, X, n_clusters=3):
        """Train clustering models"""
        models_to_train = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters)
        }
        
        for name, model in models_to_train.items():
            try:
                # Fit model
                labels = model.fit_predict(X)
                
                # Metrics
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                else:
                    silhouette = 0
                
                metrics = {
                    'n_clusters': len(np.unique(labels)),
                    'silhouette_score': silhouette,
                    'labels': labels,
                    'cluster_sizes': np.bincount(labels[labels >= 0])
                }
                
                self.models[name] = model
                self.results[name] = metrics
                
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
        
        return self.results
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            return None, None, None
        
        if self.task_type == 'classification':
            best_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        elif self.task_type == 'regression':
            best_name = max(self.results, key=lambda x: self.results[x]['r2'])
        else:  # clustering
            best_name = max(self.results, key=lambda x: self.results[x]['silhouette_score'])
        
        self.best_model = self.models[best_name]
        return best_name, self.best_model, self.results[best_name]
    
    def save_model(self, model_name, filepath):
        """Save model to file"""
        try:
            if model_name in self.models:
                # Always save features and target so the testing UI knows what to ask for
                features = st.session_state.get('train_features', [])
                target = st.session_state.get('train_target', '')
                joblib.dump({
                    'model': self.models[model_name],
                    'scaler': self.scaler,
                    'label_encoders': self.label_encoders,
                    'features': features,
                    'target': target,
                    'task_type': self.task_type
                }, filepath)
                return True
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
        return False
    
    def load_model(self, filepath):
        """Load model from file"""
        try:
            data = joblib.load(filepath)
            return data['model'], data['scaler'], data.get('label_encoders', {})
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None

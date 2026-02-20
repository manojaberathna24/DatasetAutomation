"""
Data cleaning utilities
"""
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

class DataCleaner:
    """Utilities for data cleaning operations"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = []
    
    def handle_missing_values(self, strategy='drop_rows', columns=None, fill_value=None):
        """
        Handle missing values
        
        Args:
            strategy: 'drop_rows', 'drop_columns', 'fill_mean', 'fill_median', 
                     'fill_mode', 'forward_fill', 'backward_fill', 'fill_custom'
            columns: List of columns to apply strategy (None = all with missing values)
            fill_value: Custom value for 'fill_custom' strategy
        """
        if columns is None:
            columns = self.df.columns[self.df.isnull().any()].tolist()
        
        try:
            if strategy == 'drop_rows':
                before_count = len(self.df)
                self.df = self.df.dropna(subset=columns)
                self.cleaning_log.append(f"Dropped {before_count - len(self.df)} rows with missing values in {columns}")
            
            elif strategy == 'drop_columns':
                self.df = self.df.drop(columns=columns)
                self.cleaning_log.append(f"Dropped columns: {columns}")
            
            elif strategy == 'fill_mean':
                for col in columns:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                        self.cleaning_log.append(f"Filled {col} with mean value")
            
            elif strategy == 'fill_median':
                for col in columns:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                        self.cleaning_log.append(f"Filled {col} with median value")
            
            elif strategy == 'fill_mode':
                for col in columns:
                    mode_value = self.df[col].mode()
                    if len(mode_value) > 0:
                        self.df[col].fillna(mode_value[0], inplace=True)
                        self.cleaning_log.append(f"Filled {col} with mode value")
            
            elif strategy == 'forward_fill':
                self.df[columns] = self.df[columns].ffill()
                self.cleaning_log.append(f"Forward filled columns: {columns}")
            
            elif strategy == 'backward_fill':
                self.df[columns] = self.df[columns].bfill()
                self.cleaning_log.append(f"Backward filled columns: {columns}")
            
            elif strategy == 'fill_custom' and fill_value is not None:
                self.df[columns] = self.df[columns].fillna(fill_value)
                self.cleaning_log.append(f"Filled columns {columns} with custom value: {fill_value}")
            
            return True
        except Exception as e:
            st.error(f"Error handling missing values: {str(e)}")
            return False
    
    def remove_duplicates(self, subset=None, keep='first'):
        """
        Remove duplicate rows
        
        Args:
            subset: Columns to consider for identifying duplicates (None = all columns)
            keep: 'first', 'last', or False (remove all duplicates)
        """
        try:
            before_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=subset, keep=keep)
            removed = before_count - len(self.df)
            self.cleaning_log.append(f"Removed {removed} duplicate rows")
            return True
        except Exception as e:
            st.error(f"Error removing duplicates: {str(e)}")
            return False
    
    def handle_outliers_iqr(self, columns, action='remove', multiplier=1.5):
        """
        Handle outliers using IQR method
        
        Args:
            columns: List of numeric columns
            action: 'remove', 'cap' (winsorize), or 'mark'
            multiplier: IQR multiplier (default 1.5)
        """
        try:
            for col in columns:
                if self.df[col].dtype not in ['int64', 'float64']:
                    continue
                
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                if action == 'remove':
                    before_count = len(self.df)
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    removed = before_count - len(self.df)
                    self.cleaning_log.append(f"Removed {removed} outliers from {col} using IQR")
                
                elif action == 'cap':
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.cleaning_log.append(f"Capped outliers in {col} using IQR")
                
                elif action == 'mark':
                    self.df[f'{col}_outlier'] = ((self.df[col] < lower_bound) | 
                                                  (self.df[col] > upper_bound))
                    self.cleaning_log.append(f"Marked outliers in {col}")
            
            return True
        except Exception as e:
            st.error(f"Error handling outliers: {str(e)}")
            return False
    
    def handle_outliers_zscore(self, columns, action='remove', threshold=3):
        """
        Handle outliers using Z-score method
        
        Args:
            columns: List of numeric columns
            action: 'remove', 'cap', or 'mark'
            threshold: Z-score threshold (default 3)
        """
        try:
            for col in columns:
                if self.df[col].dtype not in ['int64', 'float64']:
                    continue
                
                # Calculate z-scores (handle NaN values)
                col_data = self.df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                mean = col_data.mean()
                std = col_data.std()
                
                if std == 0:  # Avoid division by zero
                    continue
                
                # Calculate z-scores for all values (NaN will remain NaN)
                z_scores = np.abs((self.df[col] - mean) / std)
                
                if action == 'remove':
                    before_count = len(self.df)
                    # Keep rows where z-score is within threshold OR value is NaN
                    mask = (z_scores <= threshold) | self.df[col].isna()
                    self.df = self.df[mask]
                    removed = before_count - len(self.df)
                    self.cleaning_log.append(f"Removed {removed} outliers from {col} using Z-score")
                
                elif action == 'cap':
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.cleaning_log.append(f"Capped outliers in {col} using Z-score")
                
                elif action == 'mark':
                    # Mark rows as outliers (True if outlier, False otherwise, NaN stays NaN)
                    self.df[f'{col}_outlier_z'] = z_scores > threshold
                    self.cleaning_log.append(f"Marked outliers in {col} using Z-score")
            
            return True
        except Exception as e:
            st.error(f"Error handling outliers: {str(e)}")
            return False
    
    def convert_data_types(self, column, target_type):
        """Convert column to target data type"""
        try:
            if target_type == 'int':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
            elif target_type == 'float':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            elif target_type == 'str':
                self.df[column] = self.df[column].astype(str)
            elif target_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            elif target_type == 'category':
                self.df[column] = self.df[column].astype('category')
            
            self.cleaning_log.append(f"Converted {column} to {target_type}")
            return True
        except Exception as e:
            st.error(f"Error converting data type: {str(e)}")
            return False
    
    def get_cleaned_df(self):
        """Return cleaned DataFrame"""
        return self.df
    
    def get_cleaning_summary(self):
        """Get summary of cleaning operations"""
        summary = {
            'original_rows': len(self.original_df),
            'cleaned_rows': len(self.df),
            'rows_removed': len(self.original_df) - len(self.df),
            'original_columns': len(self.original_df.columns),
            'cleaned_columns': len(self.df.columns),
            'columns_removed': len(self.original_df.columns) - len(self.df.columns),
            'operations': self.cleaning_log
        }
        return summary
    
    def drop_empty_columns(self):
        """Drop columns that are 100% NaN"""
        try:
            empty_cols = self.df.columns[self.df.isnull().all()].tolist()
            if empty_cols:
                self.df = self.df.drop(columns=empty_cols)
                self.cleaning_log.append(f"Dropped completely empty columns: {empty_cols}")
                return True
            return False
        except Exception as e:
            st.error(f"Error dropping empty columns: {str(e)}")
            return False
            
    def reset(self):
        """Reset to original DataFrame"""
        self.df = self.original_df.copy()
        self.cleaning_log = []

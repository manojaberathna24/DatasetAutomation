"""
Data analysis utilities
"""
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

class DataAnalyzer:
    """Comprehensive data analysis utilities"""
    
    def __init__(self, df):
        self.df = df
    
    def get_column_types(self):
        """Detect and classify column types"""
        column_info = {}
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            null_count = self.df[col].isnull().sum()
            
            # Classify column
            if dtype in ['int64', 'float64']:
                col_type = 'Numeric'
            elif dtype == 'object':
                if unique_count < 20:
                    col_type = 'Categorical'
                else:
                    col_type = 'Text'
            elif dtype in ['datetime64', 'datetime64[ns]']:
                col_type = 'DateTime'
            else:
                col_type = 'Other'
            
            column_info[col] = {
                'dtype': dtype,
                'type': col_type,
                'unique': unique_count,
                'null_count': null_count,
                'null_percentage': (null_count / len(self.df)) * 100
            }
        
        return column_info
    
    def analyze_missing_values(self):
        """Comprehensive missing value analysis"""
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Count': self.df.isnull().sum().values,
            'Missing %': (self.df.isnull().sum() / len(self.df) * 100).values,
            'Data Type': self.df.dtypes.astype(str).values
        })
        
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(
            by='Missing %', ascending=False
        )
        
        return missing_data
    
    def detect_outliers_iqr(self, column):
        """Detect outliers using IQR method"""
        if self.df[column].dtype not in ['int64', 'float64']:
            return [], {}
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        info = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(self.df)) * 100
        }
        
        return outliers.index.tolist(), info
    
    def detect_outliers_zscore(self, column, threshold=3):
        """Detect outliers using Z-score method"""
        if self.df[column].dtype not in ['int64', 'float64']:
            return [], {}
        
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        
        info = {
            'threshold': threshold,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(self.df)) * 100,
            'mean': self.df[column].mean(),
            'std': self.df[column].std()
        }
        
        return outlier_indices.tolist(), info
    
    def get_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        summary = self.df[numeric_cols].describe().T
        
        # Add additional statistics
        summary['variance'] = self.df[numeric_cols].var()
        summary['skewness'] = self.df[numeric_cols].skew()
        summary['kurtosis'] = self.df[numeric_cols].kurtosis()
        summary['missing'] = self.df[numeric_cols].isnull().sum()
        
        return summary
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        
        return self.df[numeric_cols].corr()
    
    def detect_duplicates(self):
        """Detect duplicate rows"""
        duplicates = self.df[self.df.duplicated(keep=False)]
        
        info = {
            'duplicate_count': len(duplicates),
            'duplicate_percentage': (len(duplicates) / len(self.df)) * 100,
            'unique_duplicate_groups': len(duplicates.drop_duplicates())
        }
        
        return duplicates, info
    
    def get_value_counts(self, column, top_n=10):
        """Get value counts for a column"""
        return self.df[column].value_counts().head(top_n)
    
    def generate_full_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'basic_info': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024**2)
            },
            'column_types': self.get_column_types(),
            'missing_values': self.analyze_missing_values().to_dict('records'),
            'summary_statistics': self.get_summary_statistics().to_dict(),
            'duplicates': self.detect_duplicates()[1]
        }
        
        # Add outliers for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        for col in numeric_cols:
            _, info = self.detect_outliers_iqr(col)
            outlier_info[col] = info
        report['outliers'] = outlier_info
        
        return report

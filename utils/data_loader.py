"""
Data loading and validation utilities
"""
import pandas as pd
import streamlit as st
from io import BytesIO
import json
from config import GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS, MAX_FILE_SIZE_BYTES, UPLOAD_DIR
import os

# Optional Google Cloud Storage import
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

class DataLoader:
    """Handle dataset loading from various sources"""
    
    def __init__(self):
        self.gcs_client = None
        if GCS_AVAILABLE and GCS_BUCKET_NAME and GOOGLE_APPLICATION_CREDENTIALS:
            try:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(GCS_BUCKET_NAME)
            except Exception as e:
                st.warning(f"⚠️ Google Cloud Storage not configured. Using local storage. Error: {str(e)}")
    
    def load_file(self, uploaded_file):
        """
        Load file from Streamlit uploader
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            pandas.DataFrame or None if error
        """
        try:
            # Check file size
            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(f"❌ File size exceeds {MAX_FILE_SIZE_BYTES / (1024**2):.0f}MB limit")
                return None
            
            # Get file extension
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            # Load based on file type
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'json':
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"❌ Unsupported file format: {file_ext}")
                return None
            
            # Validate DataFrame
            if df.empty:
                st.error("❌ Uploaded file is empty")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    
    def save_to_storage(self, df, filename):
        """
        Save DataFrame to Google Cloud Storage or local storage
        
        Args:
            df: pandas.DataFrame
            filename: Name for the saved file
        """
        try:
            if self.gcs_client:
                # Save to GCS
                blob = self.bucket.blob(f"uploads/{filename}.csv")
                blob.upload_from_string(df.to_csv(index=False), content_type='text/csv')
                return f"gs://{GCS_BUCKET_NAME}/uploads/{filename}.csv"
            else:
                # Save to local storage
                filepath = os.path.join(UPLOAD_DIR, f"{filename}.csv")
                df.to_csv(filepath, index=False)
                return filepath
        except Exception as e:
            st.error(f"❌ Error saving file: {str(e)}")
            return None
    
    def load_from_storage(self, filepath):
        """
        Load DataFrame from storage
        
        Args:
            filepath: Path to file (GCS or local)
            
        Returns:
            pandas.DataFrame or None
        """
        try:
            if filepath.startswith("gs://"):
                # Load from GCS
                blob_name = filepath.replace(f"gs://{GCS_BUCKET_NAME}/", "")
                blob = self.bucket.blob(blob_name)
                content = blob.download_as_bytes()
                df = pd.read_csv(BytesIO(content))
            else:
                # Load from local
                df = pd.read_csv(filepath)
            return df
        except Exception as e:
            st.error(f"❌ Error loading from storage: {str(e)}")
            return None
    
    def get_file_info(self, df, filename):
        """
        Get metadata about the dataset
        
        Args:
            df: pandas.DataFrame
            filename: Name of the file
            
        Returns:
            dict with metadata
        """
        return {
            'name': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'column_names': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).to_dict()
        }

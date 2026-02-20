"""
Data loading and validation utilities
"""
import pandas as pd
import streamlit as st
from io import BytesIO
import json
from config import GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS, MAX_FILE_SIZE_BYTES, UPLOAD_DIR
import os
import zipfile
import patoolib
import tempfile
import shutil

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
            elif file_ext in ['zip', 'rar']:
                # Save uploaded file temporarily
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    extract_dir = os.path.join(temp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    # Extract based on extension
                    try:
                        if file_ext == 'zip':
                            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                        elif file_ext == 'rar':
                            try:
                                patoolib.extract_archive(temp_file_path, outdir=extract_dir, interactive=False)
                            except patoolib.util.PatoolError as e:
                                st.error("❌ RAR extraction failed. Please ensure 'WinRAR' or '7z' is installed on your OS and added to PATH.")
                                return None
                    except Exception as e:
                        st.error(f"❌ Error extracting archive: {str(e)}")
                        return None
                        
                    # Find compatible file
                    extracted_files = []
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            extracted_files.append(os.path.join(root, file))
                    
                    data_file = None
                    for file in extracted_files:
                        # Ignore macOS metadata files
                        if "__MACOSX" in file or os.path.basename(file).startswith("._"):
                            continue
                            
                        # Check extensions
                        if any(file.lower().endswith(ext) for ext in ['.csv', '.xlsx', '.xls', '.json']):
                            data_file = file
                            break
                    
                    if not data_file:
                        found_names = [os.path.basename(f) for f in extracted_files]
                        if not found_names:
                            st.error("❌ The archive is completely empty or extraction failed silently.")
                        else:
                            st.error(f"❌ No supported data file (CSV, Excel, JSON) found.")
                            st.info(f"Files found in archive: {', '.join(found_names[:10])}{'...' if len(found_names)>10 else ''}")
                        return None
                    
                    # Read the extracted file
                    data_ext = data_file.split('.')[-1].lower()
                    if data_ext == 'csv':
                        df = pd.read_csv(data_file)
                    elif data_ext in ['xlsx', 'xls']:
                        df = pd.read_excel(data_file)
                    elif data_ext == 'json':
                        df = pd.read_json(data_file)
                    
                    # Update file name to reflect the extracted file
                    uploaded_file.name = os.path.basename(data_file)
            else:
                st.error(f"❌ Unsupported file format: {file_ext}")
                return None
            
            # Validate DataFrame
            if df is None or df.empty:
                st.error("❌ Uploaded file is empty or could not be parsed")
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

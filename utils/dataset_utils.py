"""
Utility to get current dataset, preferring cleaned version
"""
import streamlit as st

def get_current_dataset():
    """
    Get the current dataset to use - automatically uses cleaned version if available
    Returns: (dataframe, dataset_name, is_cleaned)
    """
    if not st.session_state.current_dataset:
        return None, None, False
    
    base_name = st.session_state.current_dataset
    cleaned_name = f"{base_name}_cleaned"
    
    # Check if cleaned version exists
    if 'cleaned_datasets' in st.session_state and cleaned_name in st.session_state.cleaned_datasets:
        return st.session_state.cleaned_datasets[cleaned_name], cleaned_name, True
    
    # Return original
    if base_name in st.session_state.datasets:
        return st.session_state.datasets[base_name], base_name, False
    
    return None, None, False

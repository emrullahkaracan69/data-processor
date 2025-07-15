"""
Data Loading Module
Handles loading and initial validation of CSV and Excel files
"""

import pandas as pd
import streamlit as st


def load_data(uploaded_file):
    """
    Load CSV or Excel file with error handling
    
    Parameters:
    -----------
    uploaded_file : streamlit.UploadedFile
        File uploaded through Streamlit
        
    Returns:
    --------
    pd.DataFrame : Loaded dataframe or None if error
    """
    try:
        # Check file extension
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception:
                    uploaded_file.seek(0)  # Reset file pointer
                    continue
                    
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            return df
            
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None



def validate_data(df, max_size_mb=200):
    """
    Validate loaded data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    max_size_mb : int
        Maximum allowed file size in MB
        
    Returns:
    --------
    dict : Validation results
    """
    # Calculate dataframe size in MB
    size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    validation_results = {
        'is_valid': True,
        'size_mb': size_mb,
        'rows': df.shape[0],
        'columns': df.shape[1],
        'messages': []
    }
    
    # Check size
    if size_mb > max_size_mb:
        validation_results['is_valid'] = False
        validation_results['messages'].append(f"File size ({size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)")
    
    # Check if empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['messages'].append("Dataset is empty")
    
    # Check minimum requirements
    if df.shape[0] < 10:
        validation_results['messages'].append("Warning: Dataset has less than 10 rows")
    
    if df.shape[1] < 2:
        validation_results['messages'].append("Warning: Dataset has less than 2 columns")
    
    return validation_results
"""
Data Preprocessing Tool
A professional web application for data preprocessing and outlier handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Import custom modules
from src.outlier_handler import *
from src.data_loader import load_data, validate_data
from src.utils import get_data_summary, create_missing_value_plot, create_outlier_plot



# Page configuration
st.set_page_config(
    page_title="Data Preprocessing Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-section {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #0066cc;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)



# Initialize session state
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None
    if 'processing_steps' not in st.session_state:
        st.session_state['processing_steps'] = []
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 'upload'



# Download functions
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')



def create_download_button(df, filename="processed_data.csv"):
    """Create download button for processed data"""
    csv = convert_df_to_csv(df)
    st.download_button(
        label="📥 Download Processed Data",
        data=csv,
        file_name=filename,
        mime='text/csv',
        key='download_button'
    )



# Display functions
def display_data_info(df):
    """Display data information in a nice format"""
    summary = get_data_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{summary['shape'][0]:,}")
    with col2:
        st.metric("Columns", summary['shape'][1])
    with col3:
        st.metric("Missing Values", f"{summary['missing_values']:,}")
    with col4:
        st.metric("Duplicates", f"{summary['duplicate_rows']:,}")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Memory (MB)", f"{summary['memory_usage_mb']:.2f}")
    with col6:
        st.metric("Numeric Cols", summary['numeric_columns'])
    with col7:
        st.metric("Categorical Cols", summary['categorical_columns'])
    with col8:
        st.metric("Missing %", f"{summary['missing_percentage']:.2f}%")



# Main application
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🔧 Professional Data Preprocessing Tool")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        
        # Show current step
        if st.session_state['data'] is None:
            st.info("👆 Please upload a dataset to begin")
        else:
            st.success("✅ Dataset loaded successfully!")
            
            # Navigation buttons
            if st.button("🏠 Start Over", use_container_width=True):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()
            
            st.markdown("---")
            st.markdown("### Processing Steps:")
            st.markdown("1. 📤 Upload Data")
            st.markdown("2. 🔍 Data Overview")
            st.markdown("3. 🎯 Outlier Detection")
            st.markdown("4. 🧹 Data Cleaning")
            st.markdown("5. 💾 Download Results")
    
    # Main content area
    if st.session_state['current_step'] == 'upload':
        upload_section()
    elif st.session_state['current_step'] == 'overview':
        overview_section()
    elif st.session_state['current_step'] == 'outlier':
        outlier_section()
    elif st.session_state['current_step'] == 'cleaning':
        cleaning_section()
    elif st.session_state['current_step'] == 'download':
        download_section()


# Section functions
def upload_section():
    """File upload section"""
    st.header("📤 Upload Your Dataset")
    
    # Create upload area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### Drag and drop your file here")
        st.markdown("**Supported formats:** CSV, Excel (xlsx, xls)")
        st.markdown("**Maximum size:** 200 MB")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset for preprocessing"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File Type": uploaded_file.type
        }
        
        with col2:
            st.markdown("### File Details")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        # Load data button
        if st.button("🚀 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                
                if df is not None:
                    # Validate data
                    validation = validate_data(df)
                    
                    if validation['is_valid']:
                        st.session_state['data'] = df
                        st.session_state['processed_data'] = df.copy()
                        st.session_state['current_step'] = 'overview'
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
                    else:
                        for msg in validation['messages']:
                            st.error(msg)


def overview_section():
    """Data overview section"""
    st.header("🔍 Data Overview")
    
    df = st.session_state['data']
    
    # Display data info
    st.subheader("📊 Dataset Statistics")
    display_data_info(df)
    
    # Show data preview
    st.markdown("---")
    st.subheader("👀 Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(df.tail(10), use_container_width=True)
    
    with tab3:
        sample_size = min(10, len(df))
        st.dataframe(df.sample(n=sample_size), use_container_width=True)
    
    # Column information
    st.markdown("---")
    st.subheader("📋 Column Information")
    
    # Get column types using our function
    cat_cols, num_cols, cat_but_car, summary = grab_col_names(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.markdown("### Column Categories")
        st.write(f"**Numerical Columns ({len(num_cols)}):**")
        if num_cols:
            st.write(", ".join(num_cols[:10]))
            if len(num_cols) > 10:
                st.write(f"... and {len(num_cols) - 10} more")
        
        st.write(f"**Categorical Columns ({len(cat_cols)}):**")
        if cat_cols:
            st.write(", ".join(cat_cols[:10]))
            if len(cat_cols) > 10:
                st.write(f"... and {len(cat_cols) - 10} more")
        
        if cat_but_car:
            st.write(f"**High Cardinality Columns ({len(cat_but_car)}):**")
            st.write(", ".join(cat_but_car))
    
    # Missing values visualization
    if df.isnull().sum().sum() > 0:
        st.markdown("---")
        st.subheader("❓ Missing Values Analysis")
        
        fig = create_missing_value_plot(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Navigation button
    st.markdown("---")
    if st.button("Continue to Outlier Detection →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 'outlier'
        st.rerun()



def outlier_section():
    """Outlier detection and handling section"""
    st.header("🎯 Outlier Detection & Handling")
    
    df = st.session_state['processed_data']
    
    # Get numerical columns
    cat_cols, num_cols, cat_but_car, summary = grab_col_names(df)
    
    if not num_cols:
        st.warning("No numerical columns found in the dataset!")
        if st.button("Skip to Data Cleaning →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 'cleaning'
            st.rerun()
        return
    
    # Outlier detection settings
    st.subheader("⚙️ Outlier Detection Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["IQR (Interquartile Range)", "Multivariate (LOF)"],
            help="IQR detects outliers per column, LOF detects multivariate outliers"
        )
    
    with col2:
        if detection_method == "IQR (Interquartile Range)":
            q1 = st.slider("Lower Quartile (Q1)", 0.0, 0.5, 0.25, 0.05)
            q3 = st.slider("Upper Quartile (Q3)", 0.5, 1.0, 0.75, 0.05)
        else:
            n_neighbors = st.slider("Number of Neighbors", 5, 50, 20, 5)
    
    with col3:
        handling_method = st.selectbox(
            "Handling Method",
            ["None (Just Detect)", "Remove Outliers", "Cap Outliers"],
            help="Choose how to handle detected outliers"
        )
    
    # Column selection
    st.markdown("---")
    st.subheader("📊 Select Columns for Outlier Analysis")
    
    selected_columns = st.multiselect(
        "Choose numerical columns",
        num_cols,
        default=num_cols[:min(5, len(num_cols))],
        help="Select columns to analyze for outliers"
    )
    
    if selected_columns and st.button("🔍 Detect Outliers", type="primary"):
        
        if detection_method == "IQR (Interquartile Range)":
            # IQR method
            outlier_summary = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, col in enumerate(selected_columns):
                status_text.text(f"Analyzing {col}...")
                
                # Check for outliers
                has_outliers = check_outlier(df, col, q1, q3)
                
                if has_outliers:
                    outlier_indices = grab_outliers(df, col, index=True, q1=q1, q3=q3)
                    low, up = outlier_thresholds(df, col, q1, q3)
                    
                    outlier_summary.append({
                        'Column': col,
                        'Outliers': len(outlier_indices),
                        'Percentage': f"{(len(outlier_indices) / len(df)) * 100:.2f}%",
                        'Lower Limit': f"{low:.2f}",
                        'Upper Limit': f"{up:.2f}"
                    })
                
                progress_bar.progress((idx + 1) / len(selected_columns))
            
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            if outlier_summary:
                st.markdown("---")
                st.subheader("📈 Outlier Detection Results")
                
                results_df = pd.DataFrame(outlier_summary)
                st.dataframe(results_df, use_container_width=True)
                
                # Visualizations
                st.subheader("📊 Outlier Visualizations")
                
                viz_col = st.selectbox("Select column to visualize", 
                                      [row['Column'] for row in outlier_summary])
                
                if viz_col:
                    low, up = outlier_thresholds(df, viz_col, q1, q3)
                    fig = create_outlier_plot(df, viz_col, low, up)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Apply handling method
                if handling_method != "None (Just Detect)":
                    st.markdown("---")
                    st.subheader("🛠️ Apply Outlier Handling")
                    
                    if st.button(f"Apply {handling_method}", type="secondary"):
                        
                        processed_df = df.copy()
                        
                        with st.spinner(f"Applying {handling_method}..."):
                            for col in [row['Column'] for row in outlier_summary]:
                                if handling_method == "Remove Outliers":
                                    processed_df = remove_outlier(processed_df, col, q1, q3)
                                elif handling_method == "Cap Outliers":
                                    processed_df = replace_with_thresholds(processed_df, col, q1, q3)
                        
                        # Update session state
                        st.session_state['processed_data'] = processed_df
                        st.session_state['processing_steps'].append({
                            'step': 'Outlier Handling',
                            'method': handling_method,
                            'details': f"Applied to {len(outlier_summary)} columns",
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Show results
                        st.success(f"✅ {handling_method} applied successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Rows", len(df))
                        with col2:
                            st.metric("Processed Rows", len(processed_df))
                        with col3:
                            st.metric("Rows Affected", len(df) - len(processed_df))
                        
                        st.rerun()
            else:
                st.info("No outliers detected with current settings!")
        
        else:
            # Multivariate LOF method
            with st.spinner("Detecting multivariate outliers..."):
                results = detect_multivariate_outliers(df[selected_columns])
                
                st.markdown("---")
                st.subheader("📈 Multivariate Outlier Detection Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Outliers", results['n_outliers'])
                with col2:
                    st.metric("Percentage", f"{(results['n_outliers'] / len(df)) * 100:.2f}%")
                with col3:
                    st.metric("Threshold", f"{results['threshold']:.4f}")
                
                # Show outlier rows
                if results['n_outliers'] > 0:
                    st.subheader("🔍 Outlier Rows")
                    outlier_df = df.iloc[results['outlier_indices']]
                    st.dataframe(outlier_df.head(20), use_container_width=True)
                    
                    if len(outlier_df) > 20:
                        st.info(f"Showing first 20 of {len(outlier_df)} outlier rows")
    
    # Navigation
    st.markdown("---")
    if st.button("Continue to Data Cleaning →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 'cleaning'
        st.rerun()

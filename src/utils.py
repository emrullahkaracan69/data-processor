"""
Utility Functions Module
Helper functions for data processing and visualization
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def get_data_summary(df):
    """
    Generate comprehensive data summary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict : Summary statistics
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    return summary



def create_missing_value_plot(df):
    """
    Create interactive missing value visualization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    plotly.graph_objects.Figure : Interactive plot
    """
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=True)
    
    if missing_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=missing_df['Missing_Count'],
        y=missing_df['Column'],
        orientation='h',
        text=missing_df['Missing_Percentage'].round(2).astype(str) + '%',
        textposition='auto',
        marker_color='indianred'
    ))
    
    fig.update_layout(
        title='Missing Values by Column',
        xaxis_title='Number of Missing Values',
        yaxis_title='Columns',
        height=max(400, len(missing_df) * 30),
        showlegend=False
    )
    
    return fig



def create_outlier_plot(df, column, low_limit, up_limit):
    """
    Create box plot with outlier thresholds
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name
    low_limit : float
        Lower threshold
    up_limit : float
        Upper threshold
        
    Returns:
    --------
    plotly.graph_objects.Figure : Interactive box plot
    """
    fig = go.Figure()
    
    # Add box plot
    fig.add_trace(go.Box(
        y=df[column],
        name=column,
        boxpoints='outliers',
        marker_color='lightblue',
        line_color='darkblue'
    ))
    
    # Add threshold lines
    fig.add_hline(y=low_limit, line_dash="dash", line_color="red", 
                  annotation_text=f"Lower Limit: {low_limit:.2f}")
    fig.add_hline(y=up_limit, line_dash="dash", line_color="red", 
                  annotation_text=f"Upper Limit: {up_limit:.2f}")
    
    fig.update_layout(
        title=f'Box Plot for {column}',
        yaxis_title=column,
        showlegend=False,
        height=500
    )
    
    return fig
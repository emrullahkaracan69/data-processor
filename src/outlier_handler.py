"""
Outlier Detection and Handling Module
This module contains functions for detecting and handling outliers in datasets
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identifies categorical, numerical, and high-cardinality categorical variables
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    cat_th : int, default=10
        Threshold for numerical variables to be considered categorical
    car_th : int, default=20
        Threshold for categorical variables to be considered high-cardinality
        
    Returns:
    --------
    tuple : (cat_cols, num_cols, cat_but_car)
        Lists of column names for each variable type
    """
    # Categorical columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    
    # Numerical columns that behave like categorical
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    # High cardinality categorical columns
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    
    # Combine categorical columns
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # Numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    # Create summary statistics
    summary = {
        'total_observations': dataframe.shape[0],
        'total_variables': dataframe.shape[1],
        'categorical_cols': len(cat_cols),
        'numerical_cols': len(num_cols),
        'high_cardinality_cols': len(cat_but_car)
    }
    
    return cat_cols, num_cols, cat_but_car, summary



def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate outlier thresholds using IQR method
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    col_name : str
        Column name to calculate thresholds
    q1 : float, default=0.25
        First quartile
    q3 : float, default=0.75
        Third quartile
        
    Returns:
    --------
    tuple : (low_limit, up_limit)
        Lower and upper thresholds for outliers
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    
    return low_limit, up_limit



def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Check if column contains outliers
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    col_name : str
        Column name to check
        
    Returns:
    --------
    bool : True if outliers exist, False otherwise
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



def grab_outliers(dataframe, col_name, index=False, q1=0.25, q3=0.75):
    """
    Get outlier values and optionally their indices
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    col_name : str
        Column name to analyze
    index : bool, default=False
        Whether to return outlier indices
        
    Returns:
    --------
    pd.Index or pd.DataFrame : Outlier indices or outlier data
    """
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    outlier_df = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    
    if index:
        return outlier_df.index
    else:
        return outlier_df



def remove_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Remove outliers from dataframe
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    col_name : str
        Column name to process
        
    Returns:
    --------
    pd.DataFrame : Dataframe without outliers
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | 
                                     (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Replace outliers with threshold values (capping)
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe
    col_name : str
        Column name to process
        
    Returns:
    --------
    pd.DataFrame : Dataframe with capped outliers
    """
    dataframe = dataframe.copy()
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    return dataframe



def detect_multivariate_outliers(dataframe, n_neighbors=20, contamination='auto'):
    """
    Detect multivariate outliers using Local Outlier Factor
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        Input dataframe (only numerical columns)
    n_neighbors : int, default=20
        Number of neighbors for LOF
    contamination : float or 'auto'
        Proportion of outliers in the dataset
        
    Returns:
    --------
    dict : Dictionary containing outlier indices and scores
    """
    # Select only numerical columns
    df_numeric = dataframe.select_dtypes(include=['float64', 'int64'])
    
    # Apply LOF
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    clf.fit_predict(df_numeric)
    
    # Get outlier scores
    df_scores = clf.negative_outlier_factor_
    
    # Find threshold using elbow method (you can adjust this)
    sorted_scores = np.sort(df_scores)
    threshold = sorted_scores[int(len(sorted_scores) * 0.01)]  # 1% most extreme
    
    # Get outlier indices
    outlier_indices = dataframe.index[df_scores < threshold].tolist()
    
    return {
        'outlier_indices': outlier_indices,
        'outlier_scores': df_scores,
        'threshold': threshold,
        'n_outliers': len(outlier_indices)
    }


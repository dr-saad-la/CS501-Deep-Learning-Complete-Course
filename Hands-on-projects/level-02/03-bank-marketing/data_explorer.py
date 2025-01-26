from pprint import pprint
import requests
import zipfile
import io
import shutil
from datetime import datetime
from typing import Tuple
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Type
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score


def get_feature_groups(df: pd.DataFrame, target_col: Optional[str] = 'y',
                       numeric_types: Optional[List] = None,
                       categorical_types: Optional[List] = None
                      ) -> Tuple[List[str], List[str]]:
    """Separates DataFrame features into numerical and categorical groups.
    
    Identifies and separates features into numerical and categorical groups based on their
    data types. Handles edge cases like mixed-type columns and provides options for
    custom type specifications.
    
    Args:
        df: Input DataFrame containing features to be categorized.
        target_col: Name of the target variable to exclude from feature groups.
            Defaults to 'y'. Set to None if no target should be excluded.
        numeric_types: List of numpy/pandas dtypes to consider as numeric.
            Defaults to ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'].
        categorical_types: List of numpy/pandas dtypes to consider as categorical.
            Defaults to ['object', 'category', 'bool'].
            
    Returns:
        A tuple containing:
        - List[str]: Names of numeric feature columns
        - List[str]: Names of categorical feature columns
        
    Raises:
        ValueError: If DataFrame is empty or contains no valid features.
        TypeError: If input is not a pandas DataFrame.
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'income': [50000, 60000, 75000],
        ...     'education': ['BS', 'MS', 'PhD'],
        ...     'y': [0, 1, 1]
        ... })
        >>> num_features, cat_features = get_feature_groups(df)
        >>> print(f"Numeric: {num_features}")
        Numeric: ['age', 'income']
        >>> print(f"Categorical: {cat_features}")
        Categorical: ['education']
    """
    # Input Validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
            
    # Set default type lists if not provided
    if numeric_types is None:
        numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if categorical_types is None:
        categorical_types = ['object', 'category', 'bool']
        
    # Initialize feature lists
    numeric_features = []
    categorical_features = []
    
    # Categorize columns based on their types
    for column in df.columns:
        # Skip target column if specified
        if target_col and column == target_col:
            continue
            
        dtype = df[column].dtype
        # Check numeric types
        if dtype in numeric_types or (
            dtype.name in [str(t) for t in numeric_types]
        ):
            numeric_features.append(column)
        # Check categorical types
        elif dtype in categorical_types or (
            dtype.name in [str(t) for t in categorical_types]
        ):
            categorical_features.append(column)
            
        # Handle mixed types (e.g., numeric columns with missing values as objects)
        elif df[column].dtype == 'object':
            # Try to convert to numeric
            try:
                pd.to_numeric(df[column], errors='raise')
                numeric_features.append(column)
            except (ValueError, TypeError):
                categorical_features.append(column)
    
    # Verify we found some features
    if not numeric_features and not categorical_features:
        raise ValueError("No valid features found in DataFrame")
    
    return numeric_features, categorical_features

def validate_feature_separation(
    numeric_features: List[str], 
    categorical_features: List[str], 
    data: pd.DataFrame, 
    target_col: str = 'y'
) -> None:
    """Validates that all features are properly categorized.
    
    Ensures that the total number of categorized features (numeric + categorical)
    equals the total number of features in the dataset excluding the target variable.
    
    Args:
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
        data: Original pandas DataFrame containing all features.
        target_col: Name of the target column. Defaults to 'y'.
        
    Raises:
        AssertionError: If the feature count validation fails, with details about
            the mismatch.
            
    Examples:
        >>> data = pd.DataFrame({
        ...     'age': [25, 30], 
        ...     'income': [50000, 60000],
        ...     'education': ['BS', 'MS'],
        ...     'y': [0, 1]
        ... })
        >>> num_feat = ['age', 'income']
        >>> cat_feat = ['education']
        >>> validate_feature_separation(num_feat, cat_feat, data)
    """
    expected_count = len(data.columns) - 1       # Excluding target
    actual_count = len(numeric_features) + len(categorical_features)
    
    assert actual_count == expected_count, (
        f"Feature count mismatch: Found {actual_count} features "
        f"({len(numeric_features)} numeric + {len(categorical_features)} categorical) "
        f"but expected {expected_count} features. "
        f"Check for duplicates or missing features."
    )
    
    
def analyze_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Analyzes missing values in a DataFrame.
    
    Calculates comprehensive statistics about missing values in each column,
    including counts, percentages, and data types.
    
    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze.
        threshold (float): Minimum percentage of missing values to include in the result.
            Defaults to 0.0 (show all missing values).
            
    Returns:
        A pandas DataFrame containing missing value analysis with columns:
            - count: Number of missing values
            - percentage: Percentage of missing values
            - dtype: Data type of the column
            - total_rows: Total number of rows in the dataset
            
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [1, None, 3],
        ...     'B': ['x', None, None],
        ...     'C': [1, 2, 3]
        ... })
        >>> missing_analysis = analyze_missing_values(df)
        >>> print(missing_analysis)
               count  percentage   dtype  total_rows
        B         2      66.67  object           3
        A         1      33.33  float64          3
        
    Notes:
        - The output is sorted by percentage of missing values in descending order
        - Only columns with missing values above the threshold are included
        - Percentages are rounded to 2 decimal places
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Calculate missing value statistics
    total_rows = len(df)
    
    missing_stats = pd.DataFrame({
        'count': df.isnull().sum(),
        'percentage': (df.isnull().sum() / total_rows * 100).round(2),
        'dtype': df.dtypes,
        'total_rows': total_rows
    })
    
    # Filter and sort results
    missing_filtered = missing_stats[
        missing_stats['percentage'] > threshold
    ].sort_values('percentage', ascending=False)
    
    if len(missing_filtered) > 0:
        print(f"\nFound missing values in {len(missing_filtered)} columns:")
        print(f"- Highest: {missing_filtered['percentage'].iloc[0]:.2f}% "
              f"in column '{missing_filtered.index[0]}'")
        print(f"- Total rows analyzed: {total_rows}")
    else:
        print("\n✓ No missing values found above threshold "
              f"of {threshold}% in {len(df.columns)} columns")
    
    return missing_filtered


def analyze_target_distribution(
    df: pd.DataFrame, 
    target_col: str = 'y',
    round_digits: int = 3
) -> Dict[str, Union[Dict, float]]:
    """Analyzes the distribution of the target variable in a classification dataset.
    
    Calculates class distribution, absolute counts, and imbalance metrics for
    the target variable.
    
    Args:
        df (pd.DataFrame): The input dataset containing the target variable.
        target_col (str): Name of the target column. Defaults to 'y'.
        round_digits (int): Number of decimal places for rounding. Defaults to 3.
        
    Returns:
        Dict[str, Union[Dict, float]]: A dictionary containing:
            - distribution: Proportion of each class (normalized frequencies)
            - counts: Absolute count of samples in each class
            - imbalance_ratio: Ratio of minority to majority class frequency
            - majority_class: Label of the majority class
            - minority_class: Label of the minority class
            
    Raises:
        KeyError: If target_col is not found in the DataFrame.
        ValueError: If DataFrame is empty or target contains invalid values.
            
    Examples:
        >>> df = pd.DataFrame({'y': [0, 0, 0, 1, 1]})
        >>> result = analyze_target_distribution(df)
        >>> print(result)
        {
            'distribution': {0: 0.6, 1: 0.4},
            'counts': {0: 3, 1: 2},
            'imbalance_ratio': 0.667,
            'majority_class': 0,
            'minority_class': 1
        }
    """
    # Input validation
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if df[target_col].isnull().any():
        raise ValueError("Target variable contains missing values")
    
    # Calculate distributions and counts
    target_counts = df[target_col].value_counts()
    target_dist = target_counts / len(df)
    
    # Identify majority and minority classes
    majority_class = target_counts.idxmax()
    minority_class = target_counts.idxmin()
    
    # Calculate imbalance ratio
    imbalance_ratio = float(target_counts.min()) / float(target_counts.max())
    
    # Prepare results
    results = {
        'distribution': {k: round(float(v), round_digits) 
                        for k, v in target_dist.to_dict().items()},
        'counts': target_counts.to_dict(),
        'imbalance_ratio': round(imbalance_ratio, round_digits),
        'majority_class': majority_class,
        'minority_class': minority_class
    }
    
    # Print summary
    print("\nTarget Distribution Analysis:")
    print(f"✓ Total samples: {len(df)}")
    print(f"✓ Number of classes: {len(target_counts)}")
    print(f"✓ Majority class ({majority_class}): {target_counts.max()} samples "
          f"({target_dist.max():.1%})")
    print(f"✓ Minority class ({minority_class}): {target_counts.min()} samples "
          f"({target_dist.min():.1%})")
    print(f"✓ Imbalance ratio: {imbalance_ratio:.3f}")
    
    return results


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'y',
    figsize: Tuple[int, int] = (10, 6),
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    palette: str = 'viridis',
    rotation: int = 0
) -> None:
    """Creates a visualization of target variable distribution.
    
    Generates a bar plot showing the distribution of classes in the target variable,
    including count and percentage labels.
    
    Args:
        df (pd.DataFrame): The input dataset containing the target variable.
        target_col (str): Name of the target column. Defaults to 'y'.
        figsize (Tuple[int, int]): Figure size as (width, height). Defaults to (10, 6).
        title_fontsize (int): Font size for the title. Defaults to 12.
        label_fontsize (int): Font size for labels. Defaults to 10.
        palette (str): Color palette for the plot. Defaults to 'viridis'.
        rotation (int): Rotation angle for x-axis labels. Defaults to 0.
        
    Raises:
        KeyError: If target_col is not found in the DataFrame.
        ValueError: If DataFrame is empty or target contains invalid values.
        
    Example:
        >>> df = pd.DataFrame({'y': [0, 0, 0, 1, 1]})
        >>> plot_target_distribution(df)
    """
    # Input validation
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Calculate distribution
    target_counts = df[target_col].value_counts()
    target_percentages = (target_counts / len(df) * 100).round(1)
    
    # Create figure and axis
    plt.figure(figsize=figsize)
    
    # Create bar plot
    ax = sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        hue=target_counts.index,  # Add hue parameter
        palette=palette,
        legend=False  # Hide legend since we don't need it
    )
    
    # Add count and percentage labels on bars
    for i, (count, percentage) in enumerate(zip(target_counts, target_percentages)):
        ax.text(
            i, count/2,
            f'n = {count}\n({percentage}%)',
            ha='center',
            va='center',
            fontsize=label_fontsize,
            color='white',
            fontweight='bold'
        )
    
    # Customize plot
    plt.title(
        f'Distribution of {target_col}\n(Total samples: {len(df)})',
        fontsize=title_fontsize,
        pad=20
    )
    plt.xlabel(f'{target_col} Classes', fontsize=label_fontsize)
    plt.ylabel('Count', fontsize=label_fontsize)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=rotation)
    
    # Add grid for better readability
    plt.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()
    

def generate_data_summary(df: pd.DataFrame,
                          target_col: str = 'y'
                         ) -> Dict[str, Union[Tuple, Dict]]:
    """Generates a comprehensive summary of the dataset features and characteristics.
   
    Creates a detailed summary including dataset dimensions, feature types,
    descriptive statistics, target distribution, and missing value analysis.
   
    Args:
        df (pd.DataFrame): The input dataset to analyze.
        target_col (str): Name of the target column. Defaults to 'y'.
   
    Returns:
        Dict[str, Union[Tuple, Dict]]: A dictionary containing:
            - dataset_shape: Tuple of (rows, columns)
            - numeric_features: Dict with:
                - count: Number of numeric features
                - names: List of numeric feature names
                - statistics: Basic statistics (mean, std, min, max, etc.)
            - categorical_features: Dict with:
                - count: Number of categorical features
                - names: List of categorical feature names
                - unique_values: Count of unique values per feature
            - target_distribution: Distribution analysis of target variable
            - missing_values: Analysis of missing values by feature
           
    Raises:
        ValueError: If DataFrame is empty or contains invalid data.
        KeyError: If target_col is not found in DataFrame.
           
    Examples:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'salary': [50000, 60000, 70000],
        ...     'department': ['HR', 'IT', 'HR'],
        ...     'y': [0, 1, 1]
        ... })
        >>> summary = generate_data_summary(df)
        >>> print(f"Dataset shape: {summary['dataset_shape']}")
        >>> print(f"Number of numeric features: {summary['numeric_features']['count']}")
    """
    # Input validation
    if df.empty:
        raise ValueError("DataFrame is empty")
        
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    # Get feature groups
    numeric_features, categorical_features = get_feature_groups(df)
   
    # Generate summary dictionary
    summary = {
       'dataset_shape': df.shape,
       'numeric_features': {
           'count': len(numeric_features),
           'names': numeric_features,
           'statistics': df[numeric_features].describe().round(2).to_dict()
       },
       'categorical_features': {
           'count': len(categorical_features),
           'names': categorical_features,
           'unique_values': {col: df[col].nunique() for col in categorical_features},
           'value_counts': {col: df[col].value_counts().to_dict() 
                          for col in categorical_features}
       },
       'target_distribution': analyze_target_distribution(df, target_col=target_col),
       'missing_values': analyze_missing_values(df).to_dict()
    }
   
    # Print summary overview
    print("\nDataset Summary:")
    print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"✓ Numeric Features: {len(numeric_features)}")
    print(f"✓ Categorical Features: {len(categorical_features)}")
    print(f"✓ Missing Values: {df.isnull().sum().sum()} total")
    print(f"✓ Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

    return summary


from typing import List, Tuple, Union, Type
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import TransformerMixin

def preprocess_numeric_features(df: pd.DataFrame,
                                numeric_features: List[str],
                                scaler_type: Union[Type[TransformerMixin], None] = StandardScaler,
                                copy: bool = True
                               ) -> Tuple[pd.DataFrame, TransformerMixin]:
    """Preprocesses numerical features using specified scaling method.

    Scales numerical features using a specified scaler (StandardScaler by default).
    Supports any sklearn-compatible scaler and preserves the original DataFrame structure.

    Args:
       df (pd.DataFrame): The input dataset containing features to scale.
       numeric_features (List[str]): List of numerical column names to scale.
       scaler_type (Union[Type[TransformerMixin], None], optional): Sklearn scaler class to use.
           Defaults to StandardScaler. Set to None to return data without scaling.
       copy (bool, optional): Whether to create a copy of the input DataFrame.
           Defaults to True to preserve the original data.

    Returns:
       Tuple[pd.DataFrame, TransformerMixin]: A tuple containing:
           - pd.DataFrame: The preprocessed dataset with scaled features
           - TransformerMixin: The fitted scaler object for later use

    Raises:
       ValueError: If numeric_features is empty or contains invalid column names.
       TypeError: If scaler_type is not a valid sklearn transformer.

    Examples:
       >>> from sklearn.preprocessing import MinMaxScaler
       >>> df = pd.DataFrame({
       ...     'age': [25, 30, 35],
       ...     'salary': [50000, 60000, 70000],
       ...     'category': ['A', 'B', 'C']
       ... })
       >>> numeric_cols = ['age', 'salary']
       >>> scaled_df, scaler = preprocess_numeric_features(
       ...     df, 
       ...     numeric_cols,
       ...     scaler_type=MinMaxScaler
       ... )

    Notes:
       - The scaler is fitted on the training data and can be reused for test data
       - Original DataFrame structure and non-numeric columns are preserved
       - Common scalers: StandardScaler, MinMaxScaler, RobustScaler
    """
    # Input validation
    if not numeric_features:
        raise ValueError("numeric_features list cannot be empty")

    if not all(col in df.columns for col in numeric_features):
        missing_cols = [col for col in numeric_features if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if scaler_type is not None and not issubclass(scaler_type, TransformerMixin):
        raise TypeError("scaler_type must be a valid sklearn transformer")

    # Create copy if requested
    df_processed = df.copy() if copy else df

    if scaler_type is not None:
       # Initialize and fit scaler
       scaler = scaler_type()

       # Scale features
       try:
            df_processed[numeric_features] = scaler.fit_transform(
               df[numeric_features]
           )
            
            # Print scaling summary
            print("\nFeature Scaling Summary:")
            print(f"✓ Scaler: {scaler_type.__name__}")
            print(f"✓ Features scaled: {len(numeric_features)}")

            if isinstance(scaler, StandardScaler):
                print("\nScaling Parameters:")
                print("- Mean:", dict(zip(numeric_features, 
                                       scaler.mean_.round(4))))
                print("- Std:", dict(zip(numeric_features, 
                                      scaler.scale_.round(4))))
            elif isinstance(scaler, MinMaxScaler):
                print("\nScaling Parameters:")
                print("- Min:", dict(zip(numeric_features, 
                                      scaler.data_min_.round(4))))
                print("- Max:", dict(zip(numeric_features, 
                                      scaler.data_max_.round(4))))

    except Exception as e: 
           raise ValueError(f"Error during scaling: {str(e)}")
    else:
        scaler = None
        print("\n✓ No scaling applied - returning original features")

    return df_processed, scaler

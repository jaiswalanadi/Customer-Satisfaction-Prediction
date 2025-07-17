"""
Utility functions for Customer Satisfaction Prediction project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load CSV data file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        logger.info(f"Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        file_path: Path to save file
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def save_model(model: Any, file_path: Path) -> None:
    """
    Save model using joblib
    
    Args:
        model: Model to save
        file_path: Path to save model
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def load_model(file_path: Path) -> Any:
    """
    Load model using joblib
    
    Args:
        file_path: Path to model file
        
    Returns:
        Loaded model
    """
    try:
        model = joblib.load(file_path)
        logger.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive information about the dataset
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_summary': df.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Get categorical column summaries
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        info['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return info

def plot_target_distribution(df: pd.DataFrame, target_col: str, save_path: Path = None) -> None:
    """
    Plot target variable distribution
    
    Args:
        df: DataFrame containing target variable
        target_col: Target column name
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Filter out null values
    target_data = df[target_col].dropna()
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(target_data, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title(f'Distribution of {target_col}')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Count plot
    target_counts = target_data.value_counts().sort_index()
    ax2.bar(target_counts.index, target_counts.values, color='lightgreen', alpha=0.7)
    ax2.set_title(f'Count of {target_col} by Rating')
    ax2.set_xlabel('Rating')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Plot correlation matrix for numeric features
    
    Args:
        df: DataFrame with numeric features
        save_path: Optional path to save plot
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation matrix")
        return
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    
    plt.show()

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
    """
    Evaluate model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return results

def plot_confusion_matrix(conf_matrix: np.ndarray, model_name: str, save_path: Path = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        model_name: Name of the model
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray, 
                          model_name: str, save_path: Path = None, top_n: int = 15) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: Feature importance scores
        model_name: Name of the model
        save_path: Optional path to save plot
        top_n: Number of top features to show
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def create_model_comparison_plot(results: List[Dict], save_path: Path = None) -> None:
    """
    Create model comparison plot
    
    Args:
        results: List of model evaluation results
        save_path: Optional path to save plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = [result['model_name'] for result in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [result[metric] for result in results]
        axes[i].bar(model_names, values, color=['skyblue', 'lightgreen', 'lightcoral'][:len(model_names)])
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, value in enumerate(values):
            axes[i].text(j, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.show()

def print_data_summary(df: pd.DataFrame, title: str = "Dataset Summary") -> None:
    """
    Print comprehensive data summary
    
    Args:
        df: DataFrame to summarize
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    print(missing_summary[missing_summary['Missing Count'] > 0])
    
    print(f"\nNumeric Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        print(df[numeric_cols].describe())
    
    print(f"\nCategorical Summary:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("Available functions:")
    print("- load_data, save_data, save_model, load_model")
    print("- get_data_info, plot_target_distribution, plot_correlation_matrix")
    print("- evaluate_model, plot_confusion_matrix, plot_feature_importance")
    print("- create_model_comparison_plot, print_data_summary")

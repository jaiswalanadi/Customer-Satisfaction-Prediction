"""
Configuration settings for Customer Satisfaction Prediction project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "customer_support_tickets.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_data.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "feature_engineered_data.csv"

# Model files
RF_MODEL_FILE = MODELS_DIR / "random_forest_model.pkl"
XGB_MODEL_FILE = MODELS_DIR / "xgboost_model.pkl"
LR_MODEL_FILE = MODELS_DIR / "logistic_regression_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "encoder.pkl"

# Model parameters
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering parameters
TARGET_COLUMN = "Customer Satisfaction Rating"
ID_COLUMN = "Ticket ID"

# Categorical features
CATEGORICAL_FEATURES = [
    "Customer Gender",
    "Product Purchased", 
    "Ticket Type",
    "Ticket Subject",
    "Ticket Status",
    "Ticket Priority",
    "Ticket Channel"
]

# Numerical features
NUMERICAL_FEATURES = [
    "Customer Age"
]

# Text features for NLP
TEXT_FEATURES = [
    "Ticket Description"
]

# Features to drop
DROP_FEATURES = [
    "Ticket ID",
    "Customer Name", 
    "Customer Email",
    "Date of Purchase",
    "Resolution",
    "First Response Time",
    "Time to Resolution"
]

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': MODEL_RANDOM_STATE
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': MODEL_RANDOM_STATE
}

LR_PARAMS = {
    'random_state': MODEL_RANDOM_STATE,
    'max_iter': 1000
}

# Flask app settings
FLASK_DEBUG = True
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories if they don't exist
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
        REPORTS_DIR, FIGURES_DIR, FIGURES_DIR / "eda_plots",
        FIGURES_DIR / "model_performance", FIGURES_DIR / "feature_importance"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Configuration loaded and directories created successfully!")

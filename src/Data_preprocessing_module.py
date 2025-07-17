"""
Data preprocessing module for Customer Satisfaction Prediction
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.utils import load_data, save_data, save_model, print_data_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for customer satisfaction prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = TARGET_COLUMN
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        
        Returns:
            Raw DataFrame
        """
        logger.info("Loading raw data...")
        df = load_data(RAW_DATA_FILE)
        print_data_summary(df, "Raw Dataset Summary")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and inconsistencies
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Log initial state
        logger.info(f"Initial dataset shape: {df_clean.shape}")
        
        # Remove records without satisfaction rating (our target variable)
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[self.target_column])
        logger.info(f"Removed {initial_count - len(df_clean)} records without satisfaction rating")
        
        # Handle missing values in other columns
        missing_before = df_clean.isnull().sum().sum()
        
        # Fill missing categorical values with 'Unknown'
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != self.target_column:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Fill missing numerical values with median
        numerical_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            if col != self.target_column:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values reduced from {missing_before} to {missing_after}")
        
        # Remove duplicates based on Ticket ID
        duplicates = df_clean.duplicated(subset=[ID_COLUMN])
        df_clean = df_clean.drop_duplicates(subset=[ID_COLUMN])
        logger.info(f"Removed {duplicates.sum()} duplicate records")
        
        # Validate target variable range (should be 1-5)
        invalid_ratings = ~df_clean[self.target_column].isin([1, 2, 3, 4, 5])
        if invalid_ratings.any():
            logger.warning(f"Found {invalid_ratings.sum()} invalid satisfaction ratings")
            df_clean = df_clean[~invalid_ratings]
        
        # Clean customer age (remove outliers)
        age_col = 'Customer Age'
        q1 = df_clean[age_col].quantile(0.01)
        q99 = df_clean[age_col].quantile(0.99)
        age_outliers = (df_clean[age_col] < q1) | (df_clean[age_col] > q99)
        df_clean = df_clean[~age_outliers]
        logger.info(f"Removed {age_outliers.sum()} age outliers")
        
        # Clean text data - remove excessive whitespace and special characters
        text_columns = ['Ticket Description', 'Ticket Subject']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].str.strip()
                df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        print_data_summary(df_clean, "Cleaned Dataset Summary")
        
        return df_clean
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing data
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Creating derived features...")
        
        df_features = df.copy()
        
        # Age groups
        df_features['Age_Group'] = pd.cut(
            df_features['Customer Age'], 
            bins=[0, 25, 35, 45, 55, 100], 
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        # Product category mapping
        product_categories = {
            'Software': ['Microsoft Office', 'Autodesk AutoCAD', 'Adobe Photoshop'],
            'Electronics': ['GoPro Hero', 'Canon EOS', 'Sony Xperia', 'Apple AirPods'],
            'Home_Appliances': ['LG Smart TV', 'Nest Thermostat', 'Roomba Robot Vacuum'],
            'Computing': ['Dell XPS', 'Google Pixel', 'Amazon Echo'],
            'Entertainment': ['LG OLED', 'Philips Hue Lights']
        }
        
        def categorize_product(product):
            for category, products in product_categories.items():
                if any(p in str(product) for p in products):
                    return category
            return 'Other'
        
        df_features['Product_Category'] = df_features['Product Purchased'].apply(categorize_product)
        
        # Text features from description
        if 'Ticket Description' in df_features.columns:
            df_features['Description_Length'] = df_features['Ticket Description'].str.len()
            df_features['Description_Word_Count'] = df_features['Ticket Description'].str.split().str.len()
            
            # Sentiment keywords
            negative_keywords = ['issue', 'problem', 'error', 'not working', 'broken', 'failed']
            positive_keywords = ['good', 'excellent', 'satisfied', 'working', 'resolved']
            
            df_features['Negative_Keywords_Count'] = df_features['Ticket Description'].str.lower().apply(
                lambda x: sum(keyword in str(x) for keyword in negative_keywords)
            )
            df_features['Positive_Keywords_Count'] = df_features['Ticket Description'].str.lower().apply(
                lambda x: sum(keyword in str(x) for keyword in positive_keywords)
            )
        
        # Priority score mapping
        priority_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df_features['Priority_Score'] = df_features['Ticket Priority'].map(priority_scores)
        
        # Channel preference score
        channel_scores = {'Email': 1, 'Phone': 2, 'Chat': 3, 'Social media': 4}
        df_features['Channel_Score'] = df_features['Ticket Channel'].map(channel_scores)
        
        logger.info(f"Created derived features. New shape: {df_features.shape}")
        
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Define columns to encode
        categorical_columns = [
            'Customer Gender', 'Product Purchased', 'Ticket Type', 
            'Ticket Subject', 'Ticket Priority', 'Ticket Channel',
            'Product_Category', 'Age_Group'
        ]
        
        # Label encode categorical features
        for col in categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
        
        logger.info("Categorical encoding completed")
        
        return df_encoded
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing features and target...")
        
        # Define feature columns
        feature_columns = [
            'Customer Age', 'Priority_Score', 'Channel_Score',
            'Description_Length', 'Description_Word_Count',
            'Negative_Keywords_Count', 'Positive_Keywords_Count',
            'Customer Gender_Encoded', 'Product Purchased_Encoded',
            'Ticket Type_Encoded', 'Ticket Subject_Encoded',
            'Ticket Priority_Encoded', 'Ticket Channel_Encoded',
            'Product_Category_Encoded', 'Age_Group_Encoded'
        ]
        
        # Filter to only existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        # Prepare features
        X = df[available_features].copy()
        
        # Prepare target
        y = df[self.target_column].copy()
        
        logger.info(f"Features prepared: {X.shape}")
        logger.info(f"Target prepared: {y.shape}")
        logger.info(f"Feature columns: {available_features}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        logger.info("Scaling features...")
        
        # Identify numerical columns
        numerical_cols = [
            'Customer Age', 'Priority_Score', 'Channel_Score',
            'Description_Length', 'Description_Word_Count',
            'Negative_Keywords_Count', 'Positive_Keywords_Count'
        ]
        
        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in X_train.columns]
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if numerical_cols:
            # Fit scaler on training data
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=MODEL_RANDOM_STATE, 
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Target distribution in training set:")
        logger.info(y_train.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, df_clean: pd.DataFrame) -> None:
        """
        Save processed data and models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            df_clean: Cleaned full dataset
        """
        logger.info("Saving processed data...")
        
        # Save cleaned dataset
        save_data(df_clean, CLEANED_DATA_FILE)
        
        # Combine and save train/test data
        train_data = X_train.copy()
        train_data[self.target_column] = y_train
        save_data(train_data, TRAIN_DATA_FILE)
        
        test_data = X_test.copy()
        test_data[self.target_column] = y_test
        save_data(test_data, TEST_DATA_FILE)
        
        # Save preprocessing objects
        save_model(self.scaler, SCALER_FILE)
        save_model(self.label_encoders, ENCODER_FILE)
        
        logger.info("All processed data saved successfully")
    
    def run_preprocessing_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the complete preprocessing pipeline
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load and clean data
        df_raw = self.load_raw_data()
        df_clean = self.clean_data(df_raw)
        
        # Create derived features
        df_features = self.create_derived_features(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save processed data
        self.save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, df_clean)
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    """
    Main function to run data preprocessing
    """
    try:
        # Create directories
        create_directories()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.run_preprocessing_pipeline()
        
        print("\n" + "="*50)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")
        print(f"Feature columns: {len(preprocessor.feature_columns)}")
        print("\nFiles saved:")
        print(f"- Cleaned data: {CLEANED_DATA_FILE}")
        print(f"- Training data: {TRAIN_DATA_FILE}")
        print(f"- Test data: {TEST_DATA_FILE}")
        print(f"- Scaler: {SCALER_FILE}")
        print(f"- Encoders: {ENCODER_FILE}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

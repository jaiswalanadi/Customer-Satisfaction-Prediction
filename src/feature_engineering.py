"""
Feature engineering module for Customer Satisfaction Prediction
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.utils import load_data, save_data, save_model, load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for creating advanced features
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.pca_transformer = PCA(n_components=5)
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Load cleaned data from preprocessing step
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Loading cleaned data...")
        df = load_data(CLEANED_DATA_FILE)
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced text features from ticket descriptions
        
        Args:
            df: DataFrame with text columns
            
        Returns:
            DataFrame with text features
        """
        logger.info("Creating text features...")
        
        df_text = df.copy()
        
        if 'Ticket Description' in df_text.columns:
            descriptions = df_text['Ticket Description'].fillna('').astype(str)
            
            # Basic text statistics
            df_text['Text_Length'] = descriptions.str.len()
            df_text['Word_Count'] = descriptions.str.split().str.len()
            df_text['Sentence_Count'] = descriptions.str.count(r'[.!?]') + 1
            df_text['Avg_Word_Length'] = descriptions.apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            
            # Punctuation features
            df_text['Exclamation_Count'] = descriptions.str.count('!')
            df_text['Question_Count'] = descriptions.str.count(r'\?')
            df_text['Uppercase_Ratio'] = descriptions.apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            )
            
            # Sentiment-related features
            urgency_keywords = ['urgent', 'immediate', 'asap', 'emergency', 'critical']
            frustration_keywords = ['frustrated', 'annoyed', 'disappointed', 'angry']
            technical_keywords = ['error', 'bug', 'crash', 'freeze', 'malfunction']
            
            df_text['Urgency_Score'] = descriptions.str.lower().apply(
                lambda x: sum(keyword in x for keyword in urgency_keywords)
            )
            df_text['Frustration_Score'] = descriptions.str.lower().apply(
                lambda x: sum(keyword in x for keyword in frustration_keywords)
            )
            df_text['Technical_Score'] = descriptions.str.lower().apply(
                lambda x: sum(keyword in x for keyword in technical_keywords)
            )
            
            # TF-IDF features (only for records with ratings)
            rated_descriptions = descriptions[df_text[TARGET_COLUMN].notna()]
            if not rated_descriptions.empty:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(rated_descriptions)
                
                # Create TF-IDF feature names
                tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                
                # Create DataFrame with TF-IDF features
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=tfidf_feature_names,
                    index=rated_descriptions.index
                )
                
                # Add TF-IDF features to main DataFrame
                df_text = df_text.join(tfidf_df, how='left')
                
                # Fill NaN values for non-rated records
                df_text[tfidf_feature_names] = df_text[tfidf_feature_names].fillna(0)
                
                logger.info(f"Created {len(tfidf_feature_names)} TF-IDF features")
            
            logger.info("Text features created successfully")
        
        return df_text
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from date columns
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features...")
        
        df_temporal = df.copy()
        
        if 'Date of Purchase' in df_temporal.columns:
            # Convert to datetime
            df_temporal['Purchase_Date'] = pd.to_datetime(df_temporal['Date of Purchase'])
            
            # Extract temporal components
            df_temporal['Purchase_Year'] = df_temporal['Purchase_Date'].dt.year
            df_temporal['Purchase_Month'] = df_temporal['Purchase_Date'].dt.month
            df_temporal['Purchase_Day'] = df_temporal['Purchase_Date'].dt.day
            df_temporal['Purchase_Weekday'] = df_temporal['Purchase_Date'].dt.dayofweek
            df_temporal['Purchase_Quarter'] = df_temporal['Purchase_Date'].dt.quarter
            
            # Create season feature
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'
            
            df_temporal['Purchase_Season'] = df_temporal['Purchase_Month'].apply(get_season)
            
            # Days since purchase (assuming current date is max date in dataset)
            max_date = df_temporal['Purchase_Date'].max()
            df_temporal['Days_Since_Purchase'] = (max_date - df_temporal['Purchase_Date']).dt.days
            
            logger.info("Temporal features created successfully")
        
        return df_temporal
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_interact = df.copy()
        
        # Age and priority interaction
        if 'Customer Age' in df_interact.columns and 'Priority_Score' in df_interact.columns:
            df_interact['Age_Priority_Interaction'] = df_interact['Customer Age'] * df_interact['Priority_Score']
        
        # Text length and urgency interaction
        if 'Text_Length' in df_interact.columns and 'Urgency_Score' in df_interact.columns:
            df_interact['Length_Urgency_Interaction'] = df_interact['Text_Length'] * df_interact['Urgency_Score']
        
        # Channel and priority interaction
        if 'Channel_Score' in df_interact.columns and 'Priority_Score' in df_interact.columns:
            df_interact['Channel_Priority_Interaction'] = df_interact['Channel_Score'] * df_interact['Priority_Score']
        
        # Technical score and frustration interaction
        if 'Technical_Score' in df_interact.columns and 'Frustration_Score' in df_interact.columns:
            df_interact['Technical_Frustration_Interaction'] = df_interact['Technical_Score'] * df_interact['Frustration_Score']
        
        logger.info("Interaction features created successfully")
        
        return df_interact
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features based on groupings
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with aggregated features
        """
        logger.info("Creating aggregated features...")
        
        df_agg = df.copy()
        
        # Only work with records that have satisfaction ratings
        rated_df = df_agg[df_agg[TARGET_COLUMN].notna()]
        
        if not rated_df.empty:
            # Product-based aggregations
            if 'Product Purchased' in rated_df.columns:
                product_stats = rated_df.groupby('Product Purchased')[TARGET_COLUMN].agg([
                    'mean', 'std', 'count'
                ]).add_prefix('Product_')
                
                df_agg = df_agg.merge(product_stats, on='Product Purchased', how='left')
                df_agg['Product_std'] = df_agg['Product_std'].fillna(0)
            
            # Ticket type aggregations
            if 'Ticket Type' in rated_df.columns:
                ticket_stats = rated_df.groupby('Ticket Type')[TARGET_COLUMN].agg([
                    'mean', 'std', 'count'
                ]).add_prefix('TicketType_')
                
                df_agg = df_agg.merge(ticket_stats, on='Ticket Type', how='left')
                df_agg['TicketType_std'] = df_agg['TicketType_std'].fillna(0)
            
            # Channel aggregations
            if 'Ticket Channel' in rated_df.columns:
                channel_stats = rated_df.groupby('Ticket Channel')[TARGET_COLUMN].agg([
                    'mean', 'std', 'count'
                ]).add_prefix('Channel_')
                
                df_agg = df_agg.merge(channel_stats, on='Ticket Channel', how='left')
                df_agg['Channel_std'] = df_agg['Channel_std'].fillna(0)
            
            # Age group aggregations
            if 'Age_Group' in rated_df.columns:
                age_stats = rated_df.groupby('Age_Group')[TARGET_COLUMN].agg([
                    'mean', 'std', 'count'
                ]).add_prefix('AgeGroup_')
                
                df_agg = df_agg.merge(age_stats, on='Age_Group', how='left')
                df_agg['AgeGroup_std'] = df_agg['AgeGroup_std'].fillna(0)
            
            logger.info("Aggregated features created successfully")
        
        return df_agg
    
    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned versions of continuous features
        
        Args:
            df: DataFrame with continuous features
            
        Returns:
            DataFrame with binned features
        """
        logger.info("Creating binned features...")
        
        df_binned = df.copy()
        
        # Bin text length
        if 'Text_Length' in df_binned.columns:
            df_binned['Text_Length_Bin'] = pd.cut(
                df_binned['Text_Length'],
                bins=[0, 50, 100, 200, 500, np.inf],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
        
        # Bin word count
        if 'Word_Count' in df_binned.columns:
            df_binned['Word_Count_Bin'] = pd.cut(
                df_binned['Word_Count'],
                bins=[0, 10, 20, 50, 100, np.inf],
                labels=['Few', 'Some', 'Many', 'Lot', 'Excessive']
            )
        
        # Bin days since purchase
        if 'Days_Since_Purchase' in df_binned.columns:
            df_binned['Purchase_Recency_Bin'] = pd.cut(
                df_binned['Days_Since_Purchase'],
                bins=[0, 30, 90, 180, 365, np.inf],
                labels=['Very_Recent', 'Recent', 'Moderate', 'Old', 'Very_Old']
            )
        
        logger.info("Binned features created successfully")
        
        return df_binned
    
    def reduce_dimensionality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dimensionality reduction to high-dimensional features
        
        Args:
            df: DataFrame with high-dimensional features
            
        Returns:
            DataFrame with reduced features
        """
        logger.info("Applying dimensionality reduction...")
        
        df_reduced = df.copy()
        
        # Get TF-IDF columns
        tfidf_cols = [col for col in df_reduced.columns if col.startswith('tfidf_')]
        
        if len(tfidf_cols) > 10:  # Only apply if we have many TF-IDF features
            # Get only records with ratings for fitting
            rated_mask = df_reduced[TARGET_COLUMN].notna()
            
            if rated_mask.sum() > 0:
                # Fit PCA on rated records
                tfidf_data = df_reduced.loc[rated_mask, tfidf_cols].fillna(0)
                
                if not tfidf_data.empty:
                    pca_features = self.pca_transformer.fit_transform(tfidf_data)
                    
                    # Create PCA feature names
                    pca_feature_names = [f'pca_text_{i}' for i in range(pca_features.shape[1])]
                    
                    # Create PCA DataFrame
                    pca_df = pd.DataFrame(
                        pca_features,
                        columns=pca_feature_names,
                        index=tfidf_data.index
                    )
                    
                    # Add PCA features to main DataFrame
                    df_reduced = df_reduced.join(pca_df, how='left')
                    
                    # Fill NaN values for non-rated records
                    df_reduced[pca_feature_names] = df_reduced[pca_feature_names].fillna(0)
                    
                    # Remove original TF-IDF features to reduce dimensionality
                    df_reduced = df_reduced.drop(columns=tfidf_cols)
                    
                    logger.info(f"Reduced {len(tfidf_cols)} TF-IDF features to {len(pca_feature_names)} PCA features")
        
        return df_reduced
    
    def encode_new_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode newly created categorical features
        
        Args:
            df: DataFrame with new categorical features
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding new categorical features...")
        
        df_encoded = df.copy()
        
        # Load existing encoders
        try:
            encoders = load_model(ENCODER_FILE)
        except:
            encoders = {}
        
        # New categorical features to encode
        new_categorical_features = [
            'Purchase_Season', 'Text_Length_Bin', 'Word_Count_Bin', 'Purchase_Recency_Bin'
        ]
        
        for col in new_categorical_features:
            if col in df_encoded.columns:
                # Convert to string to handle any NaN values
                df_encoded[col] = df_encoded[col].astype(str)
                
                # Create new encoder if not exists
                if col not in encoders:
                    from sklearn.preprocessing import LabelEncoder
                    encoders[col] = LabelEncoder()
                
                # Encode the feature
                df_encoded[col + '_Encoded'] = encoders[col].fit_transform(df_encoded[col])
                logger.info(f"Encoded {col}: {len(encoders[col].classes_)} unique values")
        
        # Save updated encoders
        save_model(encoders, ENCODER_FILE)
        
        return df_encoded
    
    def select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select final features for modeling
        
        Args:
            df: DataFrame with all engineered features
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting final features...")
        
        # Define feature groups
        basic_features = [
            'Customer Age', 'Priority_Score', 'Channel_Score'
        ]
        
        text_features = [
            'Text_Length', 'Word_Count', 'Sentence_Count', 'Avg_Word_Length',
            'Exclamation_Count', 'Question_Count', 'Uppercase_Ratio',
            'Urgency_Score', 'Frustration_Score', 'Technical_Score'
        ]
        
        temporal_features = [
            'Purchase_Year', 'Purchase_Month', 'Purchase_Day', 'Purchase_Weekday',
            'Purchase_Quarter', 'Days_Since_Purchase'
        ]
        
        interaction_features = [
            'Age_Priority_Interaction', 'Length_Urgency_Interaction',
            'Channel_Priority_Interaction', 'Technical_Frustration_Interaction'
        ]
        
        aggregated_features = [
            col for col in df.columns if any(prefix in col for prefix in [
                'Product_', 'TicketType_', 'Channel_', 'AgeGroup_'
            ])
        ]
        
        encoded_features = [
            col for col in df.columns if col.endswith('_Encoded')
        ]
        
        pca_features = [
            col for col in df.columns if col.startswith('pca_')
        ]
        
        # Combine all feature groups
        all_features = (basic_features + text_features + temporal_features + 
                       interaction_features + aggregated_features + encoded_features + pca_features)
        
        # Filter to only existing columns
        final_features = [col for col in all_features if col in df.columns]
        
        # Add target column
        if TARGET_COLUMN in df.columns:
            final_features.append(TARGET_COLUMN)
        
        df_final = df[final_features].copy()
        
        logger.info(f"Selected {len(final_features)-1} features for modeling")
        logger.info(f"Feature categories:")
        logger.info(f"  - Basic: {len([f for f in basic_features if f in df_final.columns])}")
        logger.info(f"  - Text: {len([f for f in text_features if f in df_final.columns])}")
        logger.info(f"  - Temporal: {len([f for f in temporal_features if f in df_final.columns])}")
        logger.info(f"  - Interaction: {len([f for f in interaction_features if f in df_final.columns])}")
        logger.info(f"  - Aggregated: {len([f for f in aggregated_features if f in df_final.columns])}")
        logger.info(f"  - Encoded: {len([f for f in encoded_features if f in df_final.columns])}")
        logger.info(f"  - PCA: {len([f for f in pca_features if f in df_final.columns])}")
        
        return df_final
    
    def run_feature_engineering_pipeline(self) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        
        # Create text features
        df = self.create_text_features(df)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create aggregated features
        df = self.create_aggregated_features(df)
        
        # Create binned features
        df = self.create_binned_features(df)
        
        # Apply dimensionality reduction
        df = self.reduce_dimensionality(df)
        
        # Encode new categorical features
        df = self.encode_new_categorical_features(df)
        
        # Select final features
        df_final = self.select_final_features(df)
        
        # Save feature engineered data
        save_data(df_final, FEATURE_DATA_FILE)
        
        # Save feature engineering objects
        save_model(self.tfidf_vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
        save_model(self.pca_transformer, MODELS_DIR / "pca_transformer.pkl")
        
        logger.info("Feature engineering pipeline completed successfully!")
        
        return df_final

def main():
    """
    Main function to run feature engineering
    """
    try:
        # Create directories
        create_directories()
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Run feature engineering pipeline
        df_engineered = feature_engineer.run_feature_engineering_pipeline()
        
        print("\n" + "="*50)
        print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final dataset shape: {df_engineered.shape}")
        print(f"Features created: {df_engineered.shape[1] - 1}")  # -1 for target column
        
        # Display feature statistics
        print("\nFeature Statistics:")
        print(f"- Records with satisfaction ratings: {df_engineered[TARGET_COLUMN].notna().sum()}")
        print(f"- Total features: {len(df_engineered.columns) - 1}")
        print(f"- Missing values: {df_engineered.isnull().sum().sum()}")
        
        print(f"\nFile saved: {FEATURE_DATA_FILE}")
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

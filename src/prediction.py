"""
Prediction module for Customer Satisfaction Prediction
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.utils import load_model, load_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatisfactionPredictor:
    """
    Customer satisfaction prediction class
    """
    
    def __init__(self, model_name: str = 'Random Forest'):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_columns = None
        self.load_model_and_preprocessors()
    
    def load_model_and_preprocessors(self) -> None:
        """
        Load the trained model and preprocessing objects
        """
        logger.info(f"Loading {self.model_name} model and preprocessors...")
        
        # Load model based on name
        try:
            if self.model_name == 'Random Forest':
                self.model = load_model(RF_MODEL_FILE)
            elif self.model_name == 'XGBoost':
                self.model = load_model(XGB_MODEL_FILE)
            elif self.model_name == 'Logistic Regression':
                self.model = load_model(LR_MODEL_FILE)
            elif self.model_name == 'Ensemble':
                self.model = load_model(MODELS_DIR / "ensemble_model.pkl")
            else:
                raise ValueError(f"Unknown model name: {self.model_name}")
            
            logger.info(f"Successfully loaded {self.model_name} model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load preprocessing objects
        try:
            self.scaler = load_model(SCALER_FILE)
            self.encoders = load_model(ENCODER_FILE)
            logger.info("Successfully loaded preprocessing objects")
        except Exception as e:
            logger.error(f"Error loading preprocessing objects: {e}")
            raise
        
        # Get feature columns from training data
        try:
            train_data = load_data(TRAIN_DATA_FILE)
            self.feature_columns = [col for col in train_data.columns if col != TARGET_COLUMN]
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        except Exception as e:
            logger.error(f"Error loading feature columns: {e}")
            raise
    
    def preprocess_single_record(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess a single record for prediction
        
        Args:
            record: Dictionary containing customer and ticket information
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        logger.info("Preprocessing single record...")
        
        # Create DataFrame from record
        df = pd.DataFrame([record])
        
        # Apply feature engineering similar to training
        df = self._create_derived_features(df)
        df = self._encode_categorical_features(df)
        df = self._create_text_features(df)
        df = self._create_temporal_features(df)
        df = self._create_interaction_features(df)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select and order features
        df = df[self.feature_columns]
        
        # Apply scaling to numerical features
        numerical_cols = [
            'Customer Age', 'Priority_Score', 'Channel_Score',
            'Text_Length', 'Word_Count', 'Sentence_Count', 'Avg_Word_Length',
            'Exclamation_Count', 'Question_Count', 'Uppercase_Ratio',
            'Urgency_Score', 'Frustration_Score', 'Technical_Score'
        ]
        
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if numerical_cols and self.scaler:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        logger.info("Single record preprocessing completed")
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features (simplified version for prediction)"""
        df = df.copy()
        
        # Age groups
        if 'Customer Age' in df.columns:
            df['Age_Group'] = pd.cut(
                df['Customer Age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['18-25', '26-35', '36-45', '46-55', '55+']
            )
        
        # Priority score mapping
        if 'Ticket Priority' in df.columns:
            priority_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            df['Priority_Score'] = df['Ticket Priority'].map(priority_scores).fillna(2)
        
        # Channel preference score
        if 'Ticket Channel' in df.columns:
            channel_scores = {'Email': 1, 'Phone': 2, 'Chat': 3, 'Social media': 4}
            df['Channel_Score'] = df['Ticket Channel'].map(channel_scores).fillna(2)
        
        # Product category
        if 'Product Purchased' in df.columns:
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
            
            df['Product_Category'] = df['Product Purchased'].apply(categorize_product)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using pre-trained encoders"""
        df = df.copy()
        
        categorical_columns = [
            'Customer Gender', 'Product Purchased', 'Ticket Type', 
            'Ticket Subject', 'Ticket Priority', 'Ticket Channel',
            'Product_Category', 'Age_Group'
        ]
        
        for col in categorical_columns:
            if col in df.columns and col in self.encoders:
                try:
                    # Handle unseen categories
                    encoder = self.encoders[col]
                    df[col] = df[col].astype(str)
                    
                    # Transform only if value is in encoder classes
                    df[col + '_Encoded'] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0
                    )
                except Exception as e:
                    logger.warning(f"Error encoding {col}: {e}")
                    df[col + '_Encoded'] = 0
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text features from ticket description"""
        df = df.copy()
        
        if 'Ticket Description' in df.columns:
            descriptions = df['Ticket Description'].fillna('').astype(str)
            
            # Basic text statistics
            df['Text_Length'] = descriptions.str.len()
            df['Word_Count'] = descriptions.str.split().str.len()
            df['Sentence_Count'] = descriptions.str.count(r'[.!?]') + 1
            df['Avg_Word_Length'] = descriptions.apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            
            # Punctuation features
            df['Exclamation_Count'] = descriptions.str.count('!')
            df['Question_Count'] = descriptions.str.count(r'\?')
            df['Uppercase_Ratio'] = descriptions.apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            )
            
            # Sentiment-related features
            urgency_keywords = ['urgent', 'immediate', 'asap', 'emergency', 'critical']
            frustration_keywords = ['frustrated', 'annoyed', 'disappointed', 'angry']
            technical_keywords = ['error', 'bug', 'crash', 'freeze', 'malfunction']
            
            df['Urgency_Score'] = descriptions.str.lower().apply(
                lambda x: sum(keyword in x for keyword in urgency_keywords)
            )
            df['Frustration_Score'] = descriptions.str.lower().apply(
                lambda x: sum(keyword in x for keyword in frustration_keywords)
            )
            df['Technical_Score'] = descriptions.str.lower().apply(
                lambda x: sum(keyword in x for keyword in technical_keywords)
            )
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        df = df.copy()
        
        if 'Date of Purchase' in df.columns:
            df['Purchase_Date'] = pd.to_datetime(df['Date of Purchase'])
            df['Purchase_Year'] = df['Purchase_Date'].dt.year
            df['Purchase_Month'] = df['Purchase_Date'].dt.month
            df['Purchase_Day'] = df['Purchase_Date'].dt.day
            df['Purchase_Weekday'] = df['Purchase_Date'].dt.dayofweek
            df['Purchase_Quarter'] = df['Purchase_Date'].dt.quarter
            
            # Days since purchase (using current date)
            current_date = pd.Timestamp.now()
            df['Days_Since_Purchase'] = (current_date - df['Purchase_Date']).dt.days
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        df = df.copy()
        
        # Age and priority interaction
        if 'Customer Age' in df.columns and 'Priority_Score' in df.columns:
            df['Age_Priority_Interaction'] = df['Customer Age'] * df['Priority_Score']
        
        # Text length and urgency interaction
        if 'Text_Length' in df.columns and 'Urgency_Score' in df.columns:
            df['Length_Urgency_Interaction'] = df['Text_Length'] * df['Urgency_Score']
        
        # Channel and priority interaction
        if 'Channel_Score' in df.columns and 'Priority_Score' in df.columns:
            df['Channel_Priority_Interaction'] = df['Channel_Score'] * df['Priority_Score']
        
        # Technical score and frustration interaction
        if 'Technical_Score' in df.columns and 'Frustration_Score' in df.columns:
            df['Technical_Frustration_Interaction'] = df['Technical_Score'] * df['Frustration_Score']
        
        return df
    
    def predict_satisfaction(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict customer satisfaction for a single record
        
        Args:
            record: Dictionary containing customer and ticket information
            
        Returns:
            Dictionary with prediction results
        """
        logger.info("Making satisfaction prediction...")
        
        try:
            # Preprocess the record
            processed_df = self.preprocess_single_record(record)
            
            # Make prediction
            prediction = self.model.predict(processed_df)[0]
            prediction_proba = self.model.predict_proba(processed_df)[0]
            
            # Create confidence score
            confidence = np.max(prediction_proba)
            
            # Create result dictionary
            result = {
                'predicted_satisfaction': int(prediction),
                'confidence_score': float(confidence),
                'probability_distribution': {
                    f'rating_{i+1}': float(prob) for i, prob in enumerate(prediction_proba)
                },
                'model_used': self.model_name,
                'prediction_details': {
                    'most_likely_rating': int(prediction),
                    'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low',
                    'probability_percentages': {
                        f'rating_{i+1}': f"{prob*100:.1f}%" for i, prob in enumerate(prediction_proba)
                    }
                }
            }
            
            logger.info(f"Prediction completed: Rating {prediction} (Confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict satisfaction for multiple records
        
        Args:
            records: List of dictionaries containing customer and ticket information
            
        Returns:
            List of prediction results
        """
        logger.info(f"Making batch predictions for {len(records)} records...")
        
        results = []
        for i, record in enumerate(records):
            try:
                result = self.predict_satisfaction(record)
                result['record_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting record {i}: {e}")
                results.append({
                    'record_index': i,
                    'error': str(e),
                    'predicted_satisfaction': None
                })
        
        logger.info(f"Batch prediction completed: {len(results)} results")
        
        return results
    
    def get_prediction_explanation(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get explanation for the prediction
        
        Args:
            record: Dictionary containing customer and ticket information
            
        Returns:
            Dictionary with prediction explanation
        """
        logger.info("Generating prediction explanation...")
        
        try:
            # Get prediction
            prediction_result = self.predict_satisfaction(record)
            
            # Load feature importance
            feature_importance = load_model(MODELS_DIR / "feature_importance.pkl")
            
            # Get top features for this model
            if self.model_name in feature_importance:
                model_importance = feature_importance[self.model_name]
                top_features = sorted(
                    zip(model_importance['features'], model_importance['importance']),
                    key=lambda x: x[1], reverse=True
                )[:10]
            else:
                top_features = []
            
            # Process the record to get feature values
            processed_df = self.preprocess_single_record(record)
            
            # Create explanation
            explanation = {
                'prediction': prediction_result,
                'top_influential_features': [
                    {
                        'feature': feature,
                        'importance': float(importance),
                        'value': float(processed_df[feature].iloc[0]) if feature in processed_df.columns else 'N/A'
                    }
                    for feature, importance in top_features
                ],
                'input_analysis': {
                    'customer_age': record.get('Customer Age', 'N/A'),
                    'ticket_priority': record.get('Ticket Priority', 'N/A'),
                    'ticket_channel': record.get('Ticket Channel', 'N/A'),
                    'product_purchased': record.get('Product Purchased', 'N/A'),
                    'description_length': len(str(record.get('Ticket Description', ''))),
                    'urgency_indicators': self._analyze_urgency(record.get('Ticket Description', ''))
                },
                'recommendation': self._generate_recommendation(prediction_result['predicted_satisfaction'])
            }
            
            logger.info("Prediction explanation generated")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                'prediction': prediction_result,
                'error': str(e),
                'explanation': 'Unable to generate detailed explanation'
            }
    
    def _analyze_urgency(self, description: str) -> Dict[str, Any]:
        """Analyze urgency indicators in ticket description"""
        description = str(description).lower()
        
        urgency_keywords = ['urgent', 'immediate', 'asap', 'emergency', 'critical']
        frustration_keywords = ['frustrated', 'annoyed', 'disappointed', 'angry']
        technical_keywords = ['error', 'bug', 'crash', 'freeze', 'malfunction']
        
        return {
            'urgency_score': sum(keyword in description for keyword in urgency_keywords),
            'frustration_score': sum(keyword in description for keyword in frustration_keywords),
            'technical_score': sum(keyword in description for keyword in technical_keywords),
            'has_urgency_indicators': any(keyword in description for keyword in urgency_keywords),
            'has_frustration_indicators': any(keyword in description for keyword in frustration_keywords),
            'has_technical_indicators': any(keyword in description for keyword in technical_keywords)
        }
    
    def _generate_recommendation(self, predicted_rating: int) -> str:
        """Generate recommendation based on predicted satisfaction"""
        if predicted_rating >= 4:
            return "Customer likely to be satisfied. Continue with standard support process."
        elif predicted_rating == 3:
            return "Customer satisfaction is neutral. Consider additional follow-up to ensure satisfaction."
        elif predicted_rating == 2:
            return "Customer may be unsatisfied. Prioritize quick resolution and quality assurance."
        else:
            return "Customer likely to be very unsatisfied. Escalate immediately and provide premium support."

def create_sample_prediction_data() -> Dict[str, Any]:
    """
    Create sample data for testing predictions
    
    Returns:
        Dictionary with sample customer and ticket data
    """
    return {
        'Customer Age': 35,
        'Customer Gender': 'Male',
        'Product Purchased': 'Dell XPS',
        'Ticket Type': 'Technical issue',
        'Ticket Subject': 'Software bug',
        'Ticket Description': 'I am experiencing a critical software bug that is causing frequent crashes. This is urgent as it affects my work.',
        'Ticket Priority': 'High',
        'Ticket Channel': 'Email',
        'Date of Purchase': '2023-01-15'
    }

def main():
    """
    Main function to demonstrate prediction functionality
    """
    try:
        # Create sample data
        sample_data = create_sample_prediction_data()
        
        print("="*60)
        print("CUSTOMER SATISFACTION PREDICTION DEMO")
        print("="*60)
        
        # Test different models
        models_to_test = ['Random Forest', 'XGBoost', 'Logistic Regression']
        
        for model_name in models_to_test:
            try:
                print(f"\nTesting {model_name} Model:")
                print("-" * 40)
                
                # Initialize predictor
                predictor = SatisfactionPredictor(model_name)
                
                # Make prediction
                result = predictor.predict_satisfaction(sample_data)
                
                # Display results
                print(f"Predicted Satisfaction: {result['predicted_satisfaction']}/5")
                print(f"Confidence Score: {result['confidence_score']:.3f}")
                print(f"Confidence Level: {result['prediction_details']['confidence_level']}")
                
                print("\nProbability Distribution:")
                for rating, prob in result['probability_distribution'].items():
                    print(f"  {rating}: {prob:.3f}")
                
                # Get explanation
                explanation = predictor.get_prediction_explanation(sample_data)
                print(f"\nRecommendation: {explanation['recommendation']}")
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # Test batch prediction
        print(f"\nTesting Batch Prediction:")
        print("-" * 40)
        
        # Create multiple sample records
        batch_data = [
            sample_data,
            {
                'Customer Age': 45,
                'Customer Gender': 'Female',
                'Product Purchased': 'LG Smart TV',
                'Ticket Type': 'Billing inquiry',
                'Ticket Subject': 'Payment issue',
                'Ticket Description': 'I need help with my billing. The payment method needs to be updated.',
                'Ticket Priority': 'Medium',
                'Ticket Channel': 'Phone',
                'Date of Purchase': '2023-06-01'
            }
        ]
        
        try:
            predictor = SatisfactionPredictor('Random Forest')
            batch_results = predictor.predict_batch(batch_data)
            
            for i, result in enumerate(batch_results):
                if 'error' not in result:
                    print(f"Record {i+1}: Rating {result['predicted_satisfaction']}/5 "
                          f"(Confidence: {result['confidence_score']:.3f})")
                else:
                    print(f"Record {i+1}: Error - {result['error']}")
        
        except Exception as e:
            print(f"Error in batch prediction: {e}")
        
        print("\n" + "="*60)
        print("PREDICTION DEMO COMPLETED")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()

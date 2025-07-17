"""
Model training module for Customer Satisfaction Prediction
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.utils import load_data, save_model, evaluate_model, print_data_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model training class for customer satisfaction prediction
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.model_results = {}
        self.feature_importance = {}
        
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load processed training and test data
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading processed data...")
        
        # Try to load feature engineered data first
        try:
            df_engineered = load_data(FEATURE_DATA_FILE)
            logger.info("Loaded feature engineered data")
        except:
            logger.info("Feature engineered data not found, loading cleaned data")
            df_engineered = load_data(CLEANED_DATA_FILE)
        
        # Filter records with satisfaction ratings
        df_model = df_engineered[df_engineered[TARGET_COLUMN].notna()].copy()
        
        print_data_summary(df_model, "Model Training Dataset")
        
        # Prepare features and target
        feature_columns = [col for col in df_model.columns if col != TARGET_COLUMN]
        X = df_model[feature_columns]
        y = df_model[TARGET_COLUMN]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=MODEL_RANDOM_STATE, 
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self) -> None:
        """
        Initialize machine learning models
        """
        logger.info("Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(**RF_PARAMS),
            'XGBoost': xgb.XGBClassifier(**XGB_PARAMS),
            'Logistic Regression': LogisticRegression(**LR_PARAMS)
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train base models without hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Training base models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_RANDOM_STATE),
                scoring='accuracy'
            )
            
            logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store model
            self.models[name] = model
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Perform hyperparameter tuning for models
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Random Forest hyperparameters
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # XGBoost hyperparameters
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Logistic Regression hyperparameters
        lr_param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        }
        
        param_grids = {
            'Random Forest': rf_param_grid,
            'XGBoost': xgb_param_grid,
            'Logistic Regression': lr_param_grid
        }
        
        # Perform grid search for each model
        for name, base_model in self.models.items():
            if name in param_grids:
                logger.info(f"Tuning hyperparameters for {name}...")
                
                grid_search = GridSearchCV(
                    base_model, param_grids[name],
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_RANDOM_STATE),
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store best model
                self.best_models[name] = grid_search.best_estimator_
                
                logger.info(f"{name} - Best CV Score: {grid_search.best_score_:.4f}")
                logger.info(f"{name} - Best Parameters: {grid_search.best_params_}")
        
        logger.info("Hyperparameter tuning completed")
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluate trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test target
        """
        logger.info("Evaluating models...")
        
        # Use best models if available, otherwise use base models
        models_to_evaluate = self.best_models if self.best_models else self.models
        
        for name, model in models_to_evaluate.items():
            logger.info(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate model
            results = evaluate_model(y_test, y_pred, name)
            self.model_results[name] = results
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = {
                    'features': X_test.columns.tolist(),
                    'importance': model.feature_importances_
                }
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute values of coefficients
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                self.feature_importance[name] = {
                    'features': X_test.columns.tolist(),
                    'importance': importance
                }
    
    def save_models(self) -> None:
        """
        Save trained models and results
        """
        logger.info("Saving models and results...")
        
        # Use best models if available, otherwise use base models
        models_to_save = self.best_models if self.best_models else self.models
        
        # Save individual models
        for name, model in models_to_save.items():
            if name == 'Random Forest':
                save_model(model, RF_MODEL_FILE)
            elif name == 'XGBoost':
                save_model(model, XGB_MODEL_FILE)
            elif name == 'Logistic Regression':
                save_model(model, LR_MODEL_FILE)
        
        # Save model results
        save_model(self.model_results, MODELS_DIR / "model_results.pkl")
        
        # Save feature importance
        save_model(self.feature_importance, MODELS_DIR / "feature_importance.pkl")
        
        logger.info("Models and results saved successfully")
    
    def print_model_comparison(self) -> None:
        """
        Print comparison of all models
        """
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.model_results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.model_results.keys(), 
                             key=lambda x: self.model_results[x]['accuracy'])
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {self.model_results[best_model_name]['accuracy']:.4f}")
        
        print("\n" + "="*70)
    
    def print_feature_importance(self, top_n: int = 15) -> None:
        """
        Print feature importance for models
        
        Args:
            top_n: Number of top features to display
        """
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE")
        print("="*70)
        
        for name, importance_data in self.feature_importance.items():
            print(f"\n{name}:")
            print("-" * 30)
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': importance_data['features'],
                'Importance': importance_data['importance']
            }).sort_values('Importance', ascending=False).head(top_n)
            
            for _, row in importance_df.iterrows():
                print(f"{row['Feature']:<30} {row['Importance']:.4f}")
        
        print("\n" + "="*70)
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete model training pipeline
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training pipeline...")
        
        # Load processed data
        X_train, X_test, y_train, y_test = self.load_processed_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train base models
        self.train_base_models(X_train, y_train)
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Save models
        self.save_models()
        
        # Print results
        self.print_model_comparison()
        self.print_feature_importance()
        
        logger.info("Model training pipeline completed successfully!")
        
        return {
            'models': self.best_models if self.best_models else self.models,
            'results': self.model_results,
            'feature_importance': self.feature_importance,
            'data_shapes': {
                'train': X_train.shape,
                'test': X_test.shape
            }
        }

def train_ensemble_model(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Train an ensemble model combining multiple algorithms
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with ensemble results
    """
    logger.info("Training ensemble model...")
    
    from sklearn.ensemble import VotingClassifier
    
    # Load individual models
    rf_model = RandomForestClassifier(**RF_PARAMS)
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    lr_model = LogisticRegression(**LR_PARAMS)
    
    # Train individual models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_results = evaluate_model(y_test, y_pred_ensemble, 'Ensemble')
    
    # Save ensemble model
    save_model(ensemble, MODELS_DIR / "ensemble_model.pkl")
    
    logger.info(f"Ensemble Model - Accuracy: {ensemble_results['accuracy']:.4f}")
    
    return ensemble_results

def main():
    """
    Main function to run model training
    """
    try:
        # Create directories
        create_directories()
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Run training pipeline
        training_results = trainer.run_training_pipeline()
        
        # Train ensemble model
        X_train, X_test, y_train, y_test = trainer.load_processed_data()
        ensemble_results = train_ensemble_model(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Models trained: {len(training_results['models'])}")
        print(f"Training data shape: {training_results['data_shapes']['train']}")
        print(f"Test data shape: {training_results['data_shapes']['test']}")
        
        print("\nModel files saved:")
        print(f"- Random Forest: {RF_MODEL_FILE}")
        print(f"- XGBoost: {XGB_MODEL_FILE}")
        print(f"- Logistic Regression: {LR_MODEL_FILE}")
        print(f"- Ensemble: {MODELS_DIR / 'ensemble_model.pkl'}")
        print(f"- Results: {MODELS_DIR / 'model_results.pkl'}")
        print(f"- Feature Importance: {MODELS_DIR / 'feature_importance.pkl'}")
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

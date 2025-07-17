"""
Main application runner for Customer Satisfaction Prediction
"""
import sys
import logging
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config import create_directories
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import SatisfactionPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_preprocessing():
    """Run data preprocessing pipeline"""
    logger.info("Starting data preprocessing...")
    
    try:
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.run_preprocessing_pipeline()
        
        logger.info("Data preprocessing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return False

def run_feature_engineering():
    """Run feature engineering pipeline"""
    logger.info("Starting feature engineering...")
    
    try:
        feature_engineer = FeatureEngineer()
        df_engineered = feature_engineer.run_feature_engineering_pipeline()
        
        logger.info("Feature engineering completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return False

def run_model_training():
    """Run model training pipeline"""
    logger.info("Starting model training...")
    
    try:
        trainer = ModelTrainer()
        training_results = trainer.run_training_pipeline()
        
        logger.info("Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False

def run_model_evaluation():
    """Run model evaluation pipeline"""
    logger.info("Starting model evaluation...")
    
    try:
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.run_evaluation_pipeline()
        
        logger.info("Model evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        return False

def run_prediction_demo():
    """Run prediction demonstration"""
    logger.info("Starting prediction demo...")
    
    try:
        from src.prediction import main as prediction_main
        prediction_main()
        
        logger.info("Prediction demo completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in prediction demo: {e}")
        return False

def run_web_app():
    """Run Flask web application"""
    logger.info("Starting Flask web application...")
    
    try:
        from app.app import app, initialize_predictors, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
        
        # Initialize predictors
        initialize_predictors()
        
        # Run Flask app
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
        
    except Exception as e:
        logger.error(f"Error starting Flask web application: {e}")
        return False

def run_full_pipeline():
    """Run the complete ML pipeline"""
    logger.info("Starting complete ML pipeline...")
    
    steps = [
        ("Data Preprocessing", run_preprocessing),
        ("Feature Engineering", run_feature_engineering),
        ("Model Training", run_model_training),
        ("Model Evaluation", run_model_evaluation)
    ]
    
    for step_name, step_function in steps:
        logger.info(f"Running {step_name}...")
        
        if not step_function():
            logger.error(f"Failed at {step_name}")
            return False
        
        logger.info(f"{step_name} completed successfully")
    
    logger.info("Complete ML pipeline finished successfully!")
    return True

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Customer Satisfaction Prediction Application")
    parser.add_argument("--mode", type=str, default="full", 
                       choices=["full", "preprocess", "feature", "train", "evaluate", "predict", "web"],
                       help="Mode to run the application")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create directories
    create_directories()
    
    # Print welcome message
    print("="*70)
    print("CUSTOMER SATISFACTION PREDICTION APPLICATION")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Verbose: {args.verbose}")
    print("="*70)
    
    # Run based on mode
    success = False
    
    if args.mode == "full":
        success = run_full_pipeline()
    elif args.mode == "preprocess":
        success = run_preprocessing()
    elif args.mode == "feature":
        success = run_feature_engineering()
    elif args.mode == "train":
        success = run_model_training()
    elif args.mode == "evaluate":
        success = run_model_evaluation()
    elif args.mode == "predict":
        success = run_prediction_demo()
    elif args.mode == "web":
        success = run_web_app()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    # Print completion message
    if success:
        print("\n" + "="*70)
        print("APPLICATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        if args.mode == "full":
            print("All pipeline steps completed:")
            print("✓ Data preprocessing")
            print("✓ Feature engineering")
            print("✓ Model training")
            print("✓ Model evaluation")
            print("\nNext steps:")
            print("1. Run 'python main.py --mode predict' to test predictions")
            print("2. Run 'python main.py --mode web' to start web application")
        
        elif args.mode == "web":
            print("Web application is running!")
            print("Access it at: http://localhost:5000")
        
        else:
            print(f"Mode '{args.mode}' completed successfully!")
    
    else:
        print("\n" + "="*70)
        print("APPLICATION FAILED!")
        print("="*70)
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

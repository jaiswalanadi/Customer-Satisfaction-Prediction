"""
Model evaluation module for Customer Satisfaction Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import *
from src.utils import (load_data, load_model, plot_confusion_matrix, 
                      plot_feature_importance, create_model_comparison_plot)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Model evaluation class for comprehensive analysis
    """
    
    def __init__(self):
        self.models = {}
        self.test_data = None
        self.predictions = {}
        self.evaluation_results = {}
        
    def load_models_and_data(self) -> None:
        """
        Load trained models and test data
        """
        logger.info("Loading models and test data...")
        
        # Load models
        try:
            self.models['Random Forest'] = load_model(RF_MODEL_FILE)
            logger.info("Loaded Random Forest model")
        except:
            logger.warning("Random Forest model not found")
        
        try:
            self.models['XGBoost'] = load_model(XGB_MODEL_FILE)
            logger.info("Loaded XGBoost model")
        except:
            logger.warning("XGBoost model not found")
        
        try:
            self.models['Logistic Regression'] = load_model(LR_MODEL_FILE)
            logger.info("Loaded Logistic Regression model")
        except:
            logger.warning("Logistic Regression model not found")
        
        try:
            self.models['Ensemble'] = load_model(MODELS_DIR / "ensemble_model.pkl")
            logger.info("Loaded Ensemble model")
        except:
            logger.warning("Ensemble model not found")
        
        # Load test data
        try:
            test_df = load_data(TEST_DATA_FILE)
            self.test_data = {
                'X': test_df.drop(columns=[TARGET_COLUMN]),
                'y': test_df[TARGET_COLUMN]
            }
            logger.info(f"Loaded test data: {self.test_data['X'].shape}")
        except:
            logger.error("Could not load test data")
            raise
    
    def generate_predictions(self) -> None:
        """
        Generate predictions for all models
        """
        logger.info("Generating predictions...")
        
        X_test = self.test_data['X']
        
        for name, model in self.models.items():
            try:
                # Generate predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                self.predictions[name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"Generated predictions for {name}")
            except Exception as e:
                logger.error(f"Error generating predictions for {name}: {e}")
    
    def calculate_detailed_metrics(self) -> None:
        """
        Calculate detailed evaluation metrics
        """
        logger.info("Calculating detailed metrics...")
        
        y_true = self.test_data['y']
        
        for name, pred_data in self.predictions.items():
            y_pred = pred_data['predictions']
            y_pred_proba = pred_data['probabilities']
            
            # Basic metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            # Per-class metrics
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # ROC AUC for multiclass
            try:
                from sklearn.metrics import roc_auc_score
                roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None
            
            self.evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'support': support
            }
            
            logger.info(f"Calculated metrics for {name}")
    
    def plot_confusion_matrices(self, save_path: Path = None) -> None:
        """
        Plot confusion matrices for all models
        
        Args:
            save_path: Directory to save plots
        """
        logger.info("Plotting confusion matrices...")
        
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, results) in enumerate(self.evaluation_results.items()):
            if i < len(axes):
                ax = axes[i]
                conf_matrix = results['confusion_matrix']
                
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=range(1, 6), yticklabels=range(1, 6), ax=ax)
                ax.set_title(f'Confusion Matrix - {name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.evaluation_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / "confusion_matrices.png", dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance_comparison(self, save_path: Path = None) -> None:
        """
        Plot feature importance comparison across models
        
        Args:
            save_path: Directory to save plots
        """
        logger.info("Plotting feature importance comparison...")
        
        try:
            feature_importance = load_model(MODELS_DIR / "feature_importance.pkl")
        except:
            logger.warning("Feature importance data not found")
            return
        
        # Create subplots for each model
        n_models = len(feature_importance)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 6 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, importance_data) in enumerate(feature_importance.items()):
            ax = axes[i]
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': importance_data['features'],
                'Importance': importance_data['importance']
            }).sort_values('Importance', ascending=False).head(15)
            
            # Plot
            sns.barplot(data=importance_df, x='Importance', y='Feature', 
                       palette='viridis', ax=ax)
            ax.set_title(f'Top 15 Feature Importance - {name}')
            ax.set_xlabel('Importance Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / "feature_importance_comparison.png", 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance comparison saved to {save_path}")
        
        plt.show()
    
    def plot_model_performance_comparison(self, save_path: Path = None) -> None:
        """
        Plot model performance comparison
        
        Args:
            save_path: Directory to save plots
        """
        logger.info("Plotting model performance comparison...")
        
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.evaluation_results.keys())
        
        # Create comparison DataFrame
        comparison_data = []
        for metric in metrics:
            for name in model_names:
                if self.evaluation_results[name][metric] is not None:
                    comparison_data.append({
                        'Model': name,
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': self.evaluation_results[name][metric]
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model', 
                   palette='Set2')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / "model_performance_comparison.png", 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Model performance comparison saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, save_path: Path = None) -> None:
        """
        Plot prediction distribution for each model
        
        Args:
            save_path: Directory to save plots
        """
        logger.info("Plotting prediction distributions...")
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (name, pred_data) in enumerate(self.predictions.items()):
            if i < len(axes):
                ax = axes[i]
                predictions = pred_data['predictions']
                
                # Count predictions
                pred_counts = pd.Series(predictions).value_counts().sort_index()
                
                # Plot distribution
                ax.bar(pred_counts.index, pred_counts.values, alpha=0.7)
                ax.set_title(f'Prediction Distribution - {name}')
                ax.set_xlabel('Predicted Rating')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for j, count in enumerate(pred_counts.values):
                    ax.text(pred_counts.index[j], count + 5, str(count), 
                           ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(self.predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / "prediction_distributions.png", 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distributions saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_path: Path = None) -> None:
        """
        Plot ROC curves for multiclass classification
        
        Args:
            save_path: Directory to save plots
        """
        logger.info("Plotting ROC curves...")
        
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        y_true = self.test_data['y']
        classes = sorted(y_true.unique())
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=classes)
        n_classes = len(classes)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, pred_data) in enumerate(self.predictions.items()):
            if i < len(axes):
                ax = axes[i]
                y_pred_proba = pred_data['probabilities']
                
                # Compute ROC curve and AUC for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for j in range(n_classes):
                    fpr[j], tpr[j], _ = roc_curve(y_true_bin[:, j], y_pred_proba[:, j])
                    roc_auc[j] = auc(fpr[j], tpr[j])
                
                # Plot ROC curves
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for j, color in zip(range(n_classes), colors):
                    ax.plot(fpr[j], tpr[j], color=color, lw=2,
                           label=f'Class {classes[j]} (AUC = {roc_auc[j]:.2f})')
                
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curves - {name}')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / "roc_curves.png", dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self) -> str:
        """
        Generate a detailed evaluation report
        
        Returns:
            String containing the detailed report
        """
        logger.info("Generating detailed evaluation report...")
        
        report = []
        report.append("="*70)
        report.append("CUSTOMER SATISFACTION PREDICTION - MODEL EVALUATION REPORT")
        report.append("="*70)
        
        # Dataset summary
        report.append(f"\nDATASET SUMMARY:")
        report.append(f"Test set size: {len(self.test_data['y'])}")
        report.append(f"Number of features: {self.test_data['X'].shape[1]}")
        report.append(f"Target distribution:")
        target_dist = self.test_data['y'].value_counts().sort_index()
        for rating, count in target_dist.items():
            report.append(f"  Rating {rating}: {count} ({count/len(self.test_data['y'])*100:.1f}%)")
        
        # Model performance summary
        report.append(f"\nMODEL PERFORMANCE SUMMARY:")
        report.append(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        report.append("-" * 60)
        
        for name, results in self.evaluation_results.items():
            report.append(f"{name:<20} {results['accuracy']:<10.4f} {results['precision']:<10.4f} "
                         f"{results['recall']:<10.4f} {results['f1_score']:<10.4f}")
        
        # Best model identification
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['accuracy'])
        report.append(f"\nBEST MODEL: {best_model}")
        report.append(f"Best Accuracy: {self.evaluation_results[best_model]['accuracy']:.4f}")
        
        # Detailed classification reports
        for name, results in self.evaluation_results.items():
            report.append(f"\n{name.upper()} - DETAILED CLASSIFICATION REPORT:")
            report.append("-" * 50)
            
            class_report = results['classification_report']
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    if class_name.isdigit():
                        report.append(f"Rating {class_name}:")
                    else:
                        report.append(f"{class_name}:")
                    
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {metric}: {value:.4f}")
        
        # Feature importance summary
        try:
            feature_importance = load_model(MODELS_DIR / "feature_importance.pkl")
            report.append(f"\nTOP 10 FEATURE IMPORTANCE:")
            
            for name, importance_data in feature_importance.items():
                report.append(f"\n{name}:")
                
                # Sort features by importance
                sorted_features = sorted(
                    zip(importance_data['features'], importance_data['importance']),
                    key=lambda x: x[1], reverse=True
                )[:10]
                
                for feature, importance in sorted_features:
                    report.append(f"  {feature:<30} {importance:.4f}")
        except:
            report.append(f"\nFeature importance data not available")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        report.append(f"1. Use {best_model} as the primary model for production")
        report.append(f"2. Monitor model performance regularly with new data")
        report.append(f"3. Consider ensemble methods for improved performance")
        report.append(f"4. Focus on improving data quality for better predictions")
        report.append(f"5. Implement A/B testing for model deployment")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def save_evaluation_results(self, save_path: Path = None) -> None:
        """
        Save all evaluation results and plots
        
        Args:
            save_path: Directory to save results
        """
        if save_path is None:
            save_path = FIGURES_DIR / "model_performance"
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving evaluation results to {save_path}")
        
        # Save evaluation metrics
        import pickle
        with open(save_path / "evaluation_results.pkl", 'wb') as f:
            pickle.dump(self.evaluation_results, f)
        
        # Save predictions
        with open(save_path / "predictions.pkl", 'wb') as f:
            pickle.dump(self.predictions, f)
        
        # Generate and save plots
        self.plot_confusion_matrices(save_path)
        self.plot_feature_importance_comparison(save_path)
        self.plot_model_performance_comparison(save_path)
        self.plot_prediction_distribution(save_path)
        self.plot_roc_curves(save_path)
        
        # Generate and save detailed report
        detailed_report = self.generate_detailed_report()
        with open(save_path / "detailed_evaluation_report.txt", 'w') as f:
            f.write(detailed_report)
        
        logger.info("All evaluation results saved successfully")
    
    def run_evaluation_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete model evaluation pipeline
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting model evaluation pipeline...")
        
        # Load models and data
        self.load_models_and_data()
        
        # Generate predictions
        self.generate_predictions()
        
        # Calculate metrics
        self.calculate_detailed_metrics()
        
        # Save results and generate plots
        self.save_evaluation_results()
        
        # Print summary
        print(self.generate_detailed_report())
        
        logger.info("Model evaluation pipeline completed successfully!")
        
        return {
            'models_evaluated': len(self.models),
            'evaluation_results': self.evaluation_results,
            'predictions': self.predictions,
            'test_data_size': len(self.test_data['y'])
        }

def main():
    """
    Main function to run model evaluation
    """
    try:
        # Create directories
        create_directories()
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Run evaluation pipeline
        evaluation_results = evaluator.run_evaluation_pipeline()
        
        print("\n" + "="*50)
        print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Models evaluated: {evaluation_results['models_evaluated']}")
        print(f"Test data size: {evaluation_results['test_data_size']}")
        
        print("\nEvaluation files saved:")
        save_path = FIGURES_DIR / "model_performance"
        print(f"- Evaluation results: {save_path / 'evaluation_results.pkl'}")
        print(f"- Predictions: {save_path / 'predictions.pkl'}")
        print(f"- Detailed report: {save_path / 'detailed_evaluation_report.txt'}")
        print(f"- Plots: {save_path}/")
        
    except Exception as e:
        logger.error(f"Error in model evaluation pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

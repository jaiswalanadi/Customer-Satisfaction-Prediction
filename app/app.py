"""
Flask web application for Customer Satisfaction Prediction
"""
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import json
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.prediction import SatisfactionPredictor, create_sample_prediction_data
from config.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'customer_satisfaction_prediction_secret_key'

# Global predictor instances
predictors = {}

def initialize_predictors():
    """Initialize prediction models"""
    global predictors
    
    models_to_load = ['Random Forest', 'XGBoost', 'Logistic Regression', 'Ensemble']
    
    for model_name in models_to_load:
        try:
            predictors[model_name] = SatisfactionPredictor(model_name)
            logger.info(f"Successfully loaded {model_name} model")
        except Exception as e:
            logger.error(f"Failed to load {model_name} model: {e}")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', models=list(predictors.keys()))

@app.route('/predict')
def predict_form():
    """Prediction form page"""
    sample_data = create_sample_prediction_data()
    return render_template('predict.html', 
                         models=list(predictors.keys()), 
                         sample_data=sample_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get model name
        model_name = data.get('model', 'Random Forest')
        
        if model_name not in predictors:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Extract customer data
        customer_data = {
            'Customer Age': int(data.get('customer_age', 30)),
            'Customer Gender': data.get('customer_gender', 'Other'),
            'Product Purchased': data.get('product_purchased', 'Unknown'),
            'Ticket Type': data.get('ticket_type', 'Technical issue'),
            'Ticket Subject': data.get('ticket_subject', 'General inquiry'),
            'Ticket Description': data.get('ticket_description', ''),
            'Ticket Priority': data.get('ticket_priority', 'Medium'),
            'Ticket Channel': data.get('ticket_channel', 'Email'),
            'Date of Purchase': data.get('date_of_purchase', '2023-01-01')
        }
        
        # Make prediction
        predictor = predictors[model_name]
        result = predictor.predict_satisfaction(customer_data)
        
        # Get explanation
        explanation = predictor.get_prediction_explanation(customer_data)
        
        # Combine results
        response = {
            'success': True,
            'prediction': result,
            'explanation': explanation,
            'model_used': model_name
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch prediction"""
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'records' not in data:
            return jsonify({'error': 'No records provided'}), 400
        
        # Get model name
        model_name = data.get('model', 'Random Forest')
        
        if model_name not in predictors:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Process batch records
        records = data['records']
        predictor = predictors[model_name]
        
        # Make batch predictions
        results = predictor.predict_batch(records)
        
        response = {
            'success': True,
            'predictions': results,
            'model_used': model_name,
            'total_records': len(records)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch prediction API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form_submit():
    """Handle form submission for prediction"""
    try:
        # Get form data
        customer_data = {
            'Customer Age': int(request.form.get('customer_age', 30)),
            'Customer Gender': request.form.get('customer_gender', 'Other'),
            'Product Purchased': request.form.get('product_purchased', 'Unknown'),
            'Ticket Type': request.form.get('ticket_type', 'Technical issue'),
            'Ticket Subject': request.form.get('ticket_subject', 'General inquiry'),
            'Ticket Description': request.form.get('ticket_description', ''),
            'Ticket Priority': request.form.get('ticket_priority', 'Medium'),
            'Ticket Channel': request.form.get('ticket_channel', 'Email'),
            'Date of Purchase': request.form.get('date_of_purchase', '2023-01-01')
        }
        
        # Get selected model
        model_name = request.form.get('model', 'Random Forest')
        
        if model_name not in predictors:
            flash(f'Model {model_name} not available', 'error')
            return redirect(url_for('predict_form'))
        
        # Make prediction
        predictor = predictors[model_name]
        result = predictor.predict_satisfaction(customer_data)
        explanation = predictor.get_prediction_explanation(customer_data)
        
        return render_template('results.html', 
                             prediction=result, 
                             explanation=explanation,
                             customer_data=customer_data,
                             model_used=model_name)
        
    except Exception as e:
        logger.error(f"Error in form prediction: {e}")
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('predict_form'))

@app.route('/api/models')
def api_models():
    """Get available models"""
    return jsonify({
        'models': list(predictors.keys()),
        'default_model': 'Random Forest'
    })

@app.route('/api/sample_data')
def api_sample_data():
    """Get sample data for testing"""
    sample_data = create_sample_prediction_data()
    return jsonify(sample_data)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(predictors),
        'available_models': list(predictors.keys())
    })

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    try:
        # Initialize predictors
        logger.info("Initializing prediction models...")
        initialize_predictors()
        
        if not predictors:
            logger.error("No models could be loaded. Please ensure models are trained first.")
            exit(1)
        
        logger.info(f"Successfully loaded {len(predictors)} models")
        
        # Run Flask app
        logger.info(f"Starting Flask app on {FLASK_HOST}:{FLASK_PORT}")
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
        
    except Exception as e:
        logger.error(f"Error starting Flask app: {e}")
        exit(1)

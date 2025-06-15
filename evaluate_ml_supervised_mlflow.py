import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Global variables for progress tracking
ml_evaluation_progress = {}
ml_evaluation_results = {}

def update_progress(model_name, current_task, progress_percent):
    """Update progress for a specific model evaluation."""
    ml_evaluation_progress[model_name] = {
        "current_task": current_task,
        "progress_percent": progress_percent,
        "timestamp": datetime.now().isoformat()
    }

def get_ml_progress(model_name):
    """Get current progress for a model evaluation."""
    return ml_evaluation_progress.get(model_name, {
        "current_task": "Not started",
        "progress_percent": 0,
        "timestamp": datetime.now().isoformat()
    })

def clear_ml_progress(model_name):
    """Clear progress for a specific model."""
    if model_name in ml_evaluation_progress:
        del ml_evaluation_progress[model_name]
    if model_name in ml_evaluation_results:
        del ml_evaluation_results[model_name]

def load_model(model_path):
    """Load pickle model."""
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_problem_type(model):
    """Detect if it's classification or regression based on model type."""
    model_type = type(model).__name__.lower()
    
    # Common classification models
    classification_keywords = ['classifier', 'logistic', 'svm', 'randomforest', 'decisiontree', 
                             'gradient', 'xgb', 'lgb', 'naive', 'knn']
    
    # Common regression models  
    regression_keywords = ['regression', 'regressor', 'linear', 'ridge', 'lasso', 'elastic']
    
    for keyword in classification_keywords:
        if keyword in model_type:
            return 'classification'
    
    for keyword in regression_keywords:
        if keyword in model_type:
            return 'regression'
    
    # Default to regression if uncertain
    return 'regression'

def get_model_info(model):
    """Extract model information."""
    info = {
        'model_type': type(model).__name__,
        'model_params': getattr(model, 'get_params', lambda: {})(),
        'feature_count': None,
        'has_feature_importance': hasattr(model, 'feature_importances_'),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'has_coefficients': hasattr(model, 'coef_')
    }
    
    # Try to get feature count from model
    if hasattr(model, 'n_features_in_'):
        info['feature_count'] = model.n_features_in_
    elif hasattr(model, 'coef_'):
        if hasattr(model.coef_, 'shape'):
            info['feature_count'] = model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
    
    return info

def generate_model_summary_plots(model_name, model, model_info):
    """Generate model summary visualizations."""
    plots = {}
    
    try:
        plots_dir = f"static/plots/{model_name}"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Model Parameters Visualization
        if model_info['model_params']:
            fig, ax = plt.subplots(figsize=(10, 6))
            params = model_info['model_params']
            
            # Filter numeric parameters for visualization
            numeric_params = {k: v for k, v in params.items() 
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}
            
            if numeric_params:
                param_names = list(numeric_params.keys())
                param_values = list(numeric_params.values())
                
                bars = ax.bar(param_names, param_values, color='skyblue', alpha=0.7)
                ax.set_title('Model Parameters', fontsize=14, fontweight='bold')
                ax.set_ylabel('Parameter Values')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, param_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}' if isinstance(value, float) else str(value),
                           ha='center', va='bottom')
                
                plt.tight_layout()
                param_path = f"{plots_dir}/model_parameters.png"
                plt.savefig(param_path, dpi=150, bbox_inches='tight')
                plt.close()
                plots['model_parameters'] = f"plots/{model_name}/model_parameters.png"
        
        # Feature Importance (if available)
        if model_info['has_feature_importance']:
            try:
                importances = model.feature_importances_
                if len(importances) <= 20:  # Only for reasonable number of features
                    fig, ax = plt.subplots(figsize=(10, 6))
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                    
                    bars = ax.bar(range(len(indices)), importances[indices], color='lightcoral', alpha=0.7)
                    ax.set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Importance Score')
                    ax.set_xticks(range(len(indices)))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, idx in zip(bars, indices):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{importances[idx]:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    feat_path = f"{plots_dir}/feature_importance.png"
                    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    plots['feature_importance'] = f"plots/{model_name}/feature_importance.png"
            except Exception as e:
                print(f"Could not generate feature importance plot: {e}")
        
        # Coefficients Plot (for linear models)
        if model_info['has_coefficients']:
            try:
                coef = model.coef_
                if hasattr(coef, 'flatten'):
                    coef = coef.flatten()
                
                if len(coef) <= 20:  # Only for reasonable number of features
                    fig, ax = plt.subplots(figsize=(10, 6))
                    feature_names = [f'Feature_{i}' for i in range(len(coef))]
                    
                    colors = ['red' if c < 0 else 'blue' for c in coef]
                    bars = ax.bar(feature_names, coef, color=colors, alpha=0.7)
                    ax.set_title('Model Coefficients', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Coefficient Value')
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, coef):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               height + (0.01 * max(abs(coef)) if height >= 0 else -0.01 * max(abs(coef))),
                               f'{value:.3f}', ha='center', 
                               va='bottom' if height >= 0 else 'top')
                    
                    plt.tight_layout()
                    coef_path = f"{plots_dir}/coefficients.png"
                    plt.savefig(coef_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    plots['coefficients'] = f"plots/{model_name}/coefficients.png"
            except Exception as e:
                print(f"Could not generate coefficients plot: {e}")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    return plots

def run_ml_evaluation(model_name, model_path, dataset_path):
    """Run ML model evaluation using MLflow - model info only."""
    try:
        update_progress(model_name, "Initializing MLflow...", 10)
        
        # Set up MLflow
        mlflow.set_experiment(f"ML_Evaluation_{model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            update_progress(model_name, "Loading model...", 20)
            
            # Find model file
            model_file = None
            if os.path.isfile(model_path) and model_path.endswith('.pkl'):
                model_file = model_path
            else:
                for file in os.listdir(model_path):
                    if file.endswith('.pkl'):
                        model_file = os.path.join(model_path, file)
                        break
            
            if not model_file:
                raise Exception("No .pkl model file found")
            
            model = load_model(model_file)
            if model is None:
                raise Exception("Failed to load model")
            
            update_progress(model_name, "Analyzing model...", 40)
            
            # Get model information
            model_info = get_model_info(model)
            problem_type = detect_problem_type(model)
            
            # Log model info to MLflow
            mlflow.log_param("model_type", model_info['model_type'])
            mlflow.log_param("problem_type", problem_type)
            mlflow.log_param("feature_count", model_info['feature_count'])
            mlflow.log_param("has_feature_importance", model_info['has_feature_importance'])
            mlflow.log_param("has_predict_proba", model_info['has_predict_proba'])
            mlflow.log_param("has_coefficients", model_info['has_coefficients'])
            
            # Log model parameters
            for param_name, param_value in model_info['model_params'].items():
                try:
                    if isinstance(param_value, (int, float, str, bool)):
                        mlflow.log_param(f"model_{param_name}", param_value)
                except:
                    pass  # Skip parameters that can't be logged
            
            update_progress(model_name, "Generating visualizations...", 60)
            
            # Generate model summary plots
            plots = generate_model_summary_plots(model_name, model, model_info)
            
            update_progress(model_name, "Logging artifacts...", 80)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log plots
            for plot_name, plot_path in plots.items():
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, "plots")
            
            update_progress(model_name, "Finalizing results...", 95)
            
            # Prepare final results
            results = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'problem_type': problem_type,
                'model_info': model_info,
                'plots': plots,
                'mlflow_run_id': mlflow.active_run().info.run_id,
                'model_path': model_file
            }
            
            # Store results
            ml_evaluation_results[model_name] = results
            
            update_progress(model_name, "Evaluation completed!", 100)
            
            return results
            
    except Exception as e:
        update_progress(model_name, f"Error: {str(e)}", 0)
        ml_evaluation_results[model_name] = {"error": str(e)}
        raise e

def get_ml_results(model_name):
    """Get evaluation results for a specific model."""
    return ml_evaluation_results.get(model_name, {})
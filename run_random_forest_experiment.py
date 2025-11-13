import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time
import json
import os
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def setup_mlflow():
    """Setup MLflow tracking for this experiment"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("random-forest-hyperparameter-tuning")
    print("✅ MLflow setup complete for Random Forest Hyperparameter Tuning")

def load_and_preprocess_data():
    """Load and preprocess the spam dataset"""
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'message']
        df = df.dropna()
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        print(f"✅ Dataset loaded: {len(df)} messages")
        print(f"📊 Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_tfidf_features(df, max_features=2000):
    """Create TF-IDF features"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X = vectorizer.fit_transform(df['message'])
        y = df['label']
        
        # Save vectorizer
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        print(f"✅ TF-IDF features created with {max_features} features")
        
        return X, y, vectorizer
        
    except Exception as e:
        print(f"❌ Error creating TF-IDF features: {e}")
        return None, None, None

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Create and save confusion matrix"""
    try:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"❌ Error creating confusion matrix: {e}")
        return None

def get_parameter_combinations():
    """Define different parameter combinations for Random Forest"""
    param_combinations = [
        {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True
        },
        {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True
        },
        {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'log2',
            'bootstrap': True
        },
        {
            'n_estimators': 150,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'bootstrap': False
        },
        {
            'n_estimators': 75,
            'max_depth': 25,
            'min_samples_split': 3,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'bootstrap': True
        },
        {
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': False
        },
        {
            'n_estimators': 100,
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,  # Use all features
            'bootstrap': True
        },
        {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'bootstrap': True
        }
    ]
    return param_combinations

def run_single_rf_experiment(X_train, X_test, y_train, y_test, params, run_name):
    """Run a single Random Forest experiment with given parameters"""
    try:
        # Start nested MLflow run
        with mlflow.start_run(run_name=run_name, nested=True):
            start_time = time.time()
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param('model_type', 'random_forest')
            
            # Initialize model with current parameters
            rf_model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            training_start = time.time()
            print(f"🔄 Training with params: {params}")
            rf_model.fit(X_train, y_train)
            training_time = time.time() - training_start
            
            # Make predictions
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            total_time = time.time() - start_time
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'training_time': training_time,
                'total_time': total_time,
                'precision_0': class_report['0']['precision'],
                'recall_0': class_report['0']['recall'],
                'f1_0': class_report['0']['f1-score'],
                'precision_1': class_report['1']['precision'],
                'recall_1': class_report['1']['recall'],
                'f1_1': class_report['1']['f1-score']
            })
            
            # Create and log confusion matrix for the best run only
            if accuracy > 0.97:  # Only create plots for high-performing models
                cm_plot_path = plot_confusion_matrix(y_test, y_pred, f"RF_{run_name}")
                if cm_plot_path:
                    mlflow.log_artifact(cm_plot_path)
            
            # Log model
            mlflow.sklearn.log_model(rf_model, "model")
            
            # Print results
            print(f"   ✅ Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, Time: {training_time:.2f}s")
            
            return {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'training_time': training_time,
                'total_time': total_time,
                'params': params,
                'model': rf_model
            }
            
    except Exception as e:
        print(f"❌ Error in experiment {run_name}: {e}")
        return None

def run_random_forest_experiment():
    """Run Random Forest experiments with multiple parameter combinations"""
    print("\n" + "="*60)
    print("🎯 RANDOM FOREST HYPERPARAMETER TUNING EXPERIMENT")
    print("="*60)
    
    # Setup MLflow
    setup_mlflow()
    
    try:
        # Load data once
        df = load_and_preprocess_data()
        if df is None:
            return None
        
        # Create features
        X, y, vectorizer = create_tfidf_features(df, max_features=2000)
        if X is None:
            return None
        
        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        print(f"🔬 Testing {len(get_parameter_combinations())} parameter combinations\n")
        
        # Start parent run
        with mlflow.start_run(run_name="rf_hyperparameter_tuning"):
            # Log dataset info
            mlflow.log_params({
                'dataset_size': len(df),
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'num_features': X.shape[1],
                'num_parameter_combinations': len(get_parameter_combinations())
            })
            
            # Run experiments with different parameters
            results = []
            param_combinations = get_parameter_combinations()
            
            for i, params in enumerate(param_combinations, 1):
                run_name = f"rf_experiment_{i}"
                print(f"\n🔬 Experiment {i}/{len(param_combinations)}")
                print(f"📋 Parameters: {params}")
                
                result = run_single_rf_experiment(
                    X_train, X_test, y_train, y_test, 
                    params, run_name
                )
                
                if result:
                    results.append(result)
            
            # Find best model
            if results:
                best_result = max(results, key=lambda x: x['accuracy'])
                worst_result = min(results, key=lambda x: x['accuracy'])
                
                # Log comparison results
                comparison = {
                    'best_accuracy': best_result['accuracy'],
                    'best_roc_auc': best_result['roc_auc'],
                    'worst_accuracy': worst_result['accuracy'],
                    'worst_roc_auc': worst_result['roc_auc'],
                    'accuracy_range': best_result['accuracy'] - worst_result['accuracy'],
                    'avg_accuracy': np.mean([r['accuracy'] for r in results]),
                    'avg_training_time': np.mean([r['training_time'] for r in results]),
                    'total_experiments': len(results)
                }
                
                # Log best model info
                mlflow.log_metrics({
                    'best_accuracy': best_result['accuracy'],
                    'best_roc_auc': best_result['roc_auc'],
                    'avg_accuracy': comparison['avg_accuracy']
                })
                
                mlflow.log_params({
                    'best_n_estimators': best_result['params']['n_estimators'],
                    'best_max_depth': best_result['params']['max_depth'],
                    'best_max_features': best_result['params']['max_features']
                })
                
                # Save best model locally
                best_model_path = 'models/best_rf_model.pkl'
                joblib.dump(best_result['model'], best_model_path)
                mlflow.log_artifact(best_model_path)
                
                # Save results summary
                results_summary = {
                    'experiment_date': datetime.now().isoformat(),
                    'total_experiments': len(results),
                    'comparison': comparison,
                    'best_parameters': best_result['params'],
                    'best_metrics': {
                        'accuracy': float(best_result['accuracy']),
                        'roc_auc': float(best_result['roc_auc']),
                        'training_time': float(best_result['training_time'])
                    },
                    'all_results': [
                        {
                            'experiment_id': i+1,
                            'accuracy': float(r['accuracy']),
                            'roc_auc': float(r['roc_auc']),
                            'training_time': float(r['training_time']),
                            'parameters': r['params']
                        }
                        for i, r in enumerate(results)
                    ]
                }
                
                os.makedirs('metrics', exist_ok=True)
                with open('metrics/rf_hyperparameter_results.json', 'w') as f:
                    json.dump(results_summary, f, indent=2)
                
                mlflow.log_artifact('metrics/rf_hyperparameter_results.json')
                
                # Print final results
                print("\n" + "="*60)
                print("🏆 HYPERPARAMETER TUNING RESULTS")
                print("="*60)
                print(f"✅ Total successful experiments: {len(results)}/{len(param_combinations)}")
                print(f"🎯 Best Accuracy: {best_result['accuracy']:.4f}")
                print(f"📈 Best ROC AUC: {best_result['roc_auc']:.4f}")
                print(f"⏱️  Best Training Time: {best_result['training_time']:.2f}s")
                print(f"📊 Average Accuracy: {comparison['avg_accuracy']:.4f}")
                print(f"🔄 Accuracy Range: {comparison['accuracy_range']:.4f}")
                
                print(f"\n🏅 Best Parameters:")
                for key, value in best_result['params'].items():
                    print(f"   {key}: {value}")
                
                return results_summary, best_result['model']
            
            else:
                print("❌ No experiments completed successfully")
                return None, None
                
    except Exception as e:
        print(f"❌ Error in Random Forest hyperparameter tuning: {e}")
        return None, None

def plot_parameter_sweep_results():
    """Create plots showing parameter effects on performance"""
    try:
        # Load results
        with open('metrics/rf_hyperparameter_results.json', 'r') as f:
            results = json.load(f)
        
        # Create parameter vs accuracy plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: n_estimators vs accuracy
        n_estimators = [r['parameters']['n_estimators'] for r in results['all_results']]
        accuracies = [r['accuracy'] for r in results['all_results']]
        axes[0, 0].scatter(n_estimators, accuracies, alpha=0.7, s=100)
        axes[0, 0].set_xlabel('Number of Estimators')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('n_estimators vs Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: max_depth vs accuracy
        max_depths = [r['parameters']['max_depth'] if r['parameters']['max_depth'] else 50 for r in results['all_results']]
        axes[0, 1].scatter(max_depths, accuracies, alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Max Depth')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Max Depth vs Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: min_samples_split vs accuracy
        min_samples_splits = [r['parameters']['min_samples_split'] for r in results['all_results']]
        axes[1, 0].scatter(min_samples_splits, accuracies, alpha=0.7, s=100)
        axes[1, 0].set_xlabel('Min Samples Split')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Min Samples Split vs Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: training time vs accuracy
        training_times = [r['training_time'] for r in results['all_results']]
        axes[1, 1].scatter(training_times, accuracies, alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Training Time vs Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = 'rf_parameter_sweep_analysis.png'
        plt.savefig(plot_path)
        plt.close()
        
        print(f"✅ Parameter sweep analysis plot saved: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"❌ Error creating parameter sweep plots: {e}")
        return None

if __name__ == "__main__":
    # Run the hyperparameter tuning experiment
    results, best_model = run_random_forest_experiment()
    
    # Create analysis plots
    if results:
        plot_path = plot_parameter_sweep_results()
        if plot_path:
            print(f"📊 Analysis complete! Check {plot_path} for parameter effects")
        
        print("\n🎉 Hyperparameter tuning completed!")
        print("📈 View detailed results in MLflow UI:")
        print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
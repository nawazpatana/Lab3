import os
import json
import subprocess
import sys
from datetime import datetime
import os
import mlflow
from dagshub import DagsHub

def setup_mlflow():
    """Setup MLflow tracking with DagsHub"""
    
    # DagsHub repository information
    DAGSHUB_USERNAME = "nawazishpatana"
    DAGSHUB_REPO_NAME = "Lab3.mlflow"
    
    # Set up DagsHub connection
    os.environ['MLFLOW_TRACKING_URI'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    
    # Setup DagsHub - this will handle authentication
    DagsHub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    try:
        mlflow.set_experiment("spam-detection-experiment")
        print("Successfully connected to DagsHub MLflow tracking!")
        return True
    except Exception as e:
        print(f"Error setting up DagsHub experiment: {e}")
        return False

def run_logistic_regression_experiment():
    """Main experiment function with DagsHub tracking"""
    
    if not setup_mlflow():
        print("Failed to setup MLflow with DagsHub. Using local tracking as fallback.")
        mlflow.set_tracking_uri("file:///./mlruns")
        mlflow.set_experiment("spam-detection-experiment")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("random_state", 42)


def run_experiment(script_name):
    """Run an experiment script and capture its results"""
    print(f"\n🚀 Running {script_name}...")
    print("="*50)
    
    try:
        # Run the experiment script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True)
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def load_metrics(model_name):
    """Load metrics from JSON file"""
    try:
        metrics_file = f'metrics/{model_name}_metrics.json'
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except:
        return None

def compare_results():
    """Compare results from both experiments"""
    print("\n" + "="*60)
    print("📊 EXPERIMENT COMPARISON")
    print("="*60)
    
    rf_metrics = load_metrics('rf')
    lr_metrics = load_metrics('lr')
    
    if rf_metrics and lr_metrics:
        print(f"Random Forest:")
        print(f"  - Accuracy: {rf_metrics['accuracy']:.4f}")
        print(f"  - ROC AUC: {rf_metrics['roc_auc']:.4f}")
        print(f"  - Training Time: {rf_metrics['training_time']:.2f}s")
        
        print(f"\nLogistic Regression:")
        print(f"  - Accuracy: {lr_metrics['accuracy']:.4f}")
        print(f"  - ROC AUC: {lr_metrics['roc_auc']:.4f}")
        print(f"  - Training Time: {lr_metrics['training_time']:.2f}s")
        
        print(f"\n🏆 Comparison Results:")
        if rf_metrics['accuracy'] > lr_metrics['accuracy']:
            accuracy_diff = rf_metrics['accuracy'] - lr_metrics['accuracy']
            print(f"  ✓ Random Forest is better by {accuracy_diff:.4f} accuracy")
            best_model = "Random Forest"
        elif lr_metrics['accuracy'] > rf_metrics['accuracy']:
            accuracy_diff = lr_metrics['accuracy'] - rf_metrics['accuracy']
            print(f"  ✓ Logistic Regression is better by {accuracy_diff:.4f} accuracy")
            best_model = "Logistic Regression"
        else:
            print("  ⚠ Both models have the same accuracy")
            best_model = "Tie"
        
        if rf_metrics['training_time'] < lr_metrics['training_time']:
            print(f"  ✓ Random Forest is faster to train")
        else:
            print(f"  ✓ Logistic Regression is faster to train")
        
        # Save comparison results
        comparison = {
            'comparison_date': datetime.now().isoformat(),
            'best_model': best_model,
            'random_forest_accuracy': rf_metrics['accuracy'],
            'logistic_regression_accuracy': lr_metrics['accuracy'],
            'random_forest_training_time': rf_metrics['training_time'],
            'logistic_regression_training_time': lr_metrics['training_time']
        }
        
        with open('metrics/comparison_results.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✅ Best Model: {best_model}")
        
    else:
        print("❌ Could not load metrics for comparison")

def main():
    """Main coordinator function"""
    print("🚀 SMS Spam Detection Experiments Coordinator")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    # Run experiments
    rf_success = run_experiment('run_random_forest_experiment.py')
    lr_success = run_experiment('run_logistic_regression_experiment.py')
    
    # Compare results if both succeeded
    if rf_success and lr_success:
        compare_results()
    else:
        print("\n❌ One or both experiments failed")
    
    print("\n" + "="*60)
    print("🎉 Experiments completed!")
    print("📊 View MLflow results with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("🔍 Check individual experiment runs in MLflow UI")
    print("="*60)

if __name__ == "__main__":
    main()
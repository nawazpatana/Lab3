#!/usr/bin/env python3
"""
single_file_mlflow_spam.py

Single-file script that:
- loads /mnt/data/spam.csv (classic SMS spam dataset)
- preprocesses text with TF-IDF
- trains multiple models (LogisticRegression, RandomForest)
- logs params, metrics, artifacts, and models to MLflow
- compares runs (by F1) and loads the best model
- demonstrates inference using the best model

Usage:
    # Default: trains and logs to local MLflow tracking (sqlite in ./mlruns)
    python single_file_mlflow_spam.py

    # To use a remote MLflow tracking server (e.g., DagsHub), set:
    export MLFLOW_TRACKING_URI="https://dagshub.com/<USER>/<REPO>.mlflow"
    export MLFLOW_EXPERIMENT_NAME="mlflow_dagshub_demo"
    python single_file_mlflow_spam.py

Notes:
- Requires: pandas scikit-learn mlflow joblib
    pip install pandas scikit-learn mlflow joblib
- For imbalanced spam detection we use F1 as selection metric.
"""
import os
import argparse
import tempfile
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# CONFIG defaults (can override via env)
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "train.py")
DATA_PATH = os.environ.get("SPAM_CSV_PATH", "data/spam.csv")
RANDOM_STATE = 42

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}")
    # dataset commonly has columns v1 (label) and v2 (text)
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    # drop unnamed extras if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if {'v1','v2'}.issubset(df.columns):
        df = df.rename(columns={'v1':'label','v2':'text'})[['text','label']]
    else:
        # try to guess: first col label, second col text
        df.columns = df.columns[:2].tolist()
        df = df.iloc[:, :2]
        df.columns = ['label','text']
    # map labels to 0/1 if strings
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'ham':0,'spam':1}).astype(int)
    return df

def preprocess_texts(texts, max_features=10000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    return X, vect

def train_and_log(X_train, y_train, X_val, y_val, vect, experiment_name=EXPERIMENT_NAME):
    mlflow.set_experiment(experiment_name)
    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    }

    # ensure artifacts dir
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    vect_path = os.path.join(artifacts_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vect, vect_path)

    results = []
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            # log some params
            mlflow.log_param("model_name", name)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", getattr(model, "n_estimators"))
            mlflow.log_param("random_state", RANDOM_STATE)

            # train
            model.fit(X_train, y_train)

            # eval
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds)
            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds)
            rec = recall_score(y_val, preds)

            # log metrics
            mlflow.log_metric("f1", float(f1))
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("precision", float(prec))
            mlflow.log_metric("recall", float(rec))

            # log vectorizer as artifact for later inference
            mlflow.log_artifact(vect_path, artifact_path="preprocessing")

            # log the model (mlflow native)
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = run.info.run_id
            print(f"Logged run {name} (run_id={run_id}) => f1={f1:.4f}, acc={acc:.4f}")

            results.append({
                "name": name,
                "run_id": run_id,
                "f1": f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            })
    return results

def get_best_run(experiment_name=EXPERIMENT_NAME):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    # search runs ordering by f1 desc
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.f1 DESC"], max_results=100)
    if not runs:
        raise ValueError("No runs found in experiment.")
    best = runs[0]
    # extract metrics safely
    metrics = best.data.metrics
    return {
        "run_id": best.info.run_id,
        "metrics": metrics,
        "params": best.data.params,
        "tags": best.data.tags
    }

def load_model_and_vectorizer(run_id):
    # model is logged under artifact path "model" in our script
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    # vectorizer artifact path
    # use mlflow.artifacts.download_artifacts to get local file
    vect_art_path = f"runs:/{run_id}/preprocessing/tfidf_vectorizer.joblib"
    local_vect_path = mlflow.artifacts.download_artifacts(vect_art_path)
    vect = joblib.load(local_vect_path)
    return model, vect

def demo_inference(model, vect, texts):
    X = vect.transform(texts)
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    return preds, probs

def main(args):
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    print("Using experiment name:", EXPERIMENT_NAME)
    df = load_data(DATA_PATH)
    print("Loaded data shape:", df.shape)
    print("Label distribution:\n", df['label'].value_counts().to_dict())

    X = df['text'].astype(str).values
    y = df['label'].astype(int).values

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocess
    X_train_tfidf, vect = preprocess_texts(X_train_texts, max_features=10000)
    X_test_tfidf = vect.transform(X_test_texts)

    # Train & log
    print("Starting training and MLflow logging...")
    train_results = train_and_log(X_train_tfidf, y_train, X_test_tfidf, y_test, vect, experiment_name=EXPERIMENT_NAME)

    # Compare and pick best
    print("\nSelecting best run from MLflow by F1 metric...")
    best = get_best_run(EXPERIMENT_NAME)
    print("Best run_id:", best["run_id"])
    print("Best metrics:", best["metrics"])

    # Load best model + vectorizer
    model, loaded_vect = load_model_and_vectorizer(best["run_id"])
    print("Loaded best model and vectorizer from run", best["run_id"])

    # Evaluate best model on test set for confirmation
    X_test = loaded_vect.transform(X_test_texts)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    print(f"Best model confirmation on holdout test: f1={f1:.4f}, acc={acc:.4f}")

    # Demo inference on sample texts
    demo_texts = [
        "Free entry in 2 a wkly comp to win FA Cup tickets! Text WIN to 12345",
        "Hi John, can we reschedule our 3pm meeting to 4pm?"
    ]
    preds, probs = demo_inference(model, loaded_vect, demo_texts)
    for t, p, pr in zip(demo_texts, preds, probs if probs is not None else [None]*len(preds)):
        label = "spam" if int(p) == 1 else "ham"
        print(f"\nText: {t}\nPredicted: {label} (prob={pr:.4f} if available)")

    # Optionally save the best model locally (mlflow already saved artifacts in mlruns or remote tracking)
    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Save using joblib for quick local reuse
        local_model_path = os.path.join(out_dir, f"best_model_{best['run_id']}.joblib")
        joblib.dump(model, local_model_path)
        local_vect_path = os.path.join(out_dir, "tfidf_vectorizer.joblib")
        joblib.dump(loaded_vect, local_vect_path)
        print(f"\nSaved best model -> {local_model_path}")
        print(f"Saved vectorizer -> {local_vect_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spam models and log to MLflow (single-file).")
    parser.add_argument("--output-dir", "-o", default="", help="Optional local output directory to save best model & vectorizer.")
    args = parser.parse_args()
    main(args)

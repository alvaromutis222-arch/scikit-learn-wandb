import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier  # <-- tu “otro” modelo
# Puedes cambiar por LogisticRegression, SVC, etc.

import wandb

def parse_args():
    p = argparse.ArgumentParser()
    # Hiperparámetros claves (ajústalos / añade más)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="artifacts")
    return p.parse_args()

def main():
    args = parse_args()

    # Inicia wandb (toma el proyecto del env o usa uno por defecto)
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "mlops-pycon-2023"),
        job_type="train",
        config=vars(args),
    )

    # Dataset de ejemplo (reemplaza por tu fuente real)
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Pipeline: escalado + modelo
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False) if hasattr(X, "sparse") and X.sparse else StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state,
            n_jobs=-1
        )),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = None
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    wandb.log(metrics)

    # Guarda modelo como artifact
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(clf, model_path)

    artifact = wandb.Artifact(
        name="rf-sklearn-model",
        type="model",
        metadata={**metrics, "framework": "scikit-learn", "model": "RandomForestClassifier"},
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    print("Entrenamiento completo. Métricas:", metrics)

if __name__ == "__main__":
    main()

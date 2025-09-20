import argparse, os, joblib
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import wandb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="artifacts")
    # para el mini-loop de logging
    p.add_argument("--log_steps", type=int, nargs="+", default=[50, 100, 200, 300])
    return p.parse_args()

def main():
    args = parse_args()

    # 👉 nombre del proyecto y entidad (define WANDB_ENTITY en secrets para que el workspace del proyecto no salga vacío)
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "mlops-pycon-2023"),
        entity=os.getenv("WANDB_ENTITY", None),  # opcional pero recomendado
        job_type="train",
        name="experiencias-rf",                  # 👉 nombre del run
        tags=["experiencias", "sklearn", "rf"],
        config=vars(args),
    )

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Pipeline + RF con warm_start para registrar múltiples steps
    rf = RandomForestClassifier(
        n_estimators=0,            # empezamos en 0 y vamos sumando
        warm_start=True,           # 👉 nos permite crecer el bosque y medir en cada paso
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_jobs=-1
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", rf),
    ])

    step_idx = 0
    for n in args.log_steps:
        # crece el bosque
        clf.set_params(model__n_estimators=n)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None

        # 👉 registra como time-series (los charts ya “se mueven”)
        wandb.log({**metrics, "n_estimators": n}, step=step_idx)
        step_idx += 1

    # Log extra: matrices y curvas
    if y_prob is not None:
        cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
        wandb.sklearn.plot_confusion_matrix(y_test, (y_prob >= 0.5).astype(int), labels=["neg","pos"])
        wandb.sklearn.plot_roc(y_test, y_prob)
        wandb.sklearn.plot_precision_recall(y_test, y_prob)

    # Guarda modelo final (último n)
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "experiencias_model.joblib")  # 👉 nombre del archivo
    joblib.dump(clf, model_path)

    # 👉 artifact con nombre “experiencias”
    artifact = wandb.Artifact(
        name="experiencias-sklearn-model",
        type="model",
        metadata={
            "framework": "scikit-learn",
            "model": "RandomForestClassifier",
            "variant": "experiencias",
            **metrics
        },
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # resumen final (aparece en la sidebar del run)
    for k, v in metrics.items():
        wandb.run.summary[k] = v

    wandb.finish()
    print("Entrenamiento COMPLETO → modelo: experiencias | métricas:", metrics)

if __name__ == "__main__":
    main()

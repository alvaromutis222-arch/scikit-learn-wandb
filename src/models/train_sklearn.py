import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

import wandb


def parse_args():
    p = argparse.ArgumentParser()
    # Hiperparámetros principales
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="artifacts")
    # Pasos intermedios para hacer logging time-series en W&B
    p.add_argument("--log_steps", type=int, nargs="+", default=[50, 100, 200, 300])
    return p.parse_args()


def main():
    args = parse_args()

    # Inicia el run en W&B (define WANDB_ENTITY si quieres ver el run en tu org/usuario)
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "mlops-pycon-2023"),
        entity=os.getenv("WANDB_ENTITY", None),
        job_type="train",
        name="experiencias-rf",
        tags=["experiencias", "sklearn", "rf"],
        config=vars(args),
    )

    # Dataset demo (cámbialo por tu fuente real cuando quieras)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Modelo: RF con warm_start para crecer el bosque y loguear varias steps
    rf = RandomForestClassifier(
        n_estimators=0,             # empezamos en 0 y vamos sumando
        warm_start=True,            # permite aumentar n_estimators sucesivamente
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_jobs=-1,
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", rf),
    ])

    # Mini-loop para registrar métricas en varios steps (así los charts no quedan con un solo punto)
    last_metrics = {}
    step_idx = 0
    y_prob = None

    for n in args.log_steps:
        clf.set_params(model__n_estimators=n)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        last_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        # Probabilidades (para ROC-AUC y gráficos)
        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
            last_metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None

        # Log como serie temporal
        wandb.log({**last_metrics, "n_estimators": n}, step=step_idx)
        step_idx += 1

    # ----- Plots robustos (evitan error del helper wandb.sklearn) -----
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve

        if (y_prob is not None) and (getattr(y_prob, "ndim", 1) == 1) and (len(y_prob) == len(y_test)):
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1 = plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC")
            wandb.log({"ROC": wandb.Image(fig1)})
            plt.close(fig1)

            # Precision-Recall
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            fig2 = plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall")
            wandb.log({"PR": wandb.Image(fig2)})
            plt.close(fig2)

            # Matriz de confusión (umbral 0.5)
            preds = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y_test, preds)
            fig3 = plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.xticks([0, 1], ["0", "1"])
            plt.yticks([0, 1], ["0", "1"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center")
            wandb.log({"confusion_matrix": wandb.Image(fig3)})
            plt.close(fig3)
    except Exception as e:
        # Si algo falla en los gráficos, no rompas el run
        wandb.log({"plot_error": str(e)})

    # ----- Guardado del modelo y artifact -----
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "experiencias_model.joblib")
    joblib.dump(clf, model_path)

    artifact = wandb.Artifact(
        name="experiencias-sklearn-model",
        type="model",
        metadata={
            "framework": "scikit-learn",
            "model": "RandomForestClassifier",
            "variant": "experiencias",
            **last_metrics,
        },
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    # Resumen final del run en W&B
    for k, v in last_metrics.items():
        wandb.run.summary[k] = v

    wandb.finish()
    print("Entrenamiento COMPLETO → modelo: experiencias | métricas:", last_metrics)


if __name__ == "__main__":
    main()

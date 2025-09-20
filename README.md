# MLOps scikit-learn baseline

Este repo entrena un modelo de scikit-learn (RandomForest) y reporta métricas a Weights & Biases (wandb) usando GitHub Actions.

## Requisitos
- Python 3.11
- Secret `WANDB_API_KEY` configurado en GitHub (Settings → Secrets and variables → Actions).
- Variable/Env `WANDB_PROJECT` (por defecto `mlops-pycon-2023`).

## Entrenamiento local
```bash
python -m pip install -r requirements.txt
export WANDB_API_KEY=<TU_API_KEY>
export WANDB_PROJECT=mlops-pycon-2023
python src/models/train_sklearn.py --n_estimators 300 --test_size 0.2

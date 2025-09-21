# MLOps scikit-learn baseline 🚀

## 👥 Integrantes
* Jose Luis Bedoya Martinez
* Alvaro Javier Mutis Guerrero

Este proyecto es una plantilla de **MLOps (Machine Learning Operations)** que demuestra un pipeline básico para el ciclo de vida de un modelo. Incluye el entrenamiento de un modelo de machine learning con **scikit-learn** y la integración de **GitHub Actions** para automatizar el proceso, con un seguimiento de experimentos en **Weights & Biases**.

### ✨ Características Principales
* **Entrenamiento Automatizado**: Un **pipeline CI/CD** que entrena el modelo automáticamente en cada `push` a la rama principal.
* **Seguimiento de Experimentos**: Registra métricas, parámetros y visualizaciones en **Weights & Biases (wandb)** para un análisis completo de cada experimento.
* **Modelo de Referencia**: Usa un modelo **RandomForestClassifier** para el problema de clasificación del dataset Breast Cancer, con parámetros ajustables.
* **Configuración Sencilla**: Requiere solo la configuración de una clave de API en los secretos de GitHub para funcionar.

---

## 🛠️ Tecnologías Clave

Este proyecto se basa en un stack de herramientas para la gestión del ciclo de vida del machine learning.

* **Python 3.11**: El lenguaje de programación principal.
* **scikit-learn**: Una de las bibliotecas más populares para machine learning en Python. Se utiliza para el entrenamiento, la evaluación y la persistencia del modelo.
* **Weights & Biases (W&B)**: Una plataforma de seguimiento de experimentos que te permite visualizar, comparar y depurar modelos de machine learning. Es esencial para llevar un registro de los resultados y los hiperparámetros de cada ejecución.
* **GitHub Actions**: Una herramienta de integración y despliegue continuos (CI/CD) de GitHub. Automatiza el proceso de entrenamiento del modelo cada vez que se realiza un cambio en el código, asegurando que el modelo esté siempre actualizado.

---

## 🚀 Cómo Empezar

Sigue estos pasos para ejecutar el pipeline de entrenamiento de forma local o automática.

### Requisitos

Asegúrate de tener instalado **Python 3.11**. Además, necesitas configurar una variable de entorno y un secreto en tu repositorio para poder interactuar con Weights & Biases.

1.  **Obtener tu W&B API Key**: Inicia sesión en tu cuenta de [Weights & Biases](https://wandb.ai/) y ve a la configuración de tu perfil para encontrar tu **API Key**.
2.  **Configurar Secret en GitHub**: En la configuración de tu repositorio, ve a `Settings` > `Secrets and variables` > `Actions` y crea un nuevo `repository secret` llamado `WANDB_API_KEY` con el valor de tu clave.

### 💻 Ejecución Local

1.  **Clonar el repositorio**:
    ```bash
    git clone [https://github.com/](https://github.com/)<usuario>/<repo>.git
    cd <repo>
    ```
2.  **Crear e instalar dependencias**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Configurar variables de entorno**:
    ```bash
    # En macOS/Linux
    export WANDB_API_KEY=<TU_API_KEY>
    export WANDB_PROJECT=mlops-pycon-2023

    # En Windows (CMD)
    set WANDB_API_KEY=<TU_API_KEY>
    set WANDB_PROJECT=mlops-pycon-2023
    ```
    (Opcional: Si omites `WANDB_PROJECT`, se usará el valor por defecto).
4.  **Ejecutar el entrenamiento**:
    ```bash
    python src/models/train_sklearn.py --n_estimators 300 --test_size 0.2
    ```

### ⚡ Automatización con GitHub Actions

El flujo de trabajo (`.github/workflows/train.yml`) está configurado para ejecutarse automáticamente cada vez que se realiza un `push` a la rama principal (`main`) o a través de un disparo manual desde la interfaz de GitHub.

* **Activación**: Se activa con `push` en los directorios `src/` y `.github/`, o manualmente (`workflow_dispatch`).
* **Proceso**: Instala las dependencias y ejecuta el script de entrenamiento con los parámetros predefinidos.
* **Resultados**: Puedes seguir el progreso en la pestaña **`Actions`** de tu repositorio y ver los experimentos en tu dashboard de **Weights & Biases**.

---

## 📊 Visualización de Resultados

Todos los experimentos (locales y de GitHub Actions) se registran en tu cuenta de Weights & Biases.
* **URL**: [https://wandb.ai/](https://wandb.ai/)
* **Proyecto**: `mlops-pycon-2023` 

En la interfaz de W&B, podrás:
* Comparar métricas como la precisión, F1-score y ROC-AUC.
* Visualizar gráficos como la matriz de confusión y las curvas ROC.
* Examinar los hiperparámetros de cada `run`.
* Guardar y versionar el modelo final como un **Artifact**.

---

## 📂 Estructura del Proyecto

```

├── .github/workflows/     # Configuración de los workflows de GitHub Actions
│   └── train.yml
├── src/
│   └── models/
│       └── train_sklearn.py # Script de entrenamiento del modelo
├── artifacts/             # Directorio para guardar el modelo y otros resultados
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación del proyecto
```

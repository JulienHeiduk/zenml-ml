from zenml import step
from sklearn.ensemble import RandomForestClassifier
from typing import Annotated
import numpy as np
import mlflow

@step(enable_cache=False)
def train_model(
    X_train: Annotated[np.ndarray, "X_train"],
    y_train: Annotated[np.ndarray, "y_train"]
) -> RandomForestClassifier:
    """Entraîner un modèle RandomForest sur les données d'entraînement."""
    model = RandomForestClassifier()
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
    return model

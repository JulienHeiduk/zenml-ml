from zenml import step
from sklearn.metrics import accuracy_score
import logging
from typing import Annotated
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mlflow

@step(enable_cache=False)
def evaluate_model(
    model: RandomForestClassifier,
    X_test: Annotated[np.ndarray, "X_test"],
    y_test: Annotated[np.ndarray, "y_test"]
) -> None:
    """Évaluer le modèle entraîné sur les données de test."""
    logging.info("Evaluating the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
    logging.info(f"Model accuracy: {accuracy}")
    print(f"Accuracy: {accuracy}")
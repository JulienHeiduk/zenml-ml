from zenml import step
from sklearn.ensemble import RandomForestClassifier
from typing import Annotated
import numpy as np

@step(enable_cache=False)
def train_model(
    X_train: Annotated[np.ndarray, "X_train"],
    y_train: Annotated[np.ndarray, "y_train"]
) -> RandomForestClassifier:
    """Entraîner un modèle RandomForest sur les données d'entraînement."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
from zenml import step
from typing import Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from typing import Annotated

@step
def load_data() -> Tuple[
    # Utilisation de Tuple et Annotated pour nommer les sorties multiples
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """Charger le dataset Iris et le diviser en données d'entraînement et de test."""
    logging.info("Loading iris dataset...")
    data = load_iris()
    logging.info("Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

from zenml import step
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@step(enable_cache=False)
def inference_step(model: RandomForestClassifier, X_test: np.ndarray) -> np.ndarray:
    """Inference on test dataset."""
    predictions = model.predict(X_test)
    return predictions

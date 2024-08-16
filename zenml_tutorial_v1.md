# How to Integrate MLflow into a ZenML Pipeline

In this tutorial, we'll focus on adding MLflow to a ZenML pipeline for easy tracking and monitoring of your machine learning experiments. MLflow is a powerful tool that provides functionalities for logging parameters, metrics, and models, making it easier to manage the lifecycle of machine learning projects.

## Prerequisites

Before we get started, ensure that you have the following tools and libraries installed:

- [ZenML](https://zenml.io/)
- [MLflow](https://mlflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- A coding environment with Python

## Setting Up Your ZenML Pipeline

Let's define a pipeline that includes the training and evaluation of a model using the `RandomForestClassifier`, while also tracking relevant metrics and parameters using MLflow. Below, we'll detail the pipeline definition with the included MLflow integration.

### 1. Pipeline Definition (`pipelines/training_pipeline.py`)

We'll create two steps in our pipeline: `train_model` for training the model, and `evaluate_model` for evaluating the model's performance.

#### Code for Training Step

```python
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
    """Train a RandomForest model on the training data."""
    model = RandomForestClassifier()
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
    return model
```

In this step:
- We define the `train_model` function, which takes `X_train` and `y_train` as inputs.
- A `RandomForestClassifier` is instantiated and fitted to the training data.
- MLflow's `start_run()` context is used to log the parameters `n_estimators` and `max_depth` of the model once it is trained.

#### Code for Evaluation Step

```python
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
    """Evaluate the trained model on the test data."""
    logging.info("Evaluating the model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
    logging.info(f"Model accuracy: {accuracy}")
    print(f"Accuracy: {accuracy}")
```

In the evaluation step:
- We define the `evaluate_model` function that accepts the trained model and test data.
- After logging the start of the evaluation, the model's predictions are generated, and the accuracy is computed using `accuracy_score`.
- The obtained accuracy is logged to MLflow within a new run context.

### Summary

You have successfully integrated MLflow into your ZenML pipeline. This setup allows you to track important parameters and metrics throughout your machine learning workflow. As you experiment with different model configurations, the logs can help you compare and choose the best-performing models.

### Next Steps

1. **Run Your Pipeline**: Create a ZenML pipeline that utilizes the defined steps and executes the training and evaluation.
2. **Explore MLflow UI**: Use the MLflow tracking UI to visualize the logged parameters and metrics.

Feel free to modify the pipeline steps as per your requirements and explore more functionality that MLflow offers for enhancing your machine learning projects. Happy experimenting with ZenML and MLflow!
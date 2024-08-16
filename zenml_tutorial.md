# Tutorial: Building a Data Pipeline with ZenML

In this tutorial, we will walk through the steps to create a data pipeline using ZenML. ZenML is a powerful framework that streamlines the process of building reproducible and scalable machine learning workflows. We will utilize the Iris dataset to demonstrate loading data, training a model, and evaluating its performance.

## Prerequisites

Before we begin, ensure you have the following installed:

- Python 3.7 or higher
- ZenML
- scikit-learn
- Any text editor or IDE of your choice

You can install ZenML and scikit-learn using pip:

```bash
pip install zenml scikit-learn
```

## Overview of the Pipeline

The pipeline will consist of three main steps:

1. **Load Data**: Load the Iris dataset and split it into training and testing sets.
2. **Train Model**: Train a Random Forest classifier on the training data.
3. **Evaluate Model**: Evaluate the model using the test data and print out the accuracy.

### Step 1: Create the Load Data Step

First, we will create a step to load the Iris dataset. Create a new file named `load_data_step.py` in the `steps` directory and add the following code:

```python
from zenml import step
from typing import Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import logging
from typing import Annotated

@step
def load_data() -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """Load the Iris dataset and split it into train and test sets."""
    logging.info("Loading iris dataset...")
    data = load_iris()
    logging.info("Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
```

### Step 2: Create the Train Model Step

Next, we will define a step to train our model. Create a new file named `train_model_step.py` in the `steps` directory and include the following code:

```python
from zenml import step
from sklearn.ensemble import RandomForestClassifier
from typing import Annotated
import numpy as np

@step(enable_cache=False)
def train_model(
    X_train: Annotated[np.ndarray, "X_train"],
    y_train: Annotated[np.ndarray, "y_train"]
) -> RandomForestClassifier:
    """Train a RandomForest model on the training data."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
```

### Step 3: Create the Evaluate Model Step

Now, we will create the final step for model evaluation. Create a file named `evaluate_model_step.py` in the `steps` directory with the following content:

```python
from zenml import step
from sklearn.metrics import accuracy_score
import logging
from typing import Annotated
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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
    logging.info(f"Model accuracy: {accuracy}")
    print(f"Accuracy: {accuracy}")
```

### Step 4: Define the Pipeline

Next, we will define our pipeline. Create a file named `training_pipeline.py` in the `pipelines` directory and add the following code:

```python
import sys
import os
from zenml import pipeline

# Add the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.load_data_step import load_data
from steps.train_model_step import train_model
from steps.evaluate_model_step import evaluate_model

@pipeline(name="Tutorial", enable_cache=False)
def training_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    training_pipeline()
```

### Step 5: Run the Pipeline

Now that you have defined all the components, it’s time to run the pipeline. From the command line, navigate to the directory containing your `training_pipeline.py` file and execute:

```bash
python -m pipelines.training_pipeline
```

You should see logs indicating the loading of the Iris dataset, the training process, and the final accuracy of the model.

### Conclusion

Congratulations! You have successfully built and executed a simple data pipeline using ZenML. You can extend this example further by experimenting with different models, hyperparameters, or datasets. Happy coding!
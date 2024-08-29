# ZenML Pipeline Tutorial: Understanding Artefacts and Adding Inference Steps

In this tutorial, we will dive deep into the concept of artefacts in ZenML pipelines. We’ll explain how to define and use them, with a focus on the newly added inference step in an existing pipeline. 

## What is an Artefact in ZenML?

In ZenML, artefacts represent the output of processing steps within a pipeline. These outputs can be various formats such as datasets, models, metrics, or predictions. Artefacts allow you to track these outputs and reuse them across different workflows. By managing artefacts effectively, you can facilitate reproducibility, versioning, and deployment of your machine learning models.

## Adding an Inference Step to the Pipeline

We recently updated our pipeline to include an inference step, where the trained model is used to make predictions on the test dataset. Below is the updated definition of our training pipeline, along with the inference step:

### 1. Pipeline Definition (`pipelines/training_pipeline.py`)

This code defines our updated pipeline, where we integrate the inference step.

```python
import sys
import os
from zenml import pipeline

# Adding the parent directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.load_data_step import load_data
from steps.train_model_step import train_model
from steps.evaluate_model_step import evaluate_model
from steps.inference_step import inference_step

@pipeline(name="Tutorial", enable_cache=False)
def training_pipeline():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Inference
    predictions = inference_step(model, X_test)

if __name__ == "__main__":
    training_pipeline()
```

### 2. Inference Step (`steps/inference_step.py`)

This code snippet shows how we define the inference step, which takes the trained model and makes predictions on the test dataset.

```python
from zenml import step
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@step(enable_cache=False)
def inference_step(model: RandomForestClassifier, X_test: np.ndarray) -> np.ndarray:
    """Inference on test dataset."""
    predictions = model.predict(X_test)
    return predictions
```

## Loading Artefacts

Once the inference step has been executed, it generates artefacts (i.e., the predictions). To work with these artefacts later on, we can use the following script to load them.

### Load Artefact Script (`load_artefact.py`)

This snippet demonstrates how to load a specific artefact from the ZenML client using its unique identifier.

```python
from zenml.client import Client

# Replace with your artefact UUID
artifact = Client().get_artifact_version('4f12d004-1a1f-453f-9321-a4da200345d4')
loaded_artifact = artifact.load()

print(loaded_artifact)
```

### Explanation:
- **Client()**: This initializes the ZenML client that allows interaction with your ZenML environment.
- **get_artifact_version()**: We retrieve a specific version of the artefact using its unique identifier.
- **load()**: This method loads the artefact into memory, allowing you to work with it directly.

## Conclusion

In this tutorial, we learned about the significance of artefacts in ZenML pipelines and how to integrate an inference step into an existing training pipeline. We also explained how to load these artefacts for further analysis or deployment. 

To give more context on the previous steps in the pipeline, feel free to check out the article [here](https://jheiduk.com/posts/zenml_tutorial/). 

With this knowledge, you should be well-equipped to utilize artefacts in your ZenML workflows effectively. Happy coding!
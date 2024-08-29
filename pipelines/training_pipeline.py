import sys
import os
from zenml import pipeline

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steps.load_data_step import load_data
from steps.train_model_step import train_model
from steps.evaluate_model_step import evaluate_model
from steps.inference_step import inference_step

@pipeline (name="Tutorial", enable_cache=False)
def training_pipeline():
    # Charger les données
    X_train, X_test, y_train, y_test = load_data()
    
    # Entraîner le modèle
    model = train_model(X_train, y_train)
    
    # Évaluer le modèle
    evaluate_model(model, X_test, y_test)
    
    # Inférence
    predictions = inference_step(model, X_test)

if __name__ == "__main__":
    training_pipeline()
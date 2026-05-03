import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
import mlflow
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import train_test_split
from urllib.parse import urlparse


os.environ['ML_FLOW_TRACKING_URI'] ="https://dagshub.com/Praveenravva61/Ml_pipeline.mlflow"
os.environ['ML_FLOW_TRACKING_USER'] ="Praveenravva61"
os.environ['ML_FLOW_TRACKING_PASSWORD'] ="3a388ae629ba4c064e33d420a46d2faa874f841f"


# load parameter from yaml file
params= yaml.safe_load(open("params.yaml"))["train"]


def evaluate(model_path, data_path):
    data= pd.read_csv(data_path)
    x = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/Praveenravva61/Ml_pipeline.mlflow")
    
    model= pickle.load(open(model_path, "rb"))
    y_pred = model.predict(x) 
    Accuracy = accuracy_score(y, y_pred)
    print("Accuracy:", Accuracy)
    
    
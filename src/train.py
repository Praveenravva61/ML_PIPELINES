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

def hyperparameter_tunning(x_train,y_train, param_grid):
    rf= RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search

# load parameter from yaml file
params= yaml.safe_load(open("params.yaml"))["train"]


def train(data_path, model_path, random_state, max_depth, n_estimators):
    data= pd.read_csv(data_path)    
    x = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/Praveenravva61/Ml_pipeline.mlflow")
    with mlflow.start_run():
        # train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
        signature = infer_signature(x_train, y_train)
        
        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [5,10,None],
            "min_samples_split": [2,5],
            "min_samples_leaf": [1,2]
            
        }
        
        # hyperparameter tunning of model
        model = hyperparameter_tunning(x_train,y_train, param_grid)
        best_model = model.best_estimator_
        
        # predict the model
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        
        
        # log the confusion matrix and classification report
        
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", model.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", model.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split", model.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", model.best_params_['min_samples_leaf'])

        mlflow.log_text(str(cm), "model/confusion_matrix.txt")
        mlflow.log_text(cr, "model/classification_report.txt")

        # ✅ Single log_model call — capture return value for URI
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                signature=signature
            )
            print("✅ Model logged successfully")

            # ✅ Use model_uri directly from model_info (no hardcoding)
            mlflow.register_model(
                model_uri=model_info.model_uri,
                name="RandomForestBestModel"
            )
            print("✅ Model registered successfully")

        except Exception as e:
            print(f"❌ Error: {e}")

        # Save locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, "wb"))
        print(f"Model saved to {model_path}")
        
        
if __name__ == "__main__":
    train(params["data"], params["model"], params["random_state"], params["max_depth"], params["n_estimators"])
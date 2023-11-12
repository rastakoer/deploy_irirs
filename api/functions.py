import joblib
import numpy as np
from pydantic import BaseModel
import os
import mlflow
import boto3
import pandas as pd


s3_key = os.environ.get('AWS_ACCESS_KEY_ID')
s3_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')

# Recupértation du modèle sur MlFlow
mlflow.set_tracking_uri('https://isen-mlflow-fae8e0578f2f.herokuapp.com/')
logged_model = 'runs:/19059cfeabee4e7ebd8984f89cf8a631/Iris'
model = mlflow.sklearn.load_model(logged_model)

# # importer les scalers
scalers = joblib.load("scalers")

# Configuration d'une classe BaseModel pour s'assurer que les 
# données correspondent bien avec ce qui est attendu
class Config_donnees(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

class reponse_model(BaseModel):
    reponse:str


def scal(n:dict) ->list:
    """
    Fonction servant à standardiser les donnees
    Entrée json
    Sortie de type : [0.12,0.55,0.56,0.2]
    """
    transformed_data=[]
    transformed_data.append(scalers['sepal_length'].transform(np.array([n.sepal_length]).reshape(-1, 1))[0][0])
    transformed_data.append(scalers['sepal_width'].transform(np.array([n.sepal_width]).reshape(-1, 1))[0][0])
    transformed_data.append(scalers['petal_length'].transform(np.array([n.petal_length]).reshape(-1, 1))[0][0])
    transformed_data.append(scalers['petal_width'].transform(np.array([n.petal_width]).reshape(-1, 1))[0][0])
    return transformed_data

def predictions(data:list) -> dict:
    """
    Fonction permettant la prédiction 
    Sortie de type : {'reponse':'Versivolor']}
    """
    # Prédiction en utilisant le modèle
    data = np.array([data])
    pred = model.predict(data)
    # Renvoi du dictionnaire contenant la prédiction
    return {'reponse':pred[0]}


import os
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from functions import train_model
import pandas as pd

# Crédentials d'accès à AWS
os.environ['AWS_ACCESS_KEY_ID'] = "AKIA3R62MVALHESATEYJ"
os.environ['AWS_SECRET_ACCESS_KEY'] = "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"
os.environ['ARTIFACT_STORE_URI'] = "s3://isen-mlflow/models/"
os.environ['BACKEND_STORE_URI'] = "postgresql://eagbhergisskna:6e299604b7204f81d625807348dd55dd6d33d426eb2d33762b54c1dcf7367112@ec2-3-214-103-146.compute-1.amazonaws.com:5432/d9ov3338s1olla"

# Connexion à MLflow
mlflow.set_tracking_uri("https://isen-mlflow-fae8e0578f2f.herokuapp.com/")

# Configuration de l'autolog
mlflow.sklearn.autolog()
mlflow.set_experiment("kevin-experiment")


data = pd.read_csv("./iris.csv", delimiter=',')
data.rename(columns=lambda x: x.replace('.', '_'), inplace=True)
Xtr, Xte, ytr, yte= train_test_split(data.iloc[:,:-1], data.variety , test_size=0.33, random_state=42)
model, _, X_train, y_train = train_model(Xtr,ytr)

# Connexion à une expérience
experiment = mlflow.get_experiment_by_name("kevin-experiment")

with mlflow.start_run(experiment_id = experiment.experiment_id, run_name='Training_iris'):

    model_name = "KNC"
    mlflow.log_metric("train score", model.score(X_train, y_train))
    mlflow.sklearn.log_model(model,
                             "Iris",
                             input_example = X_train.head(1),
                             registered_model_name = "KNC")
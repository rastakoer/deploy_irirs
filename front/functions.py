import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def get_data(csv_path: str):
    """
    Récupération des données depuis un csv dans un dataframe puis séparation
    des données en un jeu d'entrainement et un jeu de test
    """
    data = pd.read_csv(csv_path, delimiter=',')
    Xtr, Xte, ytr, yte= train_test_split(data.iloc[:,:-1], data.variety , test_size=0.33, random_state=42)
    
    return Xtr, Xte, ytr, yte



def standardize_labelize(df: pd.DataFrame):
    """
    A partir d'un dataframe donné, labélise les colonnes catégoriques et standardise l'ensemble des features.
    Retourne le dataframe modifié et des dictionnaires contenant les encoders et scalers utilisés.
    """
    scalers = {}
    colonnes = df.columns.tolist()
    for col in colonnes:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler
    joblib.dump(scalers,"scalers")
    return df, scalers

def train_model(df: pd.DataFrame, sortie: pd.Series):
    """
    Entrainement du modèle
    """
    
    df, scalers = standardize_labelize(df)
    print(df)
    X = df
    y = sortie
    model = KNeighborsClassifier()
    model.fit(X, y)
    return model, scalers,  X, y

def test_model(df: pd.DataFrame,output:pd.Series, model: KNeighborsClassifier, scalers: dict):
    """
    Test du modèle
    """
    
    for col, scaler in scalers.items():
        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
    X = df
    y = output
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    confusion_matrix_result = confusion_matrix(y, y_pred)
    classification_report_result = classification_report(y, y_pred)
    print(f"Accuracy : {accuracy}")
    print(f"Confusion Matrix :\n{confusion_matrix_result}")
    print(f"Classification Report :\n{classification_report_result}")

if __name__ == "__main__":
    Xtr, Xte, ytr, yte = get_data("./iris.csv")
    model, scalers, _, _ = train_model(Xtr,ytr)
    test_model(Xte,yte, model, scalers)
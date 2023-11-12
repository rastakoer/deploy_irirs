from fastapi import FastAPI
import uvicorn
import functions

# Config apparence API
descritpion ="""
    Obtenez une prédiction de la variété de fleurs d'iris en fonction des paramètres suivant :
    - 
    """
app = FastAPI(
    title="API prédictions fleurs d'iris",
    summary="Api développée par Kevin LE GRAND",
    description=descritpion
)


# Définir une route POST pour la commande
@app.post("/predict", response_model=functions.reponse_model, summary="Prédictions")
def predict(n:functions.Config_donnees):
    """
    ## La réponse est de type json : {'reponse':'Versicolore',}
    """
    print(1)
    # Appel de la fonction servant à standardiser
    transform = functions.scal(n)
    
    # Appel de la fonction servant à réaliser les prédictions
    prediction= functions.predictions(transform)
    print(5)
    # Réponse au format json
    return prediction


if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=4000)

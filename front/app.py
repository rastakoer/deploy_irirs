import streamlit as st
import pandas as pd
import requests

# ---------------------------------------------------------------------
# Config streamlit
#-----------------------------------------------------------------------------------
st.set_page_config(page_title="Predict heart", page_icon=":tada:", layout="wide")



st.sidebar.header("Les parametres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('La longeur du Sepal',4.3,7.9,5.3)
    sepal_width=st.sidebar.slider('La largeur du Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longeur du Petal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'sepal_length': sepal_length, 'sepal_width': sepal_width,
      'petal_length': petal_length, 'petal_width': petal_width}
    
    return data

param= user_input()

st.subheader('Les caractéristiques de l\'iris recherché')
st.write(f"longueur des sépales: {param['sepal_length']}")
st.write(f"largeur des sépales: {param['sepal_width']}")
st.write(f"longueur des pétales: {param['petal_length']}")
st.write(f"largeur des pétales: {param['petal_width']}")

# ---------------------------------------------------------------------
#                       Appel de l'api
# ---------------------------------------------------------------------
if st.button("Prédictions"):
  response = requests.post('https://apikeviniris-4bcab722423e.herokuapp.com/predict', json=param)
  st.subheader(f"La catégorie de la fleur d'iris est : {response.json()['reponse']}")

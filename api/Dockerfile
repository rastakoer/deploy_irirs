#-----------------------------------------------------------------
#    Image docker qui sera deployer depuis le repo github sur heroku
#-----------------------------------------------------------------

# # Utilisez l'image continuumio/miniconda3 comme image de base
# FROM continuumio/miniconda3

# # Définissez le répertoire de travail
# WORKDIR /home/app

# # Recuperation du repo
# RUN git clone  https://github.com/rastakoer/deploy_irirs .
 
# # Copiez les fichiers nécessaires
# COPY requirements.txt .

# # Installez les dépendances
# RUN pip install -r requirements.txt

# COPY . .

# # Exécutez votre application
# CMD uvicorn app:app --host=0.0.0.0 --port=$PORT


#-----------------------------------------------------------------
#    Image docker qui sera deployer à l'aide de push.sh sur heroku
#-----------------------------------------------------------------

FROM continuumio/miniconda3

WORKDIR /home/app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD uvicorn app:app --host=0.0.0.0 --port=$PORT 

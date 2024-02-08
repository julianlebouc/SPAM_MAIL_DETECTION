# Spam Mail Detection with MLFlow
## Julian LEBOUC, Nicolas TACHET, Abel COUTOLLEAU, Andy HARAN
Ce projet s'inscrit dans le cadre du module 'Machine Learning Operations' du Master 2 Informatique - IA 2023/2024 de l'université du Mans.
Le but de ce dernier est de réaliser un projet de machine learning end-to-end en utilisant MLFlow
Nous avons décidé de mettre un place un modèle prédisant la nature d'un email, à savoir : spam ou non spam
Il est possible de modifier les données utilisées ainsi que les paramètres du modèle dans les fichiers de configuration.
Les paramètres, modèles, métriques et environnements sont loggés via MLFlow Tracking.
Une interface WEB a été mise en place pour tester le modèle.


# How to run?

Clone the repository

```bash
https://github.com/julianlebouc/SPAM_MAIL_DETECTION
```

## Option 1 : With MLFlow Projects
Please be aware some dependencies might be missing !
### Training Pipeline :
```bash
mlflow run path/to/git/repo -e main
```

### App testing :
```bash
mlflow run path/to/git/repo -e app
```

## Option 2 : Manual install and use

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n spamdetection python=3.8 -y
```

```bash
conda activate spamdetection
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Training pipeline
```bash
python main.py
```

### STEP 04- MLFlow Tracking
```bash
mlflow ui
```

### STEP 05- App Testing
```bash
python app.py
```

## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

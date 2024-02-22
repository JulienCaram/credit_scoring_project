# Implémentez un modèle de scoring

## Description du projet
Ce projet vise à développer et déployer un modèle de scoring pour prédire la probabilité de défaut de paiement d'un client.  
Il inclut le prétraitement des données, la modélisation, une API pour l'accès au modèle, et une configuration CI/CD pour l'intégration et le déploiement automatiques sur Heroku.

## Mission
- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.
- Mettre en oeuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

## Structure du projet
- **notebook_pretraitement_modelisation.ipynb** : Notebook Jupyter contenant le prétraitement des données et la modélisation.
- **api.py** : Fichier Python contenant l'API Flask pour servir les prédictions du modèle.
- **/test_process.csv** : Fichier de données prétraitées utilisé pour tester le modèle.
- **requirements.txt** : Liste des dépendances Python nécessaires pour exécuter le projet.
- **Procfile** : Fichier utilisé par Heroku pour le déploiement de l'application.
- **best_model.pickle** : Le modèle de machine learning entraîné, sérialisé avec pickle.
- **.github/workflows/ci-cd.yml** : Configuration pour l'intégration continue et le déploiement continu avec GitHub Actions.


# Auteurs
## Caramanno Julien

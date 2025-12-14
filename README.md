# Résumé du Projet
Ce projet s'inscrit dans le cadre d'une mission pour une société financière spécialisée dans les crédits à la consommation pour des clients ayant peu ou pas d'historique de prêt. L'objectif principal est de développer un outil complet de Credit Scoring capable de prédire automatiquement la probabilité de faillite d'un client et de classifier chaque demande en crédit accordé ou refusé.

## Objectifs Métiers et Contraintes
L'enjeu central est de construire un modèle performant tout en répondant à deux contraintes métier majeures :

- #### Le déséquilibre des classes : 
Le jeu de données présente une forte disproportion entre les bons et les mauvais payeurs, nécessitant des techniques de rééquilibrage adaptées
- #### L'asymétrie des coûts (Coût FN vs FP) :
Un Faux Négatif (FN) (accorder un crédit à un client qui ne payera pas) est considéré comme beaucoup plus coûteux qu'un Faux Positif (FP) (refuser un crédit à un bon client).L'hypothèse métier retenue est qu'un FN coûte environ 10 fois plus cher qu'un FP ($FN \approx 10 \times FP$)

### Objectif: Optimisation du Seuil de Décision
En raison de cette asymétrie, l'utilisation du seuil de classification standard (0.5) n'est pas pertinente. Une fonction de coût métier personnalisée a été définie pour pénaliser fortement les Faux Négatifs. L'objectif final est d'optimiser le seuil de décision pour minimiser ce coût total, garantissant ainsi une rentabilité optimale pour l'institution financière tout en maîtrisant le risque.


# Lancement du projet
**Assurez-vous d'être à la racine du projet.**

Installation des dépendances :
```Bash
pip instaLL -r requirements.txt
```

Construction de l'image docker : 
```Bash
docker build -t credit-scoring-model .
```
Lancement du conteneur :
```Bash
docker run -p 1234:1234 credit-scoring-model
```

# Seuil métier
Le seuil métier optimal est le point d'équilibre précis qui permet à la banque de maximiser sa rentabilité. Ce seuil est la barrière que l'on fixe pour prendre la décision (probabilité > Seuil $\rightarrow$ Refus | probabilité < Seuil $\rightarrow$ Accord).Dans notre notebook 02_model_training.ipynb, nous avons déterminé que le modèle offrant le meilleur compromis est LightGBM (une fois optimisé). Grâce à lui, nous obtenons un seuil optimal de 0,51 pour un coût métier de 29 761. Par conséquent, si un client a une probabilité de défaut de 47 %, il doit être refusé, alors que s'il a une probabilité de 45 %, il doit être accepté.

# Structure du projet
PROJET-CREDIT-SCORING/
│
├── 📂 src/                          # LE MOTEUR (Code source modulaire)
│   ├── data_prep.py                # Pipeline de transformation des données brutes
│   ├── model_utils.py              # Logique d'entraînement, Cross-Val et Optuna
│   ├── metrics.py                  # Définition mathématique du coût métier
│   └── explainability.py           # Moteur d'interprétabilité (SHAP)
│
├── 📂 notebooks/                    # LES EXPÉRIENCES (Notebooks)
│   ├── 01_data_preparation.ipynb   # Exécution du pipeline de nettoyage
│   ├── 02_model_training.ipynb     # Orchestration des entraînements et MLflow
│   ├── 03_explainability.ipynb     # Analyse des décisions du modèle
│   └── 04_mlflow_serving_test.ipynb # Simulation client / test API
│
├── 📂 model/                        # Artefact final
├── 📂 mlruns/                       # Base de données de tracking (Logs)
├── Dockerfile                      # Fichier de mise en place Docker
└── requirements.txt                # Liste des dépendances (pip)




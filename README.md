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

Test de l'API :
```Bash
curl -X POST -H "Content-Type: application/json" \
--data '{"dataframe_split": {"columns": ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"], "data": [[100002, "Cash loans", 202500.0, 406597.5, 0.083, 0.262, 0.139]]}}' \
http://localhost:1234/invocations
```


# Seuil métier
Le seuil métier optimal est le point d'équilibre précis qui permet à la banque de maximiser sa rentabilité. Ce seuil est la barrière que l'on fixe pour prendre la décision (probabilité > Seuil $\rightarrow$ Refus | probabilité < Seuil $\rightarrow$ Accord).Dans notre notebook 02_model_training.ipynb, nous avons déterminé que le modèle offrant le meilleur compromis est LightGBM (une fois optimisé). Grâce à lui, nous obtenons un seuil optimal de 0,46 pour un coût métier de 29 761. Par conséquent, si un client a une probabilité de défaut de 47 %, il doit être refusé, alors que s'il a une probabilité de 45 %, il doit être accepté.

# Structure du projet
PROJET-CREDIT-SCORING/
│
├── 📂 src/                          # LE MOTEUR (Code source modulaire)
│   ├── data_prep.py                # Pipeline de transformation des données brutes
│   ├── model_utils.py              # Logique d'entraînement, Cross-Val et Optuna
│   ├── metrics.py                  # Définition mathématique du coût métier
│   └── explainability.py           # Moteur d'interprétabilité (SHAP)
│
├── 📂 notebooks/                    # LES EXPÉRIENCES (Interactive)
│   ├── 01_data_preparation.ipynb   # Exécution du pipeline de nettoyage
│   ├── 02_model_training.ipynb     # Orchestration des entraînements et MLflow
│   ├── 03_explainability.ipynb     # Analyse des décisions du modèle
│   └── 04_mlflow_serving_test.ipynb # Simulation client / test API
│
├── 📂 model/                        # Artefact final prêt pour la prod
├── 📂 mlruns/                       # Base de données de tracking (Logs)
├── Dockerfile                      # Recette de conteneurisation
└── requirements.txt                # Liste des dépendances (pip)


Fichier, Fonctionnalités Clés & Choix Techniques

data_prep.py,
• load_dataframe : Charge les CSV avec optimisation mémoire automatique.
• load_and_process_all_data : Orchestrateur principal. Charge, nettoie et fusionne toutes les tables (Bureau, Previous, POS_CASH, etc.) en une seule étape.
• reduce_mem_usage : Optimisation technique critique (downcast des types float64/int64) réduisant l'empreinte mémoire de ~50%.
• aggregate_client : Stratégie d'agrégation complexe (Niveau 3 $\to$ 2 $\to$ 1) pour remonter tout l'historique bancaire au niveau du client unique.
• engineer_domain_features : Création de ratios financiers métiers (ex: Credit/Income, Annuity/Income) pour enrichir le modèle.

metrics.py,
• custom_business_cost : Implémente la formule Cost=10×FN+1×FP.
• get_metrics : Calcule simultanément l'AUC, le F1-Score et le Coût Métier pour le logging MLflow.

model_utils.py,
• train_cv_and_log : Exécute une Stratified K-Fold Cross-Validation (5 folds) pour valider la robustesse du modèle sans biais.
• optimize_hyperparameters_optuna : Lance une recherche Bayésienne d'hyperparamètres. Intègre un Pruning Callback pour arrêter prématurément les essais non prometteurs (gain de temps/ressources).
• plot_business_cost_threshold : Algorithme de recherche du seuil de décision optimal (0.01 à 0.99).
• prepare_data_for_training : Split Train/Validation stratifié garantissant la conservation du ratio de défauts (Target=1).
• export_model_to_folder : Extrait le modèle final depuis le registry MLflow et prépare le dossier model/ (avec conda.yaml) pour la conteneurisation Docker.
• clean_feature_names : Nettoie les caractères spéciaux (JSON) pour compatibilité LightGBM.

explainability.py,
• plot_shap_global : Génère le Summary Plot pour identifier les tendances macro-économiques (ex: impact de l'âge).
• plot_shap_local : Génère le Waterfall Plot pour expliquer à un client spécifique pourquoi son crédit est refusé (Transparence).


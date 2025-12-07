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


Fichier,Fonctionnalités Clés & Choix Techniques
data_prep.py,
• load_dataframe : Charge les CSV avec optimisation mémoire automatique.
• clean_feature_names : Nettoie les caractères spéciaux (JSON) pour compatibilité LightGBM.
• prepare_data_for_training : Gère le split Train/Val Stratifié pour préserver le ratio de 8% de défauts.

metrics.py,"
• custom_business_cost : Implémente la formule Cost=10×FN+1×FP.
• get_metrics : Calcule simultanément l'AUC, le F1-Score et le Coût Métier pour le logging MLflow."

model_utils.py,
"• train_cv_and_log : Exécute une Stratified K-Fold Cross-Validation (5 folds) pour valider la robustesse du modèle sans biais.
• optimize_hyperparameters_optuna : Lance une recherche Bayésienne d'hyperparamètres. Intègre un ""Pruning Callback"" pour arrêter prématurément les essais non prometteurs (gain de temps/ressources).
• plot_business_cost_threshold : Algorithme de recherche du seuil de décision optimal (0.01 à 0.99)."

explainability.py,
• plot_shap_global : Génère le Summary Plot pour identifier les tendances macro-économiques (ex: impact de l'âge).
• plot_shap_local : Génère le Waterfall Plot pour expliquer à un client spécifique pourquoi son crédit est refusé (Transparence).
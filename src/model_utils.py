import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os
import shutil

# Imports Scikit-Learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, make_scorer

# Imports Optuna
import optuna
from optuna.integration.mlflow import MLflowCallback
import lightgbm as lgb


from mlflow.models.signature import infer_signature
# Gestion des imports de métriques locales
try:
    from metrics import get_metrics, custom_business_cost
except ImportError:
    from src.metrics import get_metrics, custom_business_cost

def clean_feature_names(df):
    """Nettoie les noms de colonnes pour LightGBM."""
    df = df.copy()
    new_columns = []
    for col in df.columns:
        new_col = re.sub(r'[^\w]', '_', col)
        new_col = re.sub(r'__+', '_', new_col)
        new_col = new_col.strip('_')
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def prepare_data_for_training(path_train_csv, path_labels_csv):
    print("Chargement des données...")
    X = pd.read_csv(path_train_csv)
    
    # ✅ IMPORTANT : Charger comme Series (avec index)
    y = pd.read_csv(path_labels_csv).squeeze()  # .squeeze() pour Series
    
    print("Nettoyage des noms de colonnes...")
    X = clean_feature_names(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    print("Split Train/Validation (Stratifié)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ✅ CRITIQUE : Réinitialiser les index pour éviter les décalages
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    return X_train, X_val, y_train, y_val

def train_cv_and_log(model, X, y, experiment_name, run_name, n_splits=5):
    """Validation Croisée Standard avec Logging MLflow."""
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        print(f"--- Cross-Validation ({n_splits} folds) : {run_name} ---")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores_cost, scores_auc = [], []
        
        # Conversion Numpy sécurisée pour la boucle CV
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        for i, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
            # On utilise iloc si X est un DataFrame, sinon slicing numpy
            if isinstance(X, pd.DataFrame):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
                y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            
            try:
                y_proba = model.predict_proba(X_fold_val)[:, 1]
            except:
                y_proba = model.decision_function(X_fold_val)
                
            y_pred = model.predict(X_fold_val)
            m = get_metrics(y_fold_val, y_pred, y_proba)
            scores_cost.append(m['business_cost'])
            scores_auc.append(m.get('auc', 0))
            
        avg_cost = np.mean(scores_cost)
        avg_auc = np.mean(scores_auc)
        print(f"  >> Moyenne CV: Coût={avg_cost:.1f} | AUC={avg_auc:.3f}")
        
        mlflow.log_metric("cv_mean_business_cost", avg_cost)
        mlflow.log_metric("cv_mean_auc", avg_auc)
        mlflow.log_params(model.get_params())
        
        # Sauvegarde du modèle ré-entraîné sur tout le dataset
        model.fit(X, y)
        # On montre à MLflow un exemple d'entrée (X) et de sortie (predict)
        # pour qu'il comprenne que les colonnes sont des Float, pas des Objets.
        signature = infer_signature(X, model.predict(X))
        
        try: 
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,       # <--- L'ARGUMENT VITAL
                input_example=X.iloc[:5]   # Bonus : ajoute un exemple dans la doc
            )
        except Exception as e: 
            print(f"Erreur lors du log MLflow : {e}")
            pass
        
        return model, avg_cost

def optimize_hyperparameters_optuna(X_train, y_train, experiment_name, n_trials=20):
    """
    Optimise LightGBM avec Optuna (Bayésien + Pruning).
    L'espace de recherche est défini ici.
    """
    print(f"--- Optimisation Optuna ({n_trials} essais) ---")
    
    def objective(trial):
        # 1. Définition de l'espace de recherche (La "Grille" intelligente)
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'n_jobs': -1, # Le modèle utilise tous les cœurs
            'random_state': 42,
            'is_unbalance': True, # Gestion du déséquilibre
            
            # Paramètres à optimiser
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        
        # 2. Dataset LightGBM interne
        dtrain = lgb.Dataset(X_train, label=y_train)
        
        # 3. Cross-Validation interne rapide (3 folds) avec PRUNING
        # Le callback arrête l'entraînement si les résultats sont mauvais (gain de temps/RAM)
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        
        history = lgb.cv(
            param, dtrain, nfold=3, stratified=True,
            callbacks=[pruning_callback]
        )
        
        # On retourne la meilleure moyenne AUC
        return history['valid auc-mean'][-1]

    # Configuration MLflow
    mlflow.set_experiment(experiment_name)
    
    # Lancement de l'étude (maximize AUC)
    study = optuna.create_study(direction="maximize")
    
    # n_jobs=1 pour l'étude car LightGBM utilise déjà n_jobs=-1 en interne (évite saturation RAM)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    print(f"Meilleurs params: {study.best_params}")
    
    # Conversion des types pour Sklearn
    best_params = study.best_params
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    
    return best_params

def plot_business_cost_threshold(y_true, y_proba):
    """Trace la courbe du coût métier vs seuil."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        costs.append(custom_business_cost(y_true, y_pred))
        
    best_idx = np.argmin(costs)
    best_thresh = thresholds[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs)
    plt.axvline(best_thresh, color='r', linestyle='--')
    plt.title(f"Seuil Optimal: {best_thresh:.2f} (Coût: {costs[best_idx]})")
    plt.xlabel("Seuil")
    plt.ylabel("Coût Métier")
    return plt.gcf(), best_thresh

def export_model_to_folder(run_id=None, model_name=None, experiment_name=None, 
                           target_folder="../model", tracking_uri=None):
    """
    Exporte un modèle MLflow vers un dossier local pour le Dockerfile.
    
    Cette fonction copie les fichiers nécessaires (MLmodel, model.pkl, conda.yaml, 
    requirements.txt, python_env.yaml) depuis MLflow vers le dossier spécifié.
    
    Paramètres:
    -----------
    run_id : str, optional
        ID du run MLflow (ex: "af5897f2130b491a9c3ce5320561668f")
        Si fourni, utilise ce run directement.
    model_name : str, optional
        Nom du modèle dans le registry MLflow (ex: "model")
        Nécessite experiment_name si utilisé.
    experiment_name : str, optional
        Nom de l'expérience MLflow (ex: "Credit_Scoring_Final")
        Nécessaire si model_name est utilisé.
    target_folder : str, default="model"
        Dossier de destination pour les fichiers du modèle.
    tracking_uri : str, optional
        URI du tracking MLflow (ex: "file:../mlruns")
        Si None, utilise la configuration actuelle.
    
    Retourne:
    --------
    str : Chemin du dossier de destination
    
    Exemples:
    --------
    # Option 1 : Exporter depuis un run_id spécifique
    export_model_to_folder(run_id="af5897f2130b491a9c3ce5320561668f")
    
    # Option 2 : Exporter depuis le registry MLflow
    export_model_to_folder(
        model_name="model",
        experiment_name="Credit_Scoring_Final"
    )
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Déterminer le chemin source du modèle
    if run_id:
        # Option 1 : Utiliser directement un run_id
        model_uri = f"runs:/{run_id}/model"
        print(f"Export depuis run_id: {run_id}")
    elif model_name and experiment_name:
        # Option 2 : Utiliser le registry MLflow
        mlflow.set_experiment(experiment_name)
        # Récupérer le dernier modèle enregistré avec ce nom
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["None"])[0]
        model_uri = f"models:/{model_name}/{model_version.version}"
        print(f"Export depuis registry: {model_name} (version {model_version.version})")
    else:
        raise ValueError("Vous devez fournir soit 'run_id', soit 'model_name' + 'experiment_name'")
    
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(target_folder, exist_ok=True)
    
    # Télécharger le modèle depuis MLflow vers un dossier temporaire
    print(f"Téléchargement du modèle depuis MLflow...")
    temp_download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    # Copier les fichiers nécessaires vers le dossier cible
    print(f"Copie des fichiers vers {target_folder}/...")
    required_files = ["MLmodel", "model.pkl", "conda.yaml", "requirements.txt", "python_env.yaml"]
    
    for file_name in required_files:
        source_path = os.path.join(temp_download_path, file_name)
        if os.path.exists(source_path):
            dest_path = os.path.join(target_folder, file_name)
            shutil.copy2(source_path, dest_path)
            print(f"  ✓ {file_name}")
        else:
            print(f"  ⚠ {file_name} non trouvé (optionnel)")
    
    # Nettoyer le dossier temporaire
    try:
        shutil.rmtree(temp_download_path)
    except:
        pass
    
    print(f"\n✅ Modèle exporté avec succès vers {target_folder}/")
    print(f"   Vous pouvez maintenant construire l'image Docker avec: docker build -t credit-scoring-model .")
    
    return os.path.abspath(target_folder)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

# Imports Scikit-Learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, make_scorer

# Imports Optuna
import optuna
from optuna.integration.mlflow import MLflowCallback
import lightgbm as lgb

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
    """
    Charge les données, nettoie et splitte.
    CORRECTIF : Charge 'y' comme Série pour garder les index alignés.
    """
    print("Chargement des données...")
    X = pd.read_csv(path_train_csv)
    # Important : On charge y comme une Series (avec index)
    y = pd.read_csv(path_labels_csv).iloc[:, 0] 
    
    print("Nettoyage des noms de colonnes...")
    X = clean_feature_names(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    print("Split Train/Validation (Stratifié)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
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
        try: mlflow.sklearn.log_model(model, "model")
        except: pass
        
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
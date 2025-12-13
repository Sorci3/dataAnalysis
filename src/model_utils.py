import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os
import shutil
import xgboost as xgb
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, make_scorer

from mlflow.models.signature import infer_signature
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
    
    y = pd.read_csv(path_labels_csv).squeeze()  
    
    print("Nettoyage des noms de colonnes...")
    X = clean_feature_names(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    print("Split Train/Validation (Stratifié)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    return X_train, X_val, y_train, y_val





def train_cv_and_log(model, X, y, experiment_name, run_name, n_splits=5, 
                     model_name=None, description=None, tags=None):
    """
    Validation Croisée Robuste avec Logging complet.
    
    1. Calcule Coût, AUC, F1, Rappel, Précision via get_metrics() pour chaque pli.
    2. Loggue la Moyenne (Performance) et l'Écart-Type (Stabilité) dans MLflow.
    3. Ré-entraîne le modèle sur TOUT le dataset pour la production.
    4. Enregistre le modèle avec les scores de validation dans ses métadonnées.
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        
        mlflow.set_tag("model_type", type(model).__name__)
        mlflow.set_tag("validation_strategy", f"CV_{n_splits}_Folds")
        if description: mlflow.set_tag("mlflow.note.content", description)
        if tags: mlflow.set_tags(tags)
        
        print(f"--- Cross-Validation ({n_splits} folds) : {run_name} ---")
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_scores = {
            "business_cost": [],
            "auc": [],
            "f1": [],
            "recall": [],
            "precision": []
        }
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        for i, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
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
            
            fold_scores["business_cost"].append(m.get('business_cost', 0))
            fold_scores["auc"].append(m.get('auc', 0))
            fold_scores["f1"].append(m.get('f1', 0))
            fold_scores["recall"].append(m.get('recall', 0))
            fold_scores["precision"].append(m.get('precision', 0))

        final_metrics = {}
        print(f"  >> Résultats CV (Moyenne ± Std):")
        
        for metric_name, values in fold_scores.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            final_metrics[metric_name] = mean_val
            
            mlflow.log_metric(f"cv_mean_{metric_name}", mean_val)
            mlflow.log_metric(f"cv_std_{metric_name}", std_val)
            
            print(f"     - {metric_name:<15}: {mean_val:.4f} ± {std_val:.4f}")

        mlflow.log_params(model.get_params())
        
        print("  >> Entraînement final sur l'ensemble du dataset (Production)...")
        model.fit(X, y)
        
        if isinstance(X, pd.DataFrame):
            input_example = X.iloc[:5]
            prediction = model.predict(X.iloc[:5])
        else:
            input_example = X[:5]
            prediction = model.predict(X[:5])
            
        signature = infer_signature(input_example, prediction)

        model_metadata = {
            "cv_auc_mean": f"{final_metrics['auc']:.4f}",
            "cv_f1_mean": f"{final_metrics['f1']:.4f}",
            "cv_cost_mean": f"{final_metrics['business_cost']:.2f}",
            "training_samples": str(len(X))
        }
        if description: model_metadata["description"] = description

        try: 
            mlflow.sklearn.log_model(
                model, 
                "model", 
                signature=signature,
                input_example=input_example,
                metadata=model_metadata 
            )
            print(f" Modèle sauvegardé dans MLflow (Run ID: {run.info.run_id})")

            if model_name:
                model_uri = f"runs:/{run.info.run_id}/model"
                mv = mlflow.register_model(model_uri, model_name)
                print(f" Modèle versionné : {model_name} (v{mv.version})")
                
                if description:
                    client = mlflow.tracking.MlflowClient()
                    client.update_model_version(
                        name=model_name,
                        version=mv.version,
                        description=f"{description} | CV AUC: {final_metrics['auc']:.3f}"
                    )
                    
        except Exception as e: 
            print(f"Attention : Problème lors du log/registry MLflow : {e}")
        
        return model, final_metrics['business_cost']
    






def optimize_hyperparameters_optuna(X_train, y_train, experiment_name, n_trials=20):
    """
    Optimise LightGBM avec Optuna + Logging Manuel (Compatible toutes versions).
    """
    if mlflow.active_run():
        print(f"⚠️ Fermeture du run actif précédent : {mlflow.active_run().info.run_id}")
        mlflow.end_run()

    mlflow.lightgbm.autolog(disable=True)
    
    print(f"--- Optimisation Optuna LightGBM ({n_trials} essais) ---")

    def objective(trial):
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'n_jobs': -1,
                'random_state': 42,
                'is_unbalance': True,
                
                'n_estimators': trial.suggest_int('n_estimators', 100, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            
            mlflow.log_params(param)
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
            
            history = lgb.cv(
                param, dtrain, nfold=3, stratified=True,
                callbacks=[pruning_callback]
            )
            
            auc_score = history['valid auc-mean'][-1]
            
            mlflow.log_metric("auc", auc_score)
            
            return auc_score

    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="Optuna_LightGBM_Search"):
        study = optuna.create_study(
            study_name=experiment_name, 
            direction="maximize"
        )
        
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        except Exception as e:
            print(f"Interruption ou Erreur : {e}")

        print("Génération des graphiques d'analyse...")
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            fig_hist = plot_optimization_history(study)
            mlflow.log_figure(fig_hist, "optuna_history.html")
            
            fig_imp = plot_param_importances(study)
            mlflow.log_figure(fig_imp, "optuna_importance.html")
        except:
            pass

    print(f"Meilleurs params: {study.best_params}")
    
    best_params = study.best_params
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    
    return best_params





def optimize_hyperparameters_xgboost(X_train, y_train, experiment_name, n_trials=20):
    """
    Optimise XGBoost avec Optuna (Bayésien + Pruning).
    Version compatible avec les anciennes et nouvelles versions d'Optuna.
    """
    print(f"--- Optimisation Optuna XGBoost ({n_trials} essais) ---")

    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    ratio_equilibrage = count_0 / count_1
    print(f"   Ratio scale_pos_weight calculé : {ratio_equilibrage:.2f}")

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="auc"
    )






    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'n_jobs': -1,
            'random_state': 42,
            'scale_pos_weight': ratio_equilibrage, 
            'booster': 'gbtree',
            
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        

        n_estimators = trial.suggest_int('n_estimators', 400, 1000)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")

        history = xgb.cv(
            param, 
            dtrain, 
            num_boost_round=n_estimators, 
            nfold=3, 
            stratified=True,
            early_stopping_rounds=50,
            callbacks=[pruning_callback],
            seed=42,
            verbose_eval=False
        )
        return history['test-auc-mean'].iloc[-1]


    mlflow.set_experiment(experiment_name)
    
    study = optuna.create_study(
        study_name=experiment_name,
        direction="maximize"
    )
    
    try:
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=1,
            callbacks=[mlflow_callback]
        )
    except Exception as e:
        print(f"Erreur pendant l'optimisation : {e}")
        print("Retour des meilleurs paramètres trouvés jusqu'ici.")

    print(f"Meilleurs params: {study.best_params}")
    

    best_params = study.best_params
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['scale_pos_weight'] = ratio_equilibrage
    best_params['n_jobs'] = -1
    best_params['random_state'] = 42
    
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
    
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        print(f"Export depuis run_id: {run_id}")
    elif model_name and experiment_name:
        mlflow.set_experiment(experiment_name)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["None"])[0]
        model_uri = f"models:/{model_name}/{model_version.version}"
        print(f"Export depuis registry: {model_name} (version {model_version.version})")
    else:
        raise ValueError("Vous devez fournir soit 'run_id', soit 'model_name' + 'experiment_name'")
    
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"Téléchargement du modèle depuis MLflow...")
    temp_download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    
    print(f"Copie des fichiers vers {target_folder}/...")
    required_files = ["MLmodel", "model.pkl", "conda.yaml", "requirements.txt", "python_env.yaml"]
    
    for file_name in required_files:
        source_path = os.path.join(temp_download_path, file_name)
        if os.path.exists(source_path):
            dest_path = os.path.join(target_folder, file_name)
            shutil.copy2(source_path, dest_path)
            print(f"   {file_name}")
        else:
            print(f"   {file_name} non trouvé (optionnel)")
    

    try:
        shutil.rmtree(temp_download_path)
    except:
        pass
    
    print(f"\nModèle exporté avec succès vers {target_folder}/")
    print(f"   Vous pouvez maintenant construire l'image Docker avec: docker build -t credit-scoring-model .")
    
    return os.path.abspath(target_folder)
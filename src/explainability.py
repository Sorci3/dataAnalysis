import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_shap_global(model, X_sample):
    """
    Affiche l'importance globale des variables avec SHAP (Summary Plot).
    
    Utilise TreeExplainer pour les modèles tree-based (LightGBM, XGBoost, etc.)
    ou LinearExplainer pour les modèles linéaires.
    
    Parameters
    ----------
    model : sklearn-like model
        Modèle entraîné (doit avoir une méthode predict_proba ou predict)
    X_sample : pandas.DataFrame ou numpy.ndarray
        Échantillon de données pour calculer les valeurs SHAP
        (recommandé : 100-1000 échantillons pour la performance)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure contenant le plot SHAP
    """
    print("Calcul des valeurs SHAP globales (cela peut prendre du temps)...")
    
    # 1. Détection du type de modèle pour choisir l'Explainer
    # TreeExplainer est optimisé pour XGBoost/LightGBM/RandomForest
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback pour les autres modèles (ex: Régression Logistique)
        # Note: KernelExplainer est très lent, on utilise un petit échantillon
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # 2. Gestion du format de sortie de SHAP (binaire vs multiclasse)
    # Pour la classification binaire, shap_values est souvent une liste [class0, class1]
    vals_to_plot = shap_values
    if isinstance(shap_values, list):
        # On s'intéresse à la classe 1 (Défaut de paiement)
        vals_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Certains formats retournent (n_samples, n_features, n_classes)
        vals_to_plot = shap_values[:, :, 1]

    # 3. Affichage
    plt.figure(figsize=(10, 8))
    shap.summary_plot(vals_to_plot, X_sample, plot_type="bar", show=False)
    plt.title("Importance Globale des Features (SHAP)")
    plt.tight_layout()
    return plt.gcf()

def plot_shap_local(model, X_sample, client_index=0):
    """
    Affiche l'explication locale (Waterfall Plot) pour un client spécifique.
    
    Montre comment chaque feature contribue à la prédiction pour un client donné.
    Utile pour comprendre pourquoi un client a été classé comme à risque ou non.
    
    Parameters
    ----------
    model : sklearn-like model
        Modèle entraîné
    X_sample : pandas.DataFrame ou numpy.ndarray
        Échantillon de données (doit contenir le client à l'index spécifié)
    client_index : int, default=0
        Index du client à expliquer dans X_sample
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure contenant le waterfall plot SHAP
    """
    print(f"Calcul du SHAP local pour le client à l'index {client_index}...")
    
    # Création de l'explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.LinearExplainer(model, X_sample)
    
    # Calcul des valeurs SHAP
    shap_values_obj = explainer(X_sample)
    
    # Récupération de la probabilité prédite
    proba = model.predict_proba(X_sample)[client_index, 1]
    
    # Visualisation waterfall
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values_obj[client_index], show=False)
    plt.title(f"Client {client_index} - Risque de défaut : {proba:.1%}", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt.gcf()
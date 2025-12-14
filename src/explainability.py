import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_shap_global(model, X_sample):
    """
    Affiche l'importance globale des variables avec SHAP (Summary Plot).
    
    Utilise TreeExplainer pour les modèles tree-based (LightGBM, XGBoost, etc.)
    ou LinearExplainer pour les modèles linéaires.
    """
    print("Calcul des valeurs SHAP globales (cela peut prendre du temps)...")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    vals_to_plot = shap_values
    if isinstance(shap_values, list):
        vals_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals_to_plot = shap_values[:, :, 1]

    #Affichage
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
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_shap_global(model, X_sample):
    """
    Affiche l'importance globale des variables (SHAP Summary Plot).
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

"""def plot_shap_local(model, X_sample, client_index=0):
    
    Affiche l'explication locale (Waterfall) pour un client donné.
    
    print(f"Calcul du SHAP local pour le client à l'index {client_index}...")
    
    # Pour le plot "waterfall", on a besoin de l'objet Explanation complet
    explainer = shap.TreeExplainer(model)
    shap_values_obj = explainer(X_sample) # Retourne un objet Explanation
    
    # Focus sur la classe 1 (Défaut)
    # L'objet Explanation contient souvent les valeurs brutes (log-odds)
    
    plt.figure(figsize=(8, 6))
    # On visualise la prédiction pour la classe 1 du client spécifié
    # shap_values_obj[index, :, classe] (la syntaxe peut varier selon la version de SHAP)
    
    try:
        # Syntaxe standard pour TreeExplainer récent
        shap.plots.waterfall(shap_values_obj[client_index], show=False)
    except:
        # Fallback si l'objet a une structure différente (souvent le cas avec LightGBM binaire)
        shap.plots.waterfall(shap_values_obj[client_index, :, 1], show=False)
        
    plt.title(f"Impact des features pour le client {client_index}")
    plt.tight_layout()
    return plt.gcf()"""

def plot_shap_local(model, X_sample, client_index=0):
    # ... (code précédent)
    
    # Récupérer la probabilité réelle calculée par le modèle
    proba = model.predict_proba(X_sample)[client_index, 1]
    
    plt.figure()
    shap.plots.waterfall(shap_values_obj[client_index])
    
    # On ajoute la probabilité lisible dans le titre
    plt.title(f"Client {client_index} - Risque de défaut : {proba:.1%}")
    plt.tight_layout()
    return plt.gcf()
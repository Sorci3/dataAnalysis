import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, make_scorer,recall_score, precision_score

def custom_business_cost(y_true, y_pred):
    """
    Calcule le coût métier : 10 * FN + 1 * FP.
    L'objectif est de MINIMISER ce coût.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Coût métier défini dans le projet (FN coûte 10x plus que FP)
    cost = 10 * fn + 1 * fp
    return cost

def business_score(y_true, y_pred):
    """
    Version normalisée ou inverse pour les optimiseurs qui cherchent à MAXIMISER.
    Ici, on retourne simplement l'opposé du coût pour la compatibilité scikit-learn
    si besoin (greater_is_better=False).
    """
    return -custom_business_cost(y_true, y_pred)

# Scorer pour GridSearchCV ou cross_val_score (si besoin)
business_scorer = make_scorer(custom_business_cost, greater_is_better=False)

def get_metrics(y_true, y_pred, y_proba=None):
    """
    Retourne un dictionnaire standardisé de toutes les métriques.
    Gère la division par zéro pour éviter les avertissements.
    """
    metrics = {}
    
    # 1. Métriques techniques (avec zero_division=0 pour éviter les warnings si TP=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.0 # Cas où il n'y a qu'une seule classe
    
    # 2. Métrique métier
    # Assurez-vous que custom_business_cost est accessible ici
    if 'custom_business_cost' in globals():
        metrics['business_cost'] = custom_business_cost(y_true, y_pred)
    else:
        # Fallback si la fonction n'est pas importée (pour éviter le crash)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['business_cost'] = (fn * 10) + (fp * 1) # Exemple par défaut
    
    # 3. Détail Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['fn'] = fn
    metrics['fp'] = fp
    metrics['tp'] = tp
    metrics['tn'] = tn
    
    return metrics
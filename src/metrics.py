import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, make_scorer

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
    """Retourne un dictionnaire de toutes les métriques pertinentes."""
    metrics = {}
    
    # Métriques techniques
    metrics['f1_score'] = f1_score(y_true, y_pred)
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    
    # Métrique métier
    metrics['business_cost'] = custom_business_cost(y_true, y_pred)
    
    # Détail de la matrice de confusion pour analyse
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['fn'] = fn
    metrics['fp'] = fp
    
    return metrics
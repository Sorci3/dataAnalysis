import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, make_scorer,recall_score, precision_score

def custom_business_cost(y_true, y_pred):
    """
    Calcule le coût métier : 10 * FN + 1 * FP.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Coût métier défini dans le projet (FN coûte 10x plus que FP)
    cost = 10 * fn + 1 * fp
    return cost

def business_score(y_true, y_pred):
    """
    Retourne l'inverse du score métier
    """
    return -custom_business_cost(y_true, y_pred)

business_scorer = make_scorer(custom_business_cost, greater_is_better=False)

def get_metrics(y_true, y_pred, y_proba=None):
    """
    Retourne un dictionnaire standardisé de toutes les métriques.
    """
    metrics = {}
    
    # Métriques techniques
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.0
    
    #Métriques métier
    if 'custom_business_cost' in globals():
        metrics['business_cost'] = custom_business_cost(y_true, y_pred)
    else:
        # Fallback si la fonction n'est pas importée
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['business_cost'] = (fn * 10) + (fp * 1) # Exemple par défaut
   
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['fn'] = fn
    metrics['fp'] = fp
    metrics['tp'] = tp
    metrics['tn'] = tn
    
    return metrics
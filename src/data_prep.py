import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
import sys
import os

# --- 1. FONCTIONS UTILITAIRES (MEMOIRE & NETTOYAGE) ---

def reduce_mem_usage(df):
    """
    Réduit la taille du DataFrame en optimisant les types de données.
    
    Convertit les types int/float vers les plus petits types compatibles
    (int8, int16, int32, float32) pour réduire l'utilisation mémoire.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à optimiser
        
    Returns
    -------
    pandas.DataFrame
        DataFrame avec les types optimisés
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Mémoire réduite: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% gain)")
    return df

def load_dataframe(path, filename, debug=False):
    """
    Charge un fichier CSV et optimise immédiatement sa mémoire.
    
    Parameters
    ----------
    path : str
        Chemin vers le dossier contenant les fichiers CSV
    filename : str
        Nom du fichier (sans extension .csv)
    debug : bool, default=False
        Si True, charge uniquement les 10 000 premières lignes
        
    Returns
    -------
    pandas.DataFrame
        DataFrame chargé et optimisé en mémoire
    """
    file_path = os.path.join(path, f'{filename}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{filename}.csv introuvable dans {path}")
    
    nrows = 10000 if debug else None
    print(f"Chargement de {filename}...")
    df = pd.read_csv(file_path, nrows=nrows)
    return reduce_mem_usage(df)

# --- 2. FONCTIONS METIERS ---

def handle_anomalies(df):
    """
    Gère les anomalies connues dans les données.
    
    - Remplace la valeur anormale 365243 dans DAYS_EMPLOYED par NaN
      et crée un flag DAYS_EMPLOYED_ANOM
    - Convertit DAYS_BIRTH en valeurs positives (abs)
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à nettoyer
        
    Returns
    -------
    pandas.DataFrame
        DataFrame avec les anomalies traitées
    """
    df = df.copy()  # Éviter les warnings de modification sur vue
    for col in df.columns:
        if 'DAYS_EMPLOYED' in col:
            df[f'{col}_ANOM'] = df[col] == 365243
            df[col] = df[col].replace({365243: np.nan})
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    return df

def engineer_domain_features(df):
    """
    Crée des features de domaine (ratios financiers) basées sur la connaissance métier.
    
    Features créées :
    - CREDIT_INCOME_PERCENT : Ratio crédit / revenu
    - ANNUITY_INCOME_PERCENT : Ratio annuité / revenu
    - CREDIT_TERM : Ratio annuité / crédit
    - DAYS_EMPLOYED_PERCENT : Ratio jours employés / jours depuis naissance
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les colonnes nécessaires
        
    Returns
    -------
    pandas.DataFrame
        DataFrame avec les nouvelles features ajoutées
    """
    df = df.copy()
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    return df

def encode_categorical(df, df_test=None):
    """
    Encode les variables catégorielles.
    
    - Label Encoding pour les variables binaires (≤2 catégories)
    - One-Hot Encoding pour les autres variables catégorielles
    - Aligne les colonnes entre train et test après encodage
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame d'entraînement
    df_test : pandas.DataFrame, optional
        DataFrame de test (si fourni, aligne les colonnes)
        
    Returns
    -------
    pandas.DataFrame ou tuple
        DataFrame(s) avec les variables catégorielles encodées
    """
    df = df.copy()
    le = LabelEncoder()
    
    # Label Encoding pour les variables binaires
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) <= 2:
                if df_test is not None and col in df_test.columns:
                    df[col] = df[col].astype(str).fillna('MISSING')
                    df_test[col] = df_test[col].astype(str).fillna('MISSING')
                    le.fit(pd.concat([df[col], df_test[col]]).astype(str))
                    df[col] = le.transform(df[col])
                    df_test[col] = le.transform(df_test[col])
    
    # One-Hot Encoding pour toutes les variables catégorielles restantes
    df = pd.get_dummies(df)
    if df_test is not None:
        df_test = pd.get_dummies(df_test)
        train_labels = df['TARGET'] if 'TARGET' in df.columns else None
        # Alignement des colonnes (inner join pour garder uniquement les colonnes communes)
        df, df_test = df.align(df_test, join='inner', axis=1)
        if train_labels is not None:
            df['TARGET'] = train_labels
        return df, df_test
    return df

# --- 3. FONCTIONS D'AGRÉGATION (CORRIGEES) ---

def agg_numeric(df, parent_var, df_name):
    """
    Agrège les colonnes numériques d'un DataFrame avec plusieurs statistiques descriptives.

    Sélectionne toutes les colonnes numériques du DataFrame, effectue un groupby
    sur la variable parent et calcule les statistiques count, mean, max, min et sum
    pour chaque colonne. Les noms des colonnes résultantes suivent le format
    '{df_name}_{colonne}_{statistique}'.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données à agréger
    parent_var : str
        Nom de la colonne servant de clé de regroupement
    df_name : str
        Préfixe utilisé pour nommer les colonnes agrégées dans le DataFrame résultant

    Returns
    -------
    pandas.DataFrame
        DataFrame agrégé avec parent_var comme colonne et les statistiques
        calculées pour chaque variable numérique

    """
    # 1. Sélectionner les colonnes numériques + la clé
    cols_to_keep = [parent_var] + list(df.select_dtypes('number').columns)
    numeric_df = df[list(set(cols_to_keep))].copy()
    
    # 2. Groupby (parent_var devient l'index)
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    
    # 3. Aplatir les colonnes (MultiIndex) via une compréhension de liste
    # Cela garantit que le nombre de nouveaux noms correspond EXACTEMENT au nombre de colonnes
    agg.columns = [f'{df_name}_{col[0]}_{col[1]}' for col in agg.columns]
    
    # 4. Transformer l'index en colonne pour faciliter les merges futurs
    agg = agg.reset_index()
    
    return agg

def agg_categorical(df, parent_var, df_name):
    """
    Agrège les colonnes catégorielles d'un DataFrame via encodage one-hot.

    Encode les variables catégorielles en variables binaires (one-hot encoding),
    puis agrège ces variables par parent_var en calculant la somme (count) et
    la moyenne (count_norm) pour chaque catégorie. Si aucune colonne catégorielle
    n'existe, retourne un DataFrame contenant uniquement les valeurs uniques
    de parent_var.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données à agréger
    parent_var : str
        Nom de la colonne servant de clé de regroupement
    df_name : str
        Préfixe utilisé pour nommer les colonnes agrégées dans le DataFrame résultant

    Returns
    -------
    pandas.DataFrame
        DataFrame agrégé avec parent_var comme colonne et les statistiques
        d'encodage (count et count_norm) pour chaque modalité catégorielle.
        Si aucune colonne catégorielle n'existe, retourne uniquement parent_var
        avec ses valeurs uniques
    """

    categorical_cols = df.select_dtypes('object').columns
    
    # Cas où il n'y a pas de catégories (ex: installments_payments)
    if len(categorical_cols) == 0:
        # On retourne un DF avec juste la colonne ID unique pour ne pas casser le merge
        return pd.DataFrame({parent_var: df[parent_var].unique()})

    categorical = pd.get_dummies(df[categorical_cols])
    categorical[parent_var] = df[parent_var]
    
    # Groupby
    categorical_grouped = categorical.groupby(parent_var).agg(['sum', 'mean'])
    
    # Aplatir les colonnes
    new_cols = []
    for col in categorical_grouped.columns:
        stat_suffix = 'count' if col[1] == 'sum' else 'count_norm'
        new_cols.append(f'{df_name}_{col[0]}_{stat_suffix}')

    categorical_grouped.columns = new_cols
    
    # Reset index pour avoir l'ID en colonne
    categorical_grouped = categorical_grouped.reset_index()
    
    return categorical_grouped

def aggregate_client(df, group_vars, df_names):
    """
    Agrège les données en deux niveaux hiérarchiques (prêt puis client).

    Effectue une agrégation en cascade : d'abord au niveau du prêt individuel
    (group_vars[0]), puis au niveau du client (group_vars[1]). Combine les
    agrégations numériques et catégorielles à chaque niveau. Optimise la mémoire
    en supprimant les DataFrames intermédiaires après usage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame contenant les données à agréger avec au moins deux niveaux
        hiérarchiques
    group_vars : list of str
        Liste de deux éléments : [variable_niveau_prêt, variable_niveau_client]
    df_names : list of str
        Liste de deux préfixes pour nommer les colonnes agrégées à chaque niveau

    Returns
    -------
    pandas.DataFrame
        DataFrame agrégé au niveau client avec toutes les statistiques calculées
        sur les deux niveaux d'agrégation
    """

    # 1. Agréger sur SK_ID_PREV (ou équivalent)
    df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])
    df_counts = agg_categorical(df, parent_var=group_vars[0], df_name=df_names[0])
    
    # 2. Fusionner les deux (les deux ont la clé en colonne maintenant)
    if df_counts.shape[1] > 1: # Si df_counts n'est pas vide
        df_by_loan = df_agg.merge(df_counts, on=group_vars[0], how='outer')
    else:
        df_by_loan = df_agg
    
    # Optimisation mémoire
    df_by_loan = reduce_mem_usage(df_by_loan)
    del df_agg, df_counts; gc.collect()

    # 3. Ajouter SK_ID_CURR
    loan_to_client_id = df[[group_vars[0], group_vars[1]]].drop_duplicates(subset=[group_vars[0]])
    df_by_loan = df_by_loan.merge(loan_to_client_id, on=group_vars[0], how='left')
    
    # 4. Retirer SK_ID_PREV pour la 2ème agrégation
    if group_vars[0] in df_by_loan.columns:
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])
        
    # 5. Agréger sur SK_ID_CURR
    df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])
    
    del df, df_by_loan, loan_to_client_id; gc.collect()
    return df_by_client

# --- 4. FONCTION PRINCIPALE ---

def merge_and_clean(df_main, df_agg):
    """
    Fusionne un DataFrame principal avec un DataFrame agrégé et nettoie la mémoire.

    Effectue une jointure gauche (left join) entre le DataFrame principal et
    le DataFrame agrégé en utilisant une clé commune. Privilégie 'SK_ID_CURR'
    comme clé de jointure si disponible, sinon utilise la première colonne commune
    trouvée. Libère la mémoire en supprimant le DataFrame agrégé après fusion.

    Parameters
    ----------
    df_main : pandas.DataFrame
        DataFrame principal auquel ajouter les colonnes agrégées
    df_agg : pandas.DataFrame
        DataFrame agrégé contenant les nouvelles features à fusionner

    Returns
    -------
    pandas.DataFrame
        DataFrame principal enrichi avec les colonnes du DataFrame agrégé.
        Retourne df_main inchangé si aucune colonne commune n'est trouvée
    """
    # Trouver le nom de la clé de jointure (c'est la colonne commune)
    common_cols = list(set(df_main.columns) & set(df_agg.columns))
    if not common_cols:
        print("Erreur: Pas de clé commune pour le merge.")
        return df_main
    
    merge_key = 'SK_ID_CURR' if 'SK_ID_CURR' in common_cols else common_cols[0]
    
    df_main = df_main.merge(df_agg, on=merge_key, how='left')
    del df_agg
    gc.collect()
    return df_main

def load_and_process_all_data(path_relative_to_root='datasets', debug=False):
    """
    Charge et traite l'ensemble des données du projet Home Credit Default Risk.

    Pipeline complet qui charge tous les fichiers de données, effectue les
    agrégations hiérarchiques nécessaires, fusionne les tables selon leur
    structure relationnelle (bureau->client, previous->mensuel->client),
    encode les variables catégorielles et optimise l'utilisation mémoire.
    Retourne les datasets train et test prêts pour la modélisation.

    Parameters
    ----------
    path_relative_to_root : str, default='datasets'
        Chemin relatif depuis la racine du projet vers le dossier contenant
        les fichiers de données
    debug : bool, default=False
        Si True, charge uniquement un échantillon réduit des données pour
        faciliter les tests et le débogage

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        - app_train : DataFrame d'entraînement avec toutes les features agrégées
          et la colonne TARGET
        - app_test : DataFrame de test avec les mêmes features (sans TARGET)
        Retourne (None, None) en cas d'erreur critique lors du chargement
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, '..')
    final_data_path = os.path.join(root_dir, path_relative_to_root)
    
    # 1. Charger APPLICATION
    try:
        app_train = load_dataframe(final_data_path, 'application_train', debug)
        app_test = load_dataframe(final_data_path, 'application_test', debug)
    except Exception as e:
        print(f"Erreur critique : {e}")
        return None, None

    # Traitement initial
    app_train = handle_anomalies(app_train)
    app_test = handle_anomalies(app_test)
    app_train = engineer_domain_features(app_train)
    app_test = engineer_domain_features(app_test)
    
    app_train, app_test = encode_categorical(app_train, app_test)
    train_labels = app_train['TARGET']
    app_train = app_train.drop(columns=['TARGET'])
    gc.collect()

    # --- BRANCHE 1: BUREAU ---
    print("\n--- Traitement Bureau & Balance ---")
    bureau = load_dataframe(final_data_path, 'bureau', debug)
    bureau_balance = load_dataframe(final_data_path, 'bureau_balance', debug)
    
    # Agrégations N3 -> N2
    bb_agg = agg_numeric(bureau_balance, 'SK_ID_BUREAU', 'bb')
    bb_counts = agg_categorical(bureau_balance, 'SK_ID_BUREAU', 'bb')
    del bureau_balance; gc.collect()
    
    # Fusions dans Bureau. Tout est fusionné via la COLONNE SK_ID_BUREAU.
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    bureau = bureau.merge(bb_counts, on='SK_ID_BUREAU', how='left')
    del bb_agg, bb_counts; gc.collect()
    
    # Agrégations N2 -> N1
    bureau_agg = agg_numeric(bureau.drop(columns=['SK_ID_BUREAU']), 'SK_ID_CURR', 'bureau_agg')
    bureau_counts = agg_categorical(bureau.drop(columns=['SK_ID_BUREAU']), 'SK_ID_CURR', 'bureau_counts')
    del bureau; gc.collect()
    
    # Fusion dans App
    app_train = merge_and_clean(app_train, bureau_agg)
    app_test = merge_and_clean(app_test, bureau_agg)
    app_train = merge_and_clean(app_train, bureau_counts)
    app_test = merge_and_clean(app_test, bureau_counts)
    
    # --- BRANCHE 2: PREVIOUS ---
    print("\n--- Traitement Previous Application ---")
    previous = load_dataframe(final_data_path, 'previous_application', debug)
    
    prev_agg = agg_numeric(previous.drop(columns=['SK_ID_PREV']), 'SK_ID_CURR', 'prev_agg')
    prev_counts = agg_categorical(previous.drop(columns=['SK_ID_PREV']), 'SK_ID_CURR', 'prev_counts')
    
    app_train = merge_and_clean(app_train, prev_agg)
    app_test = merge_and_clean(app_test, prev_agg)
    app_train = merge_and_clean(app_train, prev_counts)
    app_test = merge_and_clean(app_test, prev_counts)
    
    # On garde la map ID pour les tables mensuelles
    # prev_id_map = previous[['SK_ID_PREV', 'SK_ID_CURR']].copy() # Inutile car inclus dans aggregate_client
    del previous; gc.collect()

    # --- TABLES MENSUELLES ---
    
    # POS_CASH
    print("\n--- Traitement POS_CASH ---")
    cash = load_dataframe(final_data_path, 'POS_CASH_balance', debug)
    cash_agg = aggregate_client(cash, ['SK_ID_PREV', 'SK_ID_CURR'], ['cash', 'client_cash'])
    del cash; gc.collect()
    app_train = merge_and_clean(app_train, cash_agg)
    app_test = merge_and_clean(app_test, cash_agg)

    # Installments
    print("\n--- Traitement Installments ---")
    install = load_dataframe(final_data_path, 'installments_payments', debug)
    install_agg = aggregate_client(install, ['SK_ID_PREV', 'SK_ID_CURR'], ['install', 'client_install'])
    del install; gc.collect()
    app_train = merge_and_clean(app_train, install_agg)
    app_test = merge_and_clean(app_test, install_agg)

    # Credit Card
    print("\n--- Traitement Credit Card ---")
    credit = load_dataframe(final_data_path, 'credit_card_balance', debug)
    credit_agg = aggregate_client(credit, ['SK_ID_PREV', 'SK_ID_CURR'], ['credit', 'client_credit'])
    del credit; gc.collect()
    app_train = merge_and_clean(app_train, credit_agg)
    app_test = merge_and_clean(app_test, credit_agg)
    
    # --- FINAL ---
    print("\nAlignement final...")
    app_train, app_test = app_train.align(app_test, join='inner', axis=1)
    app_train['TARGET'] = train_labels
    
    app_train = reduce_mem_usage(app_train)
    app_test = reduce_mem_usage(app_test)
    
    print(f"Succès. Train: {app_train.shape}, Test: {app_test.shape}")
    return app_train, app_test

def missing_values_table(df):
    """
    Génère un tableau récapitulatif des valeurs manquantes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à analyser
        
    Returns
    -------
    pandas.DataFrame
        Tableau avec les colonnes ayant des valeurs manquantes,
        triées par pourcentage décroissant
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    return mis_val_table[mis_val_table.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
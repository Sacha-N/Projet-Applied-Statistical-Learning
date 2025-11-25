import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV

# --- 1. Préparation des Données et Variables ---

df = pd.read_csv("data/output.csv", sep=";")
#print(df.head())
#df.columns

# Filtrage des lignes sans la variable cible
df_filtre_duree_travaux = df.dropna(subset=["duree_travaux"])

# Variables à retirer du jeu de données final
vars_inutiles_duree_travaux = [
    "COMM",
    "REG_CODE",
    "REG_LIBELLE",
    "DEP_LIBELLE",
    "NUM_DAU",
    "APE_DEM",
    "CJ_DEM",
    "duree_obtiention_autorisation",
    "DATE_REELLE_AUTORISATION",
    "DATE_REELLE_DAACT",
    "DATE_REELLE_DOC",
    "DPC_AUT",
    "DPC_PREM",
    "DPC_DOC",
    "DPC_DERN", 
    "delai_ouverture_chantier"]

# Variables à traiter en One-Hot Encoding
vars_categ = [
    "DEP_CODE",
    "TYPE_DAU",
    "ETAT_DAU",
    "CAT_DEM",
    "ZONE_OP",
    "NATURE_PROJET_DECLAREE",
    "UTILISATION",
    "RES_PRINCIP_OU_SECOND",
    "TYP_ANNEXE",
    "RESIDENCE"]

# Échantillonage par fraction  ---
df_sample = df_filtre_duree_travaux.sample(
    frac=0.1, 
    random_state=42 # Pour que l'échantillon soit le même à chaque exécution
)

# Nettoyage et conversion des types sur l'échantillon
df_sample = df_sample.drop(columns=vars_inutiles_duree_travaux, errors="ignore")
df_sample[vars_categ] = df_sample[vars_categ].astype("string").fillna("Missing")

#df_filtre_duree_travaux = df_filtre_duree_travaux.drop(columns=vars_inutiles_duree_travaux, errors="ignore")
#df_filtre_duree_travaux[vars_categ] = df_filtre_duree_travaux[vars_categ].astype("string").fillna("Missing")

print(f"Taille originale du DataFrame: {len(df_filtre_duree_travaux)} lignes")
print(f"Taille du DataFrame échantillonné: {len(df_sample)} lignes")

# Définition des variables X et y
#y = df_filtre_duree_travaux["duree_travaux"]
#X = df_filtre_duree_travaux.drop(columns=["duree_travaux"])

y = df_sample["duree_travaux"]
X = df_sample.drop(columns=["duree_travaux"])

# Séparation des colonnes numériques et catégorielles après nettoyage
num_cols = X.select_dtypes(include=["float", "int"]).columns
cat_cols = X.select_dtypes(include=["string"]).columns


# --- 2. Inspection de la Cardinalité (Critique pour la performance) ---
print("### Inspection de la Cardinalité des Variables Catégorielles ###")
for col in cat_cols:
    n_unique = X[col].nunique()
    print(f"La colonne '{col}' a {n_unique} valeurs uniques.")

print("Si le nombre de valeurs uniques est très élevé (> 50-100) pour plusieurs colonnes, le modèle sera très lent.")

# --- 3. Création du Pipeline de Pré-traitement ---

preprocess = ColumnTransformer(
    transformers=[
        # Mise à l'échelle des variables numériques (StandardScaler)
        ("num", StandardScaler(), num_cols), 
        # Encodage One-Hot des variables catégorielles (handle_unknown='ignore' pour les valeurs futures non vues)
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder='passthrough' # Ne rien faire avec les colonnes restantes
)

# --- 4. Modèle LASSO avec Optimisations de Performance ---

# LassoCV : Recherche automatique du meilleur alpha via Cross-Validation (cv=5)
# n_jobs=-1 : Utilise tous les cœurs du processeur pour accélérer la Cross-Validation
# solver='saga' : Solveur plus adapté aux matrices de données larges et creuses issues du One-Hot Encoding
# max_iter=50000 : Augmente le nombre maximum d'itérations pour garantir la convergence

lasso = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LassoCV(
        cv=5, 
        random_state=0,
        n_jobs=-1,        # Parallélisation
        #solver='saga',    # Solveur optimisé
        max_iter=50000    # Augmentation des itérations
    ))
])

# --- 5. Entraînement du Modèle ---

lasso.fit(X, y)

print(f"Le meilleur paramètre alpha trouvé est : {lasso['model'].alpha_}")
print(f"Le score R² (sur les données d'entraînement) est : {lasso.score(X, y):.4f}")
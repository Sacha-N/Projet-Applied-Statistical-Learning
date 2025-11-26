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
df_filtre_delai_ouverture = df.dropna(subset=["delai_ouverture_chantier"])

# Variables à retirer du jeu de données final
vars_inutiles_delai_ouverture = [
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
    "duree_travaux"]

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

# Échantillonnage par fraction (recommandé pour les grands jeux de données) ---
df_sample = df_filtre_delai_ouverture.sample(
    frac=0.05, 
    random_state=42 # Pour que l'échantillon soit le même à chaque exécution
)

# Nettoyage et conversion des types sur l'échantillon
df_sample = df_sample.drop(columns=vars_inutiles_delai_ouverture, errors="ignore")
df_sample[vars_categ] = df_sample[vars_categ].astype("string").fillna("Missing")

df_filtre_delai_ouverture = df_filtre_delai_ouverture.drop(columns=vars_inutiles_delai_ouverture, errors="ignore")
df_filtre_delai_ouverture[vars_categ] = df_filtre_delai_ouverture[vars_categ].astype("string").fillna("Missing")

print(f"Taille originale du DataFrame: {len(df_filtre_delai_ouverture)} lignes")
print(f"Taille du DataFrame échantillonné: {len(df_sample)} lignes")

# Définition des variables X et y

y = df_filtre_duree_travaux["duree_travaux"]
X = df_filtre_duree_travaux.drop(columns=["duree_travaux"])

y = df_sample["delai_ouverture_chantier"]
X = df_sample.drop(columns=["delai_ouverture_chantier"])

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
# max_iter=50000 : Augmente le nombre maximum d'itérations pour garantir la convergence
lasso = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LassoCV(
        cv=5, 
        random_state=0,
        n_jobs=-1,        # Parallélisation
        max_iter=50000    # Augmentation des itérations
    ))
])


# --- 5. Entraînement du Modèle ---

lasso.fit(X, y)

print(f"Le meilleur paramètre alpha trouvé est : {lasso['model'].alpha_}")
print(f"Le score R² (sur les données d'entraînement) est : {lasso.score(X, y):.4f}")

# --- 6. Exploitation des résultats

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

lasso = Lasso(alpha=0.1, max_iter=10000)

lasso_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", lasso)
])

lasso_pipeline.fit(X_train, y_train)


# Prédictions
y_train_pred = lasso_pipeline.predict(X_train)
y_test_pred = lasso_pipeline.predict(X_test)

# Erreurs
train_rmse = mean_squared_error(y_train, y_train_pred, squarred=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"RMSE train : {train_rmse:.2f}")
print(f"RMSE test  : {test_rmse:.2f}")
print(f"MAE train  : {train_mae:.2f}")
print(f"MAE test   : {test_mae:.2f}")
print(f"R² train   : {train_r2:.3f}")
print(f"R² test    : {test_r2:.3f}")


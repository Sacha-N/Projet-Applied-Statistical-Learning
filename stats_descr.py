# Stats descriptives ------

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/output.csv", sep=";")
print(df.head())
print(df.dtypes)

vars_duree = ["duree_travaux", "delai_ouverture_chantier", "duree_obtiention_autorisation"]

for col in vars_duree:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30)
    plt.title(f"Distribution de {col}")
    plt.xlabel("Durée (jours)")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(15, 4))  # Largeur augmentée pour 3 graphes
for i, col in enumerate(vars_duree, 1):
    plt.subplot(1, 3, i)
    plt.hist(df[col], bins=100)
    plt.title(f"{col}")
    plt.xlabel("Durée (jours)")
    plt.xlim(0, 1000) 
    plt.ylabel("Fréquence")

plt.tight_layout()
plt.show()


# Tableau des statistiques
stats = df[vars_duree].agg(["mean", "median", "min", "max"]).T
print(stats)

# Traitement des valeurs manquantes -----------

df_filtre_duree_travaux = df.dropna(subset=["duree_travaux"])
df_filtre_delai_ouverture = df.dropna(subset=["delai_ouverture_chantier"])

df_filtre_duree_travaux.isna().sum()
df_filtre_delai_ouverture.isna().sum()

# Il reste des NA importants dans les variables APE_DEM et CJ_DEM mais on s'en tape

# LASSO sur durée des travaux ----------

df_filtre_duree_travaux.head
print(df_filtre_duree_travaux.dtypes)

# Il faut retirer certaines variables inutiles qu'il reste et convertir d'autres variables en catégorielles

vars_inutiles_duree_travaux = [
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

vars_categ = [
    "ETAT_DAU",
    "CAT_DEM",
    "ZONE_OP",
    "NATURE_PROJET_DECLAREE",
    "UTILISATION",
    "RES_PRINCIP_OU_SECOND",
    "TYP_ANNEXE",
    "RESIDENCE"]



df_filtre_duree_travaux = df_filtre_duree_travaux.drop(columns=vars_inutiles_duree_travaux, errors="ignore")
df_filtre_duree_travaux[vars_categ] = df_filtre_duree_travaux[vars_categ].astype("string")

y = df_filtre_duree_travaux["duree_travaux"]
X = df_filtre_duree_travaux.drop(columns=["duree_travaux"])

cat_cols = X.select_dtypes(exclude=["float", "int"]).columns

for col in cat_cols:
    X[col] = X[col].astype("string")
num_cols = X.select_dtypes(include=["float", "int"]).columns
cat_cols = X.select_dtypes(include=["string"]).columns



preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

lasso = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LassoCV(cv=5, random_state=0))
])
lasso.fit(X, y)
print("Alpha optimal :", lasso.named_steps["model"].alpha_)

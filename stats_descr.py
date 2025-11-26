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
df.columns

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

# Nombres de permis par années et par régions

df.groupby("annee_autorisation").size()

df.groupby("REG_LIBELLE").size()

# Limitation de la base à la France métropolitaine post 2020 

regions_outremer = [
    "Guadeloupe", "Martinique", "Guyane",
    "La Réunion", "Mayotte"
]
df["DATE_REELLE_AUTORISATION"] = pd.to_datetime(df["DATE_REELLE_AUTORISATION"], errors="coerce")
df["annee_autorisation"] = df["DATE_REELLE_AUTORISATION"].dt.year
df = df[~df["REG_LIBELLE"].isin(regions_outremer)]
df = df[df["annee_autorisation"] >= 2020]
df.size



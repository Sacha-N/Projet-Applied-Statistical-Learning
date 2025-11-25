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

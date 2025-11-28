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
plt.savefig("figure.png", dpi=150)


# Tableau des statistiques
stats = df.groupby("annee_autorisation")["delai_ouverture_chantier"].agg(["mean", "median", "min", "max"])
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

# Graphiques de résultats --------

# Histogramme délai ouverture chantier 
plt.figure(figsize=(8,5))
df["delai_ouverture_chantier"].hist(bins=50)
plt.title("Histogramme du délai d'ouverture de chantier")
plt.xlabel("Délai (jours)")
plt.ylabel("Fréquence")

plt.savefig("hist_delai_ouverture.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot prédiction VS prédit gradient boosting
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_test_pred_gb, alpha=0.3, s=10)
plt.xlabel("Valeurs observées du délai")
plt.ylabel("Valeurs prédites du délai")
plt.title("Gradient Boosting : Observé vs Prédit\n(delai_ouverture_chantier)")

min_val = min(y_test.min(), y_test_pred_gb.min())
max_val = max(y_test.max(), y_test_pred_gb.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.savefig("gb_scatter_obs_pred_delai.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot résidus gradient boosting
residuals_gb = y_test - y_test_pred_gb

plt.figure(figsize=(6,4))
plt.hist(residuals_gb, bins=50)
plt.xlabel("Résidu (observé - prédit)")
plt.ylabel("Fréquence")
plt.title("Gradient Boosting : distribution des résidus\n(delai_ouverture_chantier)")

plt.savefig("gb_residuals_hist_delai.png", dpi=150, bbox_inches="tight")
plt.close()



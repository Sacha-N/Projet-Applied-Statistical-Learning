import sys, importlib.util as u
print(sys.executable)
print(u.find_spec("pandas"))
import pandas as pd
print(pd.__version__)
import matplotlib.pyplot as plt

df = pd.read_csv("Liste-des-autorisations-durbanisme-creant-des-logements.2025-10.csv", sep=";",encoding='utf-8')

print(df.shape)
print(df.head())
df.info()

# --- Aperçu général ---
print("Dimensions :", df.shape)
print("\nColonnes :")
print(df.columns.tolist())

print("\nTypes de variables :")
print(df.dtypes)

print("\nAperçu des 5 premières lignes :")
print(df.head())

# --- Valeurs manquantes ---
print("\nTaux de valeurs manquantes (top 10) :")
print(df.isna().mean().sort_values(ascending=False).head(10))

# --- Doublons ---
print("\nNombre de doublons :", df.duplicated().sum())

# --- Statistiques numériques ---
print("\nRésumé des variables numériques :")
print(df.describe())

# Variable Année au pif, pour voir à quoi ça ressemble 
print(df['Année de dépôt de la DAU'])
print(df['Année de dépôt de la DAU'].unique())
print(df['Année de dépôt de la DAU'].value_counts().sort_index()) #Valeurs abérantes (mais peu donc ça s'enlève)

# Conversion en chiffres
df['Année de dépôt de la DAU'] = pd.to_numeric(df['Année de dépôt de la DAU'], errors='coerce')

#Graphique du nombre de dépot de permis par année 
df['Année de dépôt de la DAU'].value_counts().sort_index().plot(kind='bar')
plt.title("Nombre de permis par année de dépôt")
plt.xlabel("Année de dépôt de la DAU")
plt.ylabel("Nombre de permis")
plt.show()






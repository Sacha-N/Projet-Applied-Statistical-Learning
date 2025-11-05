# Libraries
import pandas as pd

# 1. Chargement de Sitadel
chemin = "data/Liste-des-autorisations-logements-2025-10.csv"
df = pd.read_csv(chemin, sep=";",encoding='utf-8')

print(df.shape)
print(df.head())
df.info()
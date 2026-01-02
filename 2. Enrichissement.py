# Libraries
import pandas as pd
import openpyxl


# 1. Chargement des autorisations nettoyées
chemin = "data/autorisations.csv"

df = pd.read_csv(
    chemin,
    sep=";",
    encoding="utf-8",
)
df.info()
print(df.head())

# 2. Chargement de données externes
# 2.1. Grille de densité à 7 niveaux
grille_densite = pd.read_excel(
    "data/grille_densite_7_niveaux_2019.xlsx",
    skiprows=4
)
print(grille_densite.head())
grille_densite.info()
print(grille_densite.columns.tolist())

cols_grille = [
    "CODGEO",
    "DENS",# De 1 à 7
    "PMUN17" #Pop° municipale 2017
]
grille_reduite = grille_densite[cols_grille]

# On corrige les formats
df["COMM"] = df["COMM"].astype(str).str.zfill(5)
grille_densite["CODGEO"] = grille_densite["CODGEO"].astype(str).str.zfill(5)
df = df.merge(
    grille_reduite,
    how="left",
    left_on="COMM",
    right_on="CODGEO",
    validate="m:1"
)

missing_rate = df["DENS"].isna().mean()
print(f"Pourcentage sans densité: {missing_rate:.2%}")


# 2.2. Données socio-économiques par communes
chemin_dc = "data/dossier_complet.csv"
all_cols = pd.read_csv(chemin_dc, sep=";", skiprows=0, nrows=1).columns.tolist()
print(all_cols)

cols_dossier_complet = [
    "CODGEO",
    "TP6021",# taux pauvreté 60% en 2021
    "MED21", #Médiane des revenus fiscaux en 2021"
    "PIMP21", #part de ménages fiscaux imposés
    "PPEN21", #Part des pensions dans le revenu fiscal (proxy pour concentration personnes âgées)
    "DECE1621", #nbr de deces entre 2016 et 2021
    "P16_LOG", #nbr de logements dans la commune en 2022
    "P16_RP", #RP en 2016
    "P16_RSECOCC", #nbr de résidences secondaires et logements occasionnels en 2016
    "P16_LOGVAC", #nbr de logements vacants en 2016 
    "P16_MAISON",
    "P16_APPART",  #Appartements en 2016 
    "P16_NSCOL15P", #Pop 15 ans ou plus non scolarisée en 2016
    "P16_ACTOCC15P", #Actifs occupés 15 ans ou plus en 2016
    "P16_CHOM1564", #Chômeurs 15-64 ans en 2016 (princ);
]

dossier_complet = pd.read_csv(
    chemin_dc,
    sep=";",
    encoding="utf-8",
    skiprows=0,
    usecols=cols_dossier_complet
)

dossier_complet["CODGEO"] = dossier_complet["CODGEO"].astype(str).str.zfill(5)
df = df.merge(
    dossier_complet,
    how="left",
    left_on="COMM",
    right_on="CODGEO",
    validate="m:1"
)

# 3. Compte des valeurs manquantes
variable_dict = {
    # Identifiers
    "COMM": "Code INSEE de la commune (autorisations)",
    "CODGEO": "Code INSEE de la commune (sources externes)",

    # Grille densité
    "DENS": "Niveau de densité communale (1 = très dense, 7 = très peu dense)",
    "PMUN17": "Population municipale 2017",

    # Socio-éco (INSEE – dossier complet)
    "TP6021": "Taux de pauvreté à 60% du niveau de vie médian (2021)",
    "MED21": "Médiane des revenus fiscaux (€) en 2021",
    "PIMP21": "Part des ménages fiscaux imposés (%)",
    "PPEN21": "Part des pensions dans le revenu fiscal (%)",
    "DECE1621": "Nombre de décès cumulés entre 2016 et 2021",
    "P16_LOG": "Nombre total de logements (2022)",
    "P16_RP": "Nombre de résidences principales (2016)",
    "P16_RSECOCC": "Résidences secondaires et logements occasionnels (2016)",
    "P16_LOGVAC": "Logements vacants (2016)",
    "P16_MAISON": "Maisons individuelles (2016)",
    "P16_APPART": "Appartements (2016)",
    "P16_NSCOL15P": "Population 15+ ans non scolarisée (2016)",
    "P16_ACTOCC15P": "Actifs occupés 15+ ans (2016)",
    "P16_CHOM1564": "Chômeurs 15–64 ans (2016)"
}

vars_added = [v for v in variable_dict.keys() if v in df.columns]

summary_table = (
    pd.DataFrame({
        "variable": vars_added,
        "description": [variable_dict[v] for v in vars_added],
        "share_na": [df[v].isna().mean() for v in vars_added]
    })
    .sort_values("share_na", ascending=False)
    .reset_index(drop=True)
)

print(summary_table)

df.to_csv("data/autorisations_enrichies.csv", index=False, sep=";")
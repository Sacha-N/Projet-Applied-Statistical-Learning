# Libraries
import pandas as pd

# 1. Chargement de Sitadel
chemin = "data/Liste-des-autorisations-logements-2025-10.csv"
df = pd.read_csv(chemin, sep=";",encoding='utf-8', 
                 skiprows=1) #nrows = 1000 pour test

print(df.shape)
print(df.head())
df.info()

print("\nColumn names:")
print(df.columns.tolist())

# 2. Retrait des variables non pertinentes

vars_mai2022 = [
    "AN_DEPOT", #"DPC_PREM", (theoriquement il faudrait la retirer, mais bon)
    "NATURE_PROJET_COMPLETEE",
    "DESTINATION_PRINCIPALE",
    "TYPE_PRINCIP_LOGTS_CREES",
    "TYPE_TRANSFO_PRINCIPAL",
    "TYPE_PRINCIP_LOCAUX_TRANSFORMES",
    "I_PISCINE",
    "I_GARAGE",
    "I_VERANDA",
    "I_ABRI_JARDIN",
    "I_AUTRE_ANNEXE",
    "RES_PERS_AGEES",
    "RES_ETUDIANTS",
    "RES_TOURISME",
    "RES_HOTEL_SOCIALE",
    "RES_SOCIALE",
    "RES_HANDICAPES",
    "RES_AUTRE",
    "NB_LGT_INDIV_PURS",
    "NB_LGT_INDIV_GROUPES",
    "NB_LGT_RES",
    "NB_LGT_COL_HORS_RES",
    "SURF_HEB_TRANSFORMEE",
    "SURF_BUR_TRANSFORMEE",
    "SURF_COM_TRANSFORMEE",
    "SURF_ART_TRANSFORMEE",
    "SURF_IND_TRANSFORMEE",
    "SURF_AGR_TRANSFORMEE",
    "SURF_ENT_TRANSFORMEE",
    "SURF_PUB_TRANSFORMEE"
]
vars_non_pertinentes = [
    "Num_DAU", "SIREN_DEM", "SIRET_DEM", 
    "DENOM_DEM", "CODPOST_DEM", "LOCALITE_DEM",
    "ADR_NUM_TER", "ADR_TYPEVOIE_TER", "ADR_LIBVOIE_TER",
    "ADR_LIEUDIT_TER", "ADR_LOCALITE_TER", "ADR_CODPOST_TER",
    "SEC_CADASTRE1", "NUM_CADASTRE1",
    "SEC_CADASTRE2", "NUM_CADASTRE2",
    "SEC_CADASTRE3", "NUM_CADASTRE3",
]


# 3. Etudions nos dates

def nettoyer_dataset(df: pd.DataFrame):
    df = df.copy()

    # Suppression des colonnes introduites en 2022
    cols_to_drop = [c for c in vars_mai2022 if c in df.columns]

    # Suppression des colonnes administratives/identifiants
    cols_to_drop += [c for c in vars_non_pertinentes if c in df.columns]

    # On retire
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # ----------------------------------------------------
    # Conversion des dates en datetime
    # ----------------------------------------------------
    for col in ["DATE_REELLE_AUTORISATION", "DATE_REELLE_DOC", "DPC_AUT", "DATE_REELLE_DAACT", "DPC_PREM"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # ----------------------------------------------------
    # 5. Construction des trois cibles
    # ----------------------------------------------------
    if "DATE_REELLE_AUTORISATION" in df.columns and "DATE_REELLE_DOC" in df.columns:
        df["delai_ouverture_chantier"] = (
            df["DATE_REELLE_DOC"] - df["DATE_REELLE_AUTORISATION"]
        ).dt.days

    if "DATE_REELLE_DAACT" in df.columns and "DATE_REELLE_DOC" in df.columns:
        df["duree_travaux"] = (
            df["DATE_REELLE_DAACT"] - df["DATE_REELLE_DOC"]
        ).dt.days

    if "DPC_AUT" in df.columns and "DPC_PREM" in df.columns:
        df["duree_obtiention_autorisation"] = (
            df["DPC_PREM"] - df["DPC_AUT"]
        ).dt.days

    # Suppression des lignes sans cibles
    df = df.dropna(subset=["delai_ouverture_chantier", "duree_travaux","duree_obtiention_autorisation"], how="all")

    return df

df_clean = nettoyer_dataset(df)
#df_clean.to_excel("data/output.xlsx", index=False)
df_clean.to_csv("data/output.csv", index=False, sep=";")

## 4. Comparaison
# Statistics before cleaning
print("Original df shape:", df.shape)
print("Total lines in original df:", len(df))

# Statistics after cleaning
print("\nAfter cleaning:")
print("Lines with non-NA delai_ouverture_chantier:", df_clean["delai_ouverture_chantier"].notna().sum())
print("Lines with non-NA duree_travaux:", df_clean["duree_travaux"].notna().sum())
print("Lines with non-NA duree_obtiention_autorisation:", df_clean["duree_obtiention_autorisation"].notna().sum())
print("\nFinal cleaned df shape:", df_clean.shape)
print("Final cleaned df lines:", len(df_clean))

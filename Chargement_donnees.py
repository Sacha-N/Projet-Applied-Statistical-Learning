# Libraries
import pandas as pd

# 1. Chargement de Sitadel
chemin = "data/Liste-des-autorisations-durbanisme-creant-des-logements.2025-10.csv"

# 1.1 Exclusion ex ante des colonnes non pertinentes
all_cols = pd.read_csv(chemin, sep=";", skiprows=1, nrows=1).columns.tolist()

vars_mai2022 = [
    "AN_DEPOT",  # "DPC_PREM", (theoriquement il faudrait la retirer, mais bon)
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
    "SURF_PUB_TRANSFORMEE",
]
vars_non_pertinentes = [
    "Num_DAU",
    "SIREN_DEM",
    "SIRET_DEM",
    "DENOM_DEM",
    "CODPOST_DEM",
    "LOCALITE_DEM",
    "ADR_NUM_TER",
    "ADR_TYPEVOIE_TER",
    "ADR_LIBVOIE_TER",
    "ADR_LIEUDIT_TER",
    "ADR_LOCALITE_TER",
    "ADR_CODPOST_TER",
    "SEC_CADASTRE1",
    "NUM_CADASTRE1",
    "SEC_CADASTRE2",
    "NUM_CADASTRE2",
    "SEC_CADASTRE3",
    "NUM_CADASTRE3",
]

cols_to_drop = set(vars_mai2022 + vars_non_pertinentes)
use_cols = [c for c in all_cols if c not in cols_to_drop]

# 1.2 Chargement avec les colonnes filtrées
df = pd.read_csv(
    chemin,
    sep=";",
    encoding="utf-8",
    skiprows=1,
    usecols=use_cols
)

# 3. Etudions nos dates
date_cols = [
    "DATE_REELLE_AUTORISATION",
    "DATE_REELLE_DOC",
    "DPC_AUT",
    "DATE_REELLE_DAACT",
    "DPC_PREM",
]

for col in date_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Type: {df[col].dtype}")
        print(f"  Sample values:")
        print(df[col].head(10).tolist())
        print(f"  Null count: {df[col].isna().sum()}")
    else:
        print(f"\n{col}: NOT FOUND in dataframe")


## Puis, nettoyons le dataset
def nettoyer_dataset(df: pd.DataFrame):
    df = df.copy()

    # Suppression des colonnes introduites en 2022
    cols_to_drop = [c for c in vars_mai2022 if c in df.columns]

    # Suppression des colonnes administratives/identifiants
    cols_to_drop += [c for c in vars_non_pertinentes if c in df.columns]

    # On retire
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # ----------------------------------------------------
    # Conversion des dates en datetime (robuste)
    # ----------------------------------------------------
    # DD/MM/YYYY-like columns: be lenient (dayfirst) and coerce errors
    dmY_cols = ["DATE_REELLE_AUTORISATION", "DATE_REELLE_DOC", "DATE_REELLE_DAACT"]
    for col in dmY_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col], errors="coerce", dayfirst=True, infer_datetime_format=True
            )

    # YYYY-MM format (year-month only)
    for col in ["DPC_AUT", "DPC_PREM"]:
        if col in df.columns:
            # ensure strings and strip whitespace, coerce bad values
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip(), errors="coerce", format="%Y-%m"
            )

    # Remove unrealistic dates that would cause overflow when converting to ns
    min_date = pd.Timestamp("1900-01-01")  # vérifier cette data / ce format
    max_date = pd.Timestamp("2025-12-31")
    for col in dmY_cols + ["DPC_AUT", "DPC_PREM"]:
        if col in df.columns:
            mask = (df[col] < min_date) | (df[col] > max_date)
            df.loc[mask, col] = pd.NaT

    # ----------------------------------------------------
    # 5. Construction des trois cibles (only where both dates valid)
    # ----------------------------------------------------
    if "DATE_REELLE_AUTORISATION" in df.columns and "DATE_REELLE_DOC" in df.columns:
        mask = df["DATE_REELLE_AUTORISATION"].notna() & df["DATE_REELLE_DOC"].notna()
        df.loc[mask, "delai_ouverture_chantier"] = (
            df.loc[mask, "DATE_REELLE_DOC"] - df.loc[mask, "DATE_REELLE_AUTORISATION"]
        ).dt.days

    if "DATE_REELLE_DAACT" in df.columns and "DATE_REELLE_DOC" in df.columns:
        mask = df["DATE_REELLE_DAACT"].notna() & df["DATE_REELLE_DOC"].notna()
        df.loc[mask, "duree_travaux"] = (
            df.loc[mask, "DATE_REELLE_DAACT"] - df.loc[mask, "DATE_REELLE_DOC"]
        ).dt.days

    if "DPC_AUT" in df.columns and "DPC_PREM" in df.columns:
        mask = df["DPC_PREM"].notna() & df["DPC_AUT"].notna()
        df.loc[mask, "duree_obtiention_autorisation"] = (
            df.loc[mask, "DPC_AUT"] - df.loc[mask, "DPC_PREM"]
        ).dt.days

    # Convert durations to numeric and treat non-positive values as missing
    duration_cols = [
        "delai_ouverture_chantier",
        "duree_travaux",
        "duree_obtiention_autorisation",
    ]
    for col in duration_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # consider zero and negative durations invalid -> set to NA
            df.loc[df[col] <= 0, col] = pd.NA

    # Suppression des lignes sans cibles valides (NA, null, zero or negative removed above)
    existing_duration_cols = [c for c in duration_cols if c in df.columns]
    df = df.dropna(subset=existing_duration_cols, how="all")

    return df


df_clean = nettoyer_dataset(df)
# df_clean.to_excel("data/output.xlsx", index=False)
df_clean.to_csv("data/output.csv", index=False, sep=";")

## 4. Comparaison
# Statistics before cleaning
print("Original df shape:", df.shape)
print("Total lines in original df:", len(df))

# Statistics after cleaning
print("\nAfter cleaning:")
print(
    "Lines with non-NA delai_ouverture_chantier:",
    df_clean["delai_ouverture_chantier"].notna().sum(),
)
print("Lines with non-NA duree_travaux:", df_clean["duree_travaux"].notna().sum())
print(
    "Lines with non-NA duree_obtiention_autorisation:",
    df_clean["duree_obtiention_autorisation"].notna().sum(),
)
print(
    "Lines with non-0 duree_obtiention_autorisation:",
    (
        df_clean["duree_obtiention_autorisation"].notna()
        & (df_clean["duree_obtiention_autorisation"] != 0)
    ).sum(),
)
print("\nFinal cleaned df shape:", df_clean.shape)
print("Final cleaned df lines:", len(df_clean))

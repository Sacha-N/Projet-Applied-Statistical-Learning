import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##################################################
# --- 1. Préparation des Données et Variables ---
##################################################

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
    "duree_travaux", 
    "annee_autorisation"]

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

df = pd.read_csv("data/output.csv", sep=";")
#print(df.head())
#df.columns

# Filtrage des régions et des années
regions_outremer = [
    "Guadeloupe", "Martinique", "Guyane",
    "La Réunion", "Mayotte"
]
df["DATE_REELLE_AUTORISATION"] = pd.to_datetime(df["DATE_REELLE_AUTORISATION"], errors="coerce")
df["annee_autorisation"] = df["DATE_REELLE_AUTORISATION"].dt.year
df = df[~df["REG_LIBELLE"].isin(regions_outremer)]
df = df[df["annee_autorisation"] >= 2020]

# Filtrage des lignes sans la variable cible
df_filtre_delai_ouverture = df.dropna(subset=["delai_ouverture_chantier"])

# Nettoyage et conversion des types sur l'échantillon
df_filtre_delai_ouverture = df_filtre_delai_ouverture.drop(columns=vars_inutiles_delai_ouverture, errors="ignore")
df_filtre_delai_ouverture[vars_categ] = df_filtre_delai_ouverture[vars_categ].astype("string").fillna("Missing")

# Définition des variables X et y
y = df_filtre_delai_ouverture["delai_ouverture_chantier"]
X = df_filtre_delai_ouverture.drop(columns=["delai_ouverture_chantier"])

# Séparation des colonnes numériques et catégorielles après nettoyage
num_cols = X.select_dtypes(include=["float", "int"]).columns
cat_cols = X.select_dtypes(include=["string"]).columns

#####################################################
# --- 2. Création du Pipeline de Pré-traitement -----
#####################################################

preprocess = ColumnTransformer(
    transformers=[
        # Mise à l'échelle des variables numériques (StandardScaler)
        ("num", StandardScaler(), num_cols), 
        # Encodage One-Hot des variables catégorielles (handle_unknown='ignore' pour les valeurs futures non vues)
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder='passthrough' # Ne rien faire avec les colonnes restantes
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#########################
# --- 3. Modèle LASSO ---
#########################

lasso = Lasso(alpha=0.1, max_iter=10000)

lasso_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", lasso)
])

lasso_pipeline.fit(X_train, y_train)

y_train_pred = lasso_pipeline.predict(X_train)
y_test_pred = lasso_pipeline.predict(X_test)

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

# résultats pas fous mais positif que train et test soient similaires 

coefs = lasso_pipeline.named_steps["model"].coef_
nb_nonzero = np.sum(coefs != 0)
print("Nombre de coefficients non nuls :", nb_nonzero)

# Liste des coeff non nuls 
feature_names = list(num_cols) + list(cat_cols)
coefs = lasso_pipeline.named_steps["model"].coef_
nonzero_idx = coefs != 0
selected_features = [(name, coef) for name, coef in zip(feature_names, coefs) if coef != 0]
for name, coef in selected_features:
    print(f"{name}: {coef:.4f}")

#############################
# --- 4. Random forest ------
#############################

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(
        max_depth=12, 
        min_samples_split=10,
        min_samples_leaf=4,
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    ))
]) #On peut modifier la profondeur peut être pour améliorer les résultats 

rf_pipeline.fit(X_train, y_train)

y_train_pred_rf = rf_pipeline.predict(X_train)
y_test_pred_rf  = rf_pipeline.predict(X_test)

train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
test_rmse_rf  = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))

train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
test_mae_rf  = mean_absolute_error(y_test, y_test_pred_rf)

train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf  = r2_score(y_test, y_test_pred_rf)

print(f"RMSE train : {train_rmse_rf:.2f}")
print(f"RMSE test  : {test_rmse_rf:.2f}")
print(f"MAE train  : {train_mae_rf:.2f}")
print(f"MAE test   : {test_mae_rf:.2f}")
print(f"R² train   : {train_r2_rf:.3f}")
print(f"R² test    : {test_r2_rf:.3f}")

# Résultats un peu meilleur que le lasso, mais R² pas fou

#################################
# --- 5. Grandient Boosting -----
#################################

from sklearn.ensemble import GradientBoostingRegressor

gb_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

gb_pipeline.fit(X_train, y_train)

y_train_pred_gb = gb_pipeline.predict(X_train)
y_test_pred_gb  = gb_pipeline.predict(X_test)

train_rmse_gb = np.sqrt(mean_squared_error(y_train, y_train_pred_gb))
test_rmse_gb  = np.sqrt(mean_squared_error(y_test, y_test_pred_gb))

train_mae_gb = mean_absolute_error(y_train, y_train_pred_gb)
test_mae_gb  = mean_absolute_error(y_test, y_test_pred_gb)

train_r2_gb = r2_score(y_train, y_train_pred_gb)
test_r2_gb  = r2_score(y_test, y_test_pred_gb)

print(f"RMSE train : {train_rmse_gb:.2f}")
print(f"RMSE test  : {test_rmse_gb:.2f}")
print(f"MAE train  : {train_mae_gb:.2f}")
print(f"MAE test   : {test_mae_gb:.2f}")
print(f"R² train   : {train_r2_gb:.3f}")
print(f"R² test    : {test_r2_gb:.3f}")

##########################
# --- 6. KNN Regressor ---
##########################

# Beaucoup trop lent sur la taille de notre base 

from sklearn.neighbors import KNeighborsRegressor

knn_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", KNeighborsRegressor(n_neighbors=10))
])

knn_pipeline.fit(X_train, y_train)

y_train_pred_knn = knn_pipeline.predict(X_train)
y_test_pred_knn  = knn_pipeline.predict(X_test)

train_rmse_knn = np.sqrt(mean_squared_error(y_train, y_train_pred_knn))
test_rmse_knn  = np.sqrt(mean_squared_error(y_test, y_test_pred_knn))

train_mae_knn = mean_absolute_error(y_train, y_train_pred_knn)
test_mae_knn  = mean_absolute_error(y_test, y_test_pred_knn)

train_r2_knn = r2_score(y_train, y_train_pred_knn)
test_r2_knn  = r2_score(y_test, y_test_pred_knn)

print(f"RMSE train : {train_rmse_knn:.2f}")
print(f"RMSE test  : {test_rmse_knn:.2f}")
print(f"MAE train  : {train_mae_knn:.2f}")
print(f"MAE test   : {test_mae_knn:.2f}")
print(f"R² train   : {train_r2_knn:.3f}")
print(f"R² test    : {test_r2_knn:.3f}")

####################
# --- 7. SVM/SVR ---
####################

from sklearn.svm import SVR
from sklearn.svm import LinearSVR

svr_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearSVR(C=1.0, epsilon=1.0, random_state=0))
])

svr_pipeline.fit(X_train, y_train)

y_train_pred_svr = svr_pipeline.predict(X_train)
y_test_pred_svr  = svr_pipeline.predict(X_test)

train_rmse_svr = np.sqrt(mean_squared_error(y_train, y_train_pred_svr))
test_rmse_svr  = np.sqrt(mean_squared_error(y_test, y_test_pred_svr))

train_mae_svr = mean_absolute_error(y_train, y_train_pred_svr)
test_mae_svr  = mean_absolute_error(y_test, y_test_pred_svr)

train_r2_svr = r2_score(y_train, y_train_pred_svr)
test_r2_svr  = r2_score(y_test, y_test_pred_svr)

print(f"RMSE train : {train_rmse_svr:.2f}")
print(f"RMSE test  : {test_rmse_svr:.2f}")
print(f"MAE train  : {train_mae_svr:.2f}")
print(f"MAE test   : {test_mae_svr:.2f}")
print(f"R² train   : {train_r2_svr:.3f}")
print(f"R² test    : {test_r2_svr:.3f}")


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Étape 1 : Chargement du fichier CSV
df = pd.read_csv("dataset/titanic.csv")

print(f"Dimensions initiales du dataset : {df.shape[0]} lignes et {df.shape[1]} colonnes.")
print(df.head(3))

# Étape 2 : Liste des colonnes à exclure des variables explicatives X
colonnes_a_supprimer = [
    "Survived",     # Variable cible (placée dans y)
    "boat",         # Fuite de données (Data Leakage)
    "body",         # Fuite de données (Data Leakage)
    "PassengerId",  # Identifiant unique sans valeur prédictive
    "Name",         # Texte brut à haute cardinalité
    "Ticket",       # Code alphanumérique à haute cardinalité
    "Cabin",        # 77.5 % de valeurs manquantes
    "home.dest"     # 43.1 % de valeurs manquantes et texte libre
]

# Création de X (variables explicatives) et y (variable cible)
X = df.drop(columns=colonnes_a_supprimer)
y = df["Survived"]

print("\nColonnes conservées dans X :", list(X.columns))
print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de y : {y.shape}")

# Étape 3 : Découpage en deux ensembles (Train et Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nEnsemble d'entraînement (Train) : X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Ensemble de test        (Test)  : X_test  = {X_test.shape},  y_test  = {y_test.shape}")

print("\nVérification de la stratification (proportion de survivants) :")
print(f" - Dans y_train : {y_train.mean() * 100:.2f} %")
print(f" - Dans y_test  : {y_test.mean() * 100:.2f} %")

# Étape 4 : Pipeline pour les variables numériques standards (Age, SibSp, Parch)
colonnes_numeriques_normales = ["Age", "SibSp", "Parch"]

pipeline_numerique = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("standardisation", StandardScaler())
])

# Étape 5 : Pipeline spécifique pour la variable Fare
colonne_fare = ["Fare"]

pipeline_fare = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("logarithme", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("standardisation", StandardScaler())
])

# Étape 6 : Pipeline pour la variable binaire Sex
colonne_sexe = ["Sex"]

pipeline_sexe = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
    ("encodage", OrdinalEncoder(categories=[["male", "female"]]))
])

# Étape 7 : Pipeline pour les variables catégorielles (Pclass, Embarked)
colonnes_categorielles = ["Pclass", "Embarked"]

pipeline_categoriel = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
    ("encodage", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

# Étape 8 : Regroupement des 4 branches de traitement dans le ColumnTransformer
pretraitement = ColumnTransformer(
    transformers=[
        ("numerique", pipeline_numerique, colonnes_numeriques_normales),
        ("fare", pipeline_fare, colonne_fare),
        ("sexe", pipeline_sexe, colonne_sexe),
        ("categoriel", pipeline_categoriel, colonnes_categorielles)
    ]
)

# Encapsulation dans le pipeline final de prétraitement
pipeline_final = Pipeline([
    ("pretraitement", pretraitement)
])

# Étape 9 : Exécution du prétraitement
# 1. Apprentissage des statistiques et transformation sur le jeu d'entraînement
X_train_prepare = pipeline_final.fit_transform(X_train)

# 2. Application stricte des mêmes transformations sur le jeu de test
X_test_prepare = pipeline_final.transform(X_test)

print("\nDimensions de X_train_prepare :", X_train_prepare.shape)
print("Dimensions de X_test_prepare  :", X_test_prepare.shape)

# Récupération automatique des noms des 9 colonnes après transformation
noms_colonnes_finales = pipeline_final.named_steps["pretraitement"].get_feature_names_out()

# Conversion en DataFrame pandas pour une lecture claire
df_train_prepare = pd.DataFrame(X_train_prepare, columns=noms_colonnes_finales, index=X_train.index)
df_test_prepare = pd.DataFrame(X_test_prepare, columns=noms_colonnes_finales, index=X_test.index)

print("\nListe des 9 variables finales envoyées au SVM :")
for i, col in enumerate(noms_colonnes_finales, 1):
    print(f"  {i}. {col}")

print("\nContrôle de sécurité avant le SVM :")
print("Nombre total de valeurs manquantes (NaN) dans X_train_prepare :", np.isnan(X_train_prepare).sum())
print("Nombre total de valeurs manquantes (NaN) dans X_test_prepare  :", np.isnan(X_test_prepare).sum())

# Étape 10 : Exportation des ensembles Train et Test au format CSV
train_final_df = df_train_prepare.copy()
train_final_df["Survived"] = y_train.values

test_final_df = df_test_prepare.copy()
test_final_df["Survived"] = y_test.values

chemin_train = "dataset/train.csv"
chemin_test = "dataset/test.csv"

train_final_df.to_csv(chemin_train, index=False)
test_final_df.to_csv(chemin_test, index=False)

print(f"\nFichier d'entraînement exporté : {chemin_train} ({train_final_df.shape[0]} lignes, {train_final_df.shape[1]} colonnes)")
print(f"Fichier de test exporté        : {chemin_test} ({test_final_df.shape[0]} lignes, {test_final_df.shape[1]} colonnes)")
print("Prétraitement terminé avec succès.")

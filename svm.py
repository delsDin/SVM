"""
Projet : Etude et application de l'algorithme Support Vector Machine (SVM)
Jeu de données : Titanic (1309 passagers)
Description :
    Ce script realise l'ensemble de la chaine de traitement en Python :
    1. Chargement des données brutes (dataset/titanic.csv).
    2. Suppression des colonnes non pertinentes et des fuites de données.
    3. Séparation stratifiée en jeu d'entrainement (80 %) et jeu de test (20 %).
    4. Construction du pipeline de prétraitement (imputation, log1p, standardisation, encodage).
    5. Exportation des données préparées vers dataset/train.csv et dataset/test.csv.
    6. Séparation interne pour la validation et recherche des hyperparamètres avec GridSearchCV.
    7. Evaluation sur le jeu de validation puis réentrainement sur tout le jeu d'entrainement.
    8. Evaluation finale sur le jeu de test et analyse détaillée des résultats.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# PARTIE A : PRETRAITEMENT DES DONNEES

# Etape 1 : Chargement du fichier CSV brut
df = pd.read_csv("dataset/titanic.csv")

print("\nPARTIE A : PRETRAITEMENT DES DONNEES\n\n")
print(f"\nDimensions initiales du dataset : {df.shape[0]} lignes et {df.shape[1]} colonnes.\n")

# Etape 2 : Sélection des variables explicatives (X) et de la cible (y)
# On retire la cible Survived, les fuites de données (boat, body),
# l'identifiant (PassengerId) et les colonnes textuelles bruitees ou trop vides.
colonnes_a_supprimer = [
    "Survived",
    "boat",
    "body",
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin",
    "home.dest"
]

X_brut = df.drop(columns=colonnes_a_supprimer)
y_brut = df["Survived"]

print("Colonnes conservees dans X :", list(X_brut.columns))

# Etape 3 : Découpage stratifié en Entrainement (80 %) et Test (20 %)
# Ce découpage est fait avant tout calcul statistique pour eviter la fuite de données.
X_train_brut, X_test_brut, y_train, y_test = train_test_split(
    X_brut,
    y_brut,
    test_size=0.2,
    random_state=42,
    stratify=y_brut
)

print(f"Ensemble d'entrainement : {X_train_brut.shape[0]} passagers (survie : {y_train.mean() * 100:.2f} %)")
print(f"Ensemble de test        : {X_test_brut.shape[0]} passagers (survie : {y_test.mean() * 100:.2f} %)")

# Etape 4 : Définition des quatre branches du pipeline de prétraitement

# Branche 1 : Variables numeriques classiques (Age, SibSp, Parch)
colonnes_numeriques_normales = ["Age", "SibSp", "Parch"]
pipeline_numerique = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("standardisation", StandardScaler())
])

# Branche 2 : Variable Fare (asymétrie corrigée avec log1p avant standardisation)
colonne_fare = ["Fare"]
pipeline_fare = Pipeline([
    ("imputation", SimpleImputer(strategy="median")),
    ("logarithme", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("standardisation", StandardScaler())
])

# Branche 3 : Variable binaire Sex (male -> 0, female -> 1)
colonne_sexe = ["Sex"]
pipeline_sexe = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
    ("encodage", OrdinalEncoder(categories=[["male", "female"]]))
])

# Branche 4 : Variables categorielles Pclass et Embarked (encodage One-Hot)
colonnes_categorielles = ["Pclass", "Embarked"]
pipeline_categoriel = Pipeline([
    ("imputation", SimpleImputer(strategy="most_frequent")),
    ("encodage", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

# Assemblage dans un ColumnTransformer
pretraitement = ColumnTransformer(
    transformers=[
        ("numerique", pipeline_numerique, colonnes_numeriques_normales),
        ("fare", pipeline_fare, colonne_fare),
        ("sexe", pipeline_sexe, colonne_sexe),
        ("categoriel", pipeline_categoriel, colonnes_categorielles)
    ]
)

# Pipeline final de prétraitement
pipeline_pretraitement = Pipeline([
    ("pretraitement", pretraitement)
])

# Etape 5 : Ajustement sur le jeu d'entrainement et transformation du jeu de test
X_train_prepare = pipeline_pretraitement.fit_transform(X_train_brut)
X_test_prepare = pipeline_pretraitement.transform(X_test_brut)

noms_colonnes_finales = pipeline_pretraitement.named_steps["pretraitement"].get_feature_names_out()

X_train = pd.DataFrame(X_train_prepare, columns=noms_colonnes_finales)
X_test = pd.DataFrame(X_test_prepare, columns=noms_colonnes_finales)

print(f"\nDimensions après prétraitement : X_train = {X_train.shape}, X_test = {X_test.shape}")
print("Valeurs manquantes dans X_train :", X_train.isnull().sum().sum())
print("Valeurs manquantes dans X_test  :", X_test.isnull().sum().sum())

# Etape 6 : Sauvegarde des fichiers CSV préparés dans le dossier dataset
train_final_df = X_train.copy()
train_final_df["Survived"] = y_train.values

test_final_df = X_test.copy()
test_final_df["Survived"] = y_test.values

train_final_df.to_csv("dataset/train.csv", index=False)
test_final_df.to_csv("dataset/test.csv", index=False)
print("Fichiers dataset/train.csv et dataset/test.csv sauvegardés avec succès.\n")


# PARTIE B : MODELISATION SVM ET OPTIMISATION DES HYPERPARAMETRES

print("\nPARTIE B : MODELISATION SVM ET RECHERCHE DES HYPERPARAMETRES\n")

# Etape 7 : Séparation interne Entrainement (837) / Validation (210)
X_train_local, X_val, y_train_local, y_val = train_test_split(
    X_train,
    y_train.reset_index(drop=True),
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Sous-ensemble d'entrainement interne : {X_train_local.shape[0]} observations")
print(f"Sous-ensemble de validation interne  : {X_val.shape[0]} observations")

# Etape 8 : Construction du pipeline SVM et de la grille d'hyperparamètres
pipeline_svm = Pipeline([
    ("classificateur", SVC())
])

# Grille de 42 configurations testant les noyaux linear, rbf et poly
parametres = [
    {
        "classificateur__kernel": ["linear"],
        "classificateur__C": [0.1, 1, 10, 100]
    },
    {
        "classificateur__kernel": ["rbf"],
        "classificateur__C": [0.1, 1, 10, 100],
        "classificateur__gamma": ["scale", "auto", 0.01, 0.1, 1]
    },
    {
        "classificateur__kernel": ["poly"],
        "classificateur__C": [0.1, 1, 10],
        "classificateur__degree": [2, 3, 4],
        "classificateur__gamma": ["scale", "auto"]
    }
]

validation_croisee = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

recherche = GridSearchCV(
    estimator=pipeline_svm,
    param_grid=parametres,
    scoring="f1",
    cv=validation_croisee,
    n_jobs=-1,
    verbose=1
)

print("\nLancement de GridSearchCV (42 configurations x 5 plis = 210 ajustements)...")
recherche.fit(X_train_local, y_train_local)

print("\nMeilleurs hyperparamètres trouvés :")
print(recherche.best_params_)
print(f"Meilleur score F1 en validation croisée : {recherche.best_score_:.4f}")


# PARTIE C : EVALUATION SUR LA VALIDATION ET SUR LE TEST FINAL

# Etape 9 : Evaluation sur le groupe de validation (210 observations)
meilleur_modele = recherche.best_estimator_
y_val_pred = meilleur_modele.predict(X_val)

print("\nEVALUATION SUR L'ENSEMBLE DE VALIDATION (210 PASSAGERS)")
print(f"Accuracy  : {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision : {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall    : {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-score  : {f1_score(y_val, y_val_pred):.4f}")
print("Matrice de confusion (Validation) :")
print(confusion_matrix(y_val, y_val_pred))

# Etape 10 : Réentrainement du meilleur modèle sur les 1047 observations d'entrainement
print("\nRéentrainement du meilleur modèle sur la totalite des 1047 passagers d'entrainement...")
meilleur_modele.fit(X_train, y_train.reset_index(drop=True))

# Etape 11 : Evaluation finale sur le jeu de test (262 observations)
y_test_reset = y_test.reset_index(drop=True)
y_test_pred = meilleur_modele.predict(X_test)
y_test_scores = meilleur_modele.decision_function(X_test)

print("\nEVALUATION FINALE SUR L'ENSEMBLE DE TEST (262 PASSAGERS)")
print(f"Accuracy  : {accuracy_score(y_test_reset, y_test_pred):.4f} ({accuracy_score(y_test_reset, y_test_pred) * 100:.2f} %)")
print(f"Precision : {precision_score(y_test_reset, y_test_pred):.4f} ({precision_score(y_test_reset, y_test_pred) * 100:.2f} %)")
print(f"Recall    : {recall_score(y_test_reset, y_test_pred):.4f} ({recall_score(y_test_reset, y_test_pred) * 100:.2f} %)")
print(f"F1-score  : {f1_score(y_test_reset, y_test_pred):.4f} ({f1_score(y_test_reset, y_test_pred) * 100:.2f} %)")
print(f"ROC-AUC   : {roc_auc_score(y_test_reset, y_test_scores):.4f} ({roc_auc_score(y_test_reset, y_test_scores) * 100:.2f} %)")

print("\nRapport de classification detaille (Test) :")
print(classification_report(y_test_reset, y_test_pred))

matrice_test = confusion_matrix(y_test_reset, y_test_pred)
print("Matrice de confusion (Test) :")
print(matrice_test)


# PARTIE D : INTERPRETATION DU MODELE

modele_svc = meilleur_modele.named_steps["classificateur"]
nb_supports = modele_svc.n_support_
print("\nELEMENTS D'INTERPRETATION DU MODELE SVM")
print(f"Nombre de vecteurs de support : {nb_supports.sum()} sur {len(X_train)} passagers d'entrainement")
print(f"  Classe 0 (Décédés)    : {nb_supports[0]} vecteurs de support")
print(f"  Classe 1 (Survivants) : {nb_supports[1]} vecteurs de support")

# Calcul de l'importance des variables par permutation sur le jeu de test
importance = permutation_importance(
    meilleur_modele,
    X_test,
    y_test_reset,
    scoring="f1",
    n_repeats=20,
    random_state=42
)

print("\nClassement des variables par importance (impact sur le F1-score) :")
indices_tries = importance.importances_mean.argsort()[::-1]
for idx in indices_tries:
    nom_var = X_test.columns[idx]
    baisse_f1 = importance.importances_mean[idx]
    print(f"  {nom_var:<25} : {baisse_f1:.4f}")

print("\nExécution complète du projet terminée avec succès.")

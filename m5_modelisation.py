
# ============================================================
# PROJET SVM - TITANIC
# M5 : OPTIMISATION ET EVALUATION FINALE
# ============================================================

# ============================================================
# 1. IMPORTATIONS ET CHARGEMENT DES DONNEES
# ============================================================

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# Chargement des jeux de données prétraités
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# Séparation des variables explicatives et de la cible
X_train = train_df.drop(columns=["Survived"])
y_train = train_df["Survived"]

X_test = test_df.drop(columns=["Survived"])
y_test = test_df["Survived"]

print("Dimensions des données d'entraînement :", X_train.shape)
print("Dimensions des données de test :", X_test.shape)
print("\nColonnes utilisées :", list(X_train.columns))
print("\nValeurs manquantes dans X_train :", X_train.isnull().sum().sum())
print("Valeurs manquantes dans X_test :", X_test.isnull().sum().sum())

# ============================================================
# 2. SEPARATION ENTRAINEMENT / VALIDATION
# ============================================================

X_train_local, X_val, y_train_local, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


# ============================================================
# 3. CONSTRUCTION DU PIPELINE
# ============================================================

pipeline_svm = Pipeline([
    ('classificateur', SVC())
])

# ============================================================
# 4. GRILLE DES HYPERPARAMETRES
# ============================================================

parametres = [
    {
        'classificateur__kernel': ['linear'],
        'classificateur__C': [0.1, 1, 10, 100]
    },
    {
        'classificateur__kernel': ['rbf'],
        'classificateur__C': [0.1, 1, 10, 100],
        'classificateur__gamma': [
            'scale', 'auto', 0.01, 0.1, 1
        ]
    },
    {
        'classificateur__kernel': ['poly'],
        'classificateur__C': [0.1, 1, 10],
        'classificateur__degree': [2, 3, 4],
        'classificateur__gamma': ['scale', 'auto']
    }
]


# ============================================================
# 5. VALIDATION CROISEE
# ============================================================

validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. RECHERCHE DES MEILLEURS PARAMETRES
# ============================================================

recherche = GridSearchCV(
    estimator=pipeline_svm,
    param_grid=parametres,
    scoring='f1',
    cv=validation,
    n_jobs=-1,
    verbose=1
)

print("\nRecherche des meilleurs hyperparamètres...")

recherche.fit(X_train_local, y_train_local)

print("\nMeilleurs paramètres :")
print(recherche.best_params_)

print("\nMeilleur F1-score en validation croisée :")
print(f"{recherche.best_score_:.4f}")


# ============================================================
# 7. EVALUATION SUR LE GROUPE DE VALIDATION
# ============================================================

meilleur_modele = recherche.best_estimator_

y_val_pred = meilleur_modele.predict(X_val)

print("\n===== EVALUATION SUR LA VALIDATION =====")

print("Accuracy :", accuracy_score(y_val, y_val_pred))
print("Precision :", precision_score(y_val, y_val_pred))
print("Recall :", recall_score(y_val, y_val_pred))
print("F1-score :", f1_score(y_val, y_val_pred))

print("\nRapport de classification :")
print(classification_report(y_val, y_val_pred))

print("Matrice de confusion :")
print(confusion_matrix(y_val, y_val_pred))


# ============================================================
# 8. REENTRAINEMENT SUR LES 1047 OBSERVATIONS
# ============================================================

print("\nRéentraînement sur les 1047 observations...")

meilleur_modele.fit(X_train, y_train)


# ============================================================
# 9. EVALUATION FINALE SUR LES 262 OBSERVATIONS
# ============================================================

y_test_pred = meilleur_modele.predict(X_test)

# Scores de décision pour le ROC-AUC
y_test_scores = meilleur_modele.decision_function(X_test)

print("\n======================================")
print("       EVALUATION FINALE")
print("       SUR LES 262 OBSERVATIONS")
print("======================================")

print("\nAccuracy :", accuracy_score(y_test, y_test_pred))

print("Precision :", precision_score(y_test, y_test_pred))

print("Recall :", recall_score(y_test, y_test_pred))

print("F1-score :", f1_score(y_test, y_test_pred))

print("ROC-AUC :", roc_auc_score(y_test, y_test_scores))

print("\nRapport de classification :")
print(classification_report(y_test, y_test_pred))

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_test_pred))

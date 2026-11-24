import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# Chargement du dataset
df = pd.read_csv("titanic.csv")

# Séparation des données d'entraînement et de test
df_train=df.iloc[:891].copy() # sélection des 891 premieres lignes
df_test=df.iloc[891:].copy() #sélection des 418 lignes restantes sur les 1309

# Séparation des variables explicatives et la cible

X_train = df_train.drop(columns=[
    'Survived', 'PassengerId', 'Name', 'Ticket',
    'Cabin', 'boat', 'body', 'home.dest'
])

y_train = df_train['Survived']

X_test = df_test.drop(columns=[
    'Survived', 'PassengerId', 'Name', 'Ticket',
    'Cabin', 'boat', 'body', 'home.dest'
])

y_test = df_test['Survived']

# Définition des variables numériques
colonnes_normales = ['Age', 'SibSp', 'Parch']
colonne_fare = ['Fare']

# Prétraitement des variables numériques
pipeline_numerique = Pipeline([
    ('imputation', SimpleImputer(strategy='median')),
    ('standardisation', StandardScaler())
])

# Prétraitement de Fare
pipeline_fare = Pipeline([
    ('imputation', SimpleImputer(strategy='median')),
    ('logarithme', FunctionTransformer(np.log1p)),
    ('standardisation', StandardScaler())
])

# Encodage de Sex
pipeline_sexe = Pipeline([
    ('imputation', SimpleImputer(strategy='most_frequent')),
    ('encodage', OrdinalEncoder(
        categories=[['male', 'female']]
    ))
])

# Encodage de Pclass et Embarked
colonnes_categorielles = ['Pclass', 'Embarked']

pipeline_categoriel = Pipeline([
    ('imputation', SimpleImputer(strategy='most_frequent')),
    ('encodage', OneHotEncoder(
        drop='first',
        handle_unknown='ignore'
    ))
])

# Regroupement des traitements
pretraitement = ColumnTransformer([
    ('numerique', pipeline_numerique, colonnes_normales),
    ('fare', pipeline_fare, colonne_fare),
    ('sexe', pipeline_sexe, ['Sex']),
    ('categoriel', pipeline_categoriel, colonnes_categorielles)
])

# Création du pipeline final
pipeline_final = Pipeline([
    ('pretraitement', pretraitement)
])

# Les transformations sur les données d'entraînement
X_train_prepare = pipeline_final.fit_transform(X_train)

# Application des transformations aux données de test
X_test_prepare = pipeline_final.transform(X_test)

# Affichage des résultats
print("Dimensions des données d'entraînement :",
      X_train_prepare.shape)

print("Dimensions des données de test :",
      X_test_prepare.shape)

print("Prétraitement terminé avec succès.")
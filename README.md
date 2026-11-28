# Projet SVM sur le jeu de données Titanic

## 1. Présentation du projet

Ce projet porte sur l'étude et l'application de l'algorithme **Support Vector Machine (SVM)** pour résoudre un problème de classification binaire à partir du jeu de données **Titanic**.

L'objectif est de prédire si un passager a survécu (`Survived = 1`) ou non (`Survived = 0`) au naufrage, en suivant toutes les étapes d'un projet de Machine Learning :
* Analyse exploratoire des données (EDA) sans modification du fichier d'origine.
* Nettoyage et suppression des colonnes non pertinentes ou contenant une fuite de données (*Data Leakage*).
* Découpage stratifié en ensemble d'entraînement (80 %) et ensemble de test (20 %).
* Traitement des valeurs manquantes, correction de l'asymétrie, standardisation et encodage des catégories dans un pipeline `scikit-learn`.
* Entraînement du modèle SVM (`SVC`) et recherche des meilleurs hyperparamètres par validation croisée (`GridSearchCV`).
* Évaluation des performances et interprétation des résultats sur le jeu de test.

## 2. Jeu de données

Le fichier complet se trouve dans `dataset/titanic.csv` et regroupe **1 309 passagers** décrits par **15 colonnes**.

La variable cible est `Survived` :
* `0` : Passager décédé (809 passagers, soit 61,80 %).
* `1` : Passager survivant (500 passagers, soit 38,20 %).

Lors du prétraitement, le jeu de données est séparé avec `train_test_split` (`test_size=0.2`, `random_state=42`, `stratify=y`) en :
* **1 047 observations** pour l'entraînement (`dataset/train.csv`).
* **262 observations** pour le test final (`dataset/test.csv`).

## 3. Prétraitement des données

Huit colonnes sont retirées avant la modélisation :
* `Survived` : isolée dans le vecteur cible `y`.
* `boat` et `body` : supprimées car elles constituent une fuite de données enregistrée après le naufrage.
* `PassengerId` : simple identifiant numérique séquentiel.
* `Name` et `Ticket` : textes à très haute cardinalité.
* `Cabin` et `home.dest` : trop grand nombre de valeurs manquantes (77,5 % et 43,1 %).

Les 7 variables explicatives conservées sont transformées en **9 variables numériques** grâce à un `ColumnTransformer` ajusté uniquement sur le jeu d'entraînement :
1. **Variables numériques (`Age`, `SibSp`, `Parch`)** : imputation par la médiane (`SimpleImputer(strategy="median")`) puis standardisation (`StandardScaler()`).
2. **Variable `Fare`** : imputation par la médiane, transformation logarithmique `np.log1p` pour corriger la forte asymétrie des prix, puis standardisation (`StandardScaler()`).
3. **Variable `Sex`** : imputation par la valeur la plus fréquente puis encodage binaire (`OrdinalEncoder` : `male` = 0, `female` = 1).
4. **Variables `Pclass` et `Embarked`** : imputation par la valeur la plus fréquente puis encodage disjonctif (`OneHotEncoder(drop="first")`).

## 4. Modélisation et optimisation des hyperparamètres

Le classificateur utilisé est `SVC()` de `scikit-learn`.

Pour régler le modèle sans toucher aux 262 passagers du test final, les 1 047 passagers d'entraînement sont divisés en 837 passagers d'entraînement interne et 210 passagers de validation.

La recherche des meilleurs hyperparamètres est effectuée avec `GridSearchCV` :
* Validation croisée stratifiée à 5 plis (`StratifiedKFold`).
* Métrique d'optimisation : `f1`.
* Grille de **42 configurations** couvrant trois noyaux (`linear`, `rbf`, `poly`), plusieurs valeurs de régularisation `C` (`0.1, 1, 10, 100`), d'échelle `gamma` (`scale, auto, 0.01, 0.1, 1`) et de degré polynomial (`2, 3, 4`).

### Meilleurs hyperparamètres retenus
* **Noyau (`kernel`)** : `rbf`
* **Paramètre de régularisation (`C`)** : `100`
* **Paramètre de portée (`gamma`)** : `0.01`
* **Meilleur F1-score en validation croisée** : `0.7273`

## 5. Résultats de l'évaluation finale

Après réentraînement du meilleur modèle sur la totalité des 1 047 observations d'entraînement, l'évaluation sur les **262 passagers du jeu de test** donne les performances suivantes :

| Métrique | Score sur la Validation (210 passagers) | Score sur le Test Final (262 passagers) |
| :--- | :---: | :---: |
| **Accuracy (Précision globale)** | 80,95 % | **81,30 %** |
| **Précision** | 77,78 % | **78,65 %** |
| **Rappel (Recall)** | 70,00 % | **70,00 %** |
| **F1-score** | 73,68 % | **74,07 %** |
| **ROC-AUC** | Non calculé | **83,94 %** |

### Matrice de confusion sur le jeu de test (262 passagers)
* **Vrais Négatifs (Décédés bien prédits)** : 143
* **Vrais Positifs (Survivants bien prédits)** : 70
* **Faux Positifs (Prédits survivants à tort)** : 19
* **Faux Négatifs (Prédits décédés à tort)** : 30

Le modèle classe correctement **213 passagers sur 262** et s'appuie sur **455 vecteurs de support**.

## 6. Structure des fichiers du projet

* `dataset/titanic.csv` : Jeu de données complet initial (1 309 lignes, 15 colonnes).
* `dataset/titanic.md` : Dictionnaire descriptif des 15 variables.
* `dataset/train.csv` et `dataset/test.csv` : Jeux d'entraînement (1 047 lignes) et de test (262 lignes) exportés après prétraitement.
* `EDA.ipynb` : Notebook d'analyse exploratoire des données.
* `Pretraitement.ipynb` : Notebook détaillé du pipeline de nettoyage et de prétraitement.
* `Tutoriel_SVM_Titanic.ipynb` : Notebook tutoriel complet regroupant le prétraitement, la modélisation SVM, les graphiques d'évaluation et l'interprétation des résultats.
* `svm.py` : Script Python complet et documenté allant du prétraitement jusqu'à l'évaluation finale et l'analyse d'importance des variables.
* `m5_modelisation.py` : Script dédié au module de modélisation et d'évaluation M5.
* `m6_scriptfinal.py` : Script de lancement global du projet.
* `requirements.txt` : Liste des bibliothèques Python nécessaires au projet.
* `rapports/` : Dossier contenant les parties rédigées du rapport final.

## 7. Installation et exécution

Création de l'environnement de travail

```bash
python -m venv .venv | python3 -m venv .venv
```

Installer les bibliothèques Python requises :

```bash
python -m pip install -r requirements.txt | python3 -m pip install -r requirements.txt
```

Pour lancer le script complet depuis la racine du projet :

```bash
python svm.py | python3 svm.py
```
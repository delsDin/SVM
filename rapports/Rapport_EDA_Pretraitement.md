# Rapport d'Analyse Exploratoire des Données (EDA) et de Prétraitement - Projet Titanic (SVM)

Ce document synthétise de manière claire, méthodique et accessible aux débutants en Data Science et en Machine Learning l'ensemble de la démarche réalisée dans les notebooks [EDA.ipynb](/EDA.ipynb) et [Pretraitement.ipynb](/Pretraitement.ipynb).

L'objectif final est de préparer le jeu de données complet du **RMS Titanic** ([dataset/titanic.csv](/dataset/titanic.csv)) pour l'entraînement d'un modèle de classification binaire par **Séparateur à Vaste Marge** (*Support Vector Machine* ou **SVM**).



## Introduction : Les exigences mathématiques d'un modèle SVM

Contrairement aux algorithmes basés sur des arbres de décision, un **SVM est un modèle géométrique**. Il place chaque passager comme un point dans un espace mathématique à plusieurs dimensions et cherche à tracer une frontière (un *hyperplan*) qui sépare les survivants (`Survived = 1`) des personnes décédées (`Survived = 0`) avec la plus grande marge de sécurité possible.

Pour calculer les distances entre les passagers, le SVM impose trois contraintes techniques strictes :
1. **Zéro valeur manquante (`NaN`)** : Une formule de distance géométrique ne peut pas effectuer de calcul sur une case vide.
2. **Uniquement des valeurs numériques** : Les textes et catégories (`"male"`, `"female"`, `"Southampton"`) doivent être traduits en nombres.
3. **Une échelle homogène et sans asymétrie extrême** : Si une variable varie de `0` à `512` (le prix du billet `Fare`) et une autre de `0` à `1` (le sexe `Sex`), la distance géométrique sera écrasée à plus de 99 % par le prix du billet. Toutes les variables continues doivent donc être centrées et réduites.



## Partie 1 : Rapport de l'Analyse Exploratoire des Données (EDA)

**Règle méthodologique respectée** : Durant toute l'EDA ([EDA.ipynb](/EDA.ipynb)), le fichier [dataset/titanic.csv](/dataset/titanic.csv) est analysé dans son intégralité sans subir aucune modification ni aucun découpage.

### 1.1. Structure générale et intégrité des données
- **Dimensions** : Le dataset comporte **1 309 lignes** (passagers) et **15 colonnes** (1 variable cible et 14 variables descriptives).
- **Types de variables** :
  - 5 colonnes d'entiers (`int64`) : `PassengerId`, `Survived`, `Pclass`, `SibSp`, `Parch`.
  - 3 colonnes de nombres décimaux (`float64`) : `Age`, `Fare`, `body` (le format `float64` est automatiquement attribué par `pandas` lorsqu'une colonne numérique contient des valeurs manquantes `NaN`).
  - 7 colonnes textuelles (`object`) : `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`, `boat`, `home.dest`.
- **Contrôle des doublons** :
  - **0 ligne dupliquée** (`df.duplicated().sum() == 0`).
  - **4 passagers partagent un nom identique** (deux *"Connolly, Miss. Kate"* et deux *"Kelly, Mr. James"*). L'examen de leurs âges, numéros de billets (`Ticket`) et ports d'embarquement prouve qu'il s'agit de véritables homonymes historiques et non d'erreurs de saisie. Aucune ligne ne doit être supprimée.

### 1.2. Audit des valeurs manquantes (`NaN`)

Sur les 15 colonnes du dataset, 7 présentent des valeurs manquantes à des degrés très divers :

| Variable | Valeurs présentes | Valeurs manquantes (`NaN`) | Pourcentage manquant (%) | Diagnostic issu de l'EDA |
| :--- | :---: | :---: | :---: | :--- |
| **`body`** | 121 | 1 188 | **90,76 %** | Numéro d'identification du corps repêché (fuite de données). |
| **`Cabin`** | 295 | 1 014 | **77,46 %** | Numéro de cabine, absent pour plus des trois quarts des passagers. |
| **`boat`** | 486 | 823 | **62,87 %** | Numéro du canot de sauvetage (fuite de données). |
| **`home.dest`** | 745 | 564 | **43,09 %** | Domicile / destination en texte libre, très incomplet. |
| **`Age`** | 1 046 | 263 | **20,09 %** | Variable clé dont 1 valeur sur 5 manque : nécessite une **imputation**. |
| **`Embarked`** | 1 307 | 2 | **0,15 %** | Seulement 2 passagères de 1ère classe sans port renseigné. |
| **`Fare`** | 1 308 | 1 | **0,08 %** | Seulement 1 passager de 3ème classe (`PassengerId = 1044`) sans tarif. |

*(Les 8 autres colonnes — `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `SibSp`, `Parch`, `Ticket` — sont complètes à 100 %).*

### 1.3. Détection du *Data Leakage* (Fuite de données du futur)

Le *Data Leakage* désigne la présence, parmi les variables explicatives, d'une information collectée **après** l'événement à prédire et qui révèle directement la cible `Survived` :
- **La variable `boat` (canot de sauvetage)** : Sur les 486 passagers ayant une valeur renseignée dans `boat`, **477 ont survécu (98,1 %)** (contre seulement 2,8 % de survie parmi les 823 passagers sans canot renseigné).
- **La variable `body` (numéro de corps repêché)** : Sur les 121 passagers ayant un numéro dans `body`, **121 sont décédés (100 %)**.

**Conclusion** : Ces deux variables n'existaient pas au moment du départ du Titanic. Les conserver empêcherait le SVM d'apprendre un modèle prédictif réel. Elles doivent être supprimées en priorité.

### 1.4. Analyse de la variable cible (`Survived`)

- **Décédés (`Survived = 0`)** : **809 passagers** (**61,80 %**).
- **Survivants (`Survived = 1`)** : **500 passagers** (**38,20 %**).

**Interprétation** :
1. Le jeu de données présente un léger déséquilibre de classes (~62 % / 38 %), ce qui impose d'utiliser un découpage **stratifié (`stratify=y`)** lors de la séparation Train / Test.
2. **Score de référence (*Baseline*)** : Un modèle trivial qui prédirait systématiquement `0` (décès) obtiendrait déjà **61,80 %** de bonnes réponses (*accuracy*). Un bon modèle SVM devra donc dépasser largement ce seuil (autour de 80 %).

### 1.5. Analyse des variables catégorielles (`Sex`, `Pclass`, `Embarked`)

| Variable | Modalités et effectifs | Taux de survie observé | Interprétation métier et statistique |
| :--- | :--- | :--- | :--- |
| **`Sex`** | `female` (466)<br>`male` (843) | **72,75 %**<br>**19,10 %** | **Variable la plus discriminante du dataset** (corrélation de Spearman de **+0,53** avec `Survived`), conséquence directe de la règle d'évacuation *« les femmes et les enfants d'abord »*. |
| **`Pclass`** | `1` — 1ère classe (323)<br>`2` — 2ème classe (277)<br>`3` — 3ème classe (709) | **61,92 %**<br>**42,96 %**<br>**25,53 %** | La classe socio-économique détermine la proximité des cabines avec le pont des embarcations. Plus de la moitié du navire voyageait en 3ème classe avec seulement 1 chance sur 4 de survivre. |
| **`Embarked`** | `C` — Cherbourg (270)<br>`Q` — Queenstown (123)<br>`S` — Southampton (914) | **55,56 %**<br>**35,77 %**<br>**33,26 %** | Le taux de survie élevé à Cherbourg (`C`) s'explique par le croisement avec `Pclass` : **52,2 %** des passagers montés à Cherbourg voyageaient en **1ère classe**, contre seulement 1,6 % à Queenstown (`Q`, où 91,9 % étaient en 3ème classe). |

### 1.6. Analyse des variables numériques (`Age`, `Fare`, `SibSp`, `Parch`), asymétrie et valeurs extrêmes

| Variable | Moyenne | Écart-type | Minimum | Médiane (50 %) | Maximum | Écart Interquartile (IQR) | Coefficient d'asymétrie (*Skewness*) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Age`** | 29,88 | 14,41 | 0,17 | **28,00** | 80,00 | 18,00 | **0,41** (quasi symétrique) |
| **`Fare`** | 33,30 | 51,76 | 0,00 | **14,45** | 512,33 | 23,38 | **4,37** (très asymétrique) |
| **`SibSp`** | 0,50 | 1,04 | 0,00 | **0,00** | 8,00 | 1,00 | **3,84** (concentré en 0 et 1) |
| **`Parch`** | 0,39 | 0,87 | 0,00 | **0,00** | 9,00 | 0,00 | **3,67** (concentré en 0) |

**Trois constats majeurs pour le SVM** :
1. **Robustesse de la médiane face aux valeurs extrêmes (*outliers*)** : Sur `Fare`, la moitié des passagers a payé moins de **14,45 £** (la médiane), mais quelques billets de luxe atteignant **512,33 £** tirent la moyenne jusqu'à **33,30 £**. C'est pourquoi l'imputation des valeurs manquantes doit se faire par la **médiane** et non par la moyenne.
2. **Correction de l'asymétrie de `Fare` par logarithme (`np.log1p`)** : Avec une asymétrie (*Skewness*) de **4,37**, la distribution brute de `Fare` est écrasée vers la gauche avec une très longue traîne vers la droite. L'application de la fonction $\ln(1 + \text{Fare})$ (qui gère les billets gratuits à `0,00 £` car $\ln(1+0)=0$) ramène l'asymétrie à **0,54** et sépare nettement la distribution des billets bon marché (forte mortalité) de celle des billets onéreux (forte survie).
3. **Dynamique familiale (`Age`, `SibSp`, `Parch`)** :
   - Les jeunes enfants (`Age < 10 ans`) présentent un pic net de survie.
   - Les passagers accompagnés de 1 ou 2 proches (`SibSp` ou `Parch` égal à 1 ou 2) survivent mieux (~50–55 %) que les passagers seuls (~30–35 %), tandis que les familles nombreuses (`SibSp >= 3` ou `Parch >= 4`, majoritairement en 3ème classe) subissent une mortalité très élevée (> 85 %).

### 1.7. Variables à haute cardinalité et interactions multivariées
- **Identifiant et textes à haute cardinalité** :
  - `PassengerId` (1 309 valeurs uniques) est un simple index numérique sans signification physique.
  - `Name` (1 307 valeurs uniques), `Ticket` (929 valeurs uniques), `home.dest` (369 valeurs uniques, 43,1 % manquant) et `Cabin` (186 valeurs uniques, 77,5 % manquant) possèdent beaucoup trop de valeurs distinctes. Les encoder directement avec un `OneHotEncoder` créerait plus de 2 000 colonnes creuses, provoquant un surapprentissage (*Overfitting*) et le fléau de la dimension (*Curse of Dimensionality*) dans le SVM.
- **Interaction non-linéaire `Pclass` × `Sex`** :
  - Femmes de 1ère classe : **96,5 % de survie** (139 sur 144).
  - Femmes de 2ème classe : **88,7 % de survie**.
  - Femmes de 3ème classe : **49,1 % de survie**.
  - Hommes de 1ère classe : **34,1 % de survie**, contre **14,6 %** en 2ème classe et **15,2 %** en 3ème classe.

---

## Partie 2 : Rapport du Prétraitement des Données

Le prétraitement ([Pretraitement.ipynb](/Pretraitement.ipynb)) traduit chaque diagnostic de l'EDA en une chaîne de transformation automatisée et reproductible à l'aide de `scikit-learn`.

### 2.1. Étape 1 : Sélection des variables ($X$) et isolation de la cible ($y$)

On sépare d'abord la cible `y = df["Survived"]` et on construit la matrice `X` en supprimant les 8 colonnes non retenues :
- **Fuites de données (*Data Leakage*)** : `boat`, `body`.
- **Identifiant et colonnes textuelles bruitées ou à haute cardinalité** : `PassengerId`, `Name`, `Ticket`, `Cabin`, `home.dest`.
- **Cible** : `Survived`.

Il reste **7 variables explicatives pertinentes** dans `X` : `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.

### 2.2. Étape 2 : Découpage stratifié en 2 ensembles (`Train` / `Test`)

**Pourquoi découper avant le nettoyage statistique ?**
Pour empêcher toute fuite d'information du jeu de test vers le jeu d'entraînement, les paramètres statistiques (médianes, moyennes, écarts-types, modalités les plus fréquentes) doivent être calculés **exclusivement sur le jeu d'entraînement (`X_train`)**.

Nous utilisons `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` :
- **`X_train`, `y_train` (80 %)** : **1 047 passagers** (taux de survie : **38,20 %**).
- **`X_test`, `y_test` (20 %)** : **262 passagers** (taux de survie : **38,17 %**).
- Grâce à `stratify=y`, l'équilibre entre survivants et victimes est strictement identique dans les deux ensembles.

### 2.3. Étape 3 : Architecture des 4 pipelines spécialisés (`ColumnTransformer`)

Chaque famille de variables traverse un sous-pipeline dédié au sein d'un `ColumnTransformer` :

1. **Branche `numerique` (`Age`, `SibSp`, `Parch`)** :
   - `SimpleImputer(strategy='median')` : Remplace les valeurs manquantes par la médiane apprise sur `X_train`.
   - `StandardScaler()` : Centre et réduit chaque variable ($z = \frac{x - \mu}{\sigma}$) pour obtenir une moyenne de `0` et un écart-type de `1` sur `X_train`.
2. **Branche `fare` (`Fare`)** :
   - `SimpleImputer(strategy='median')` : Remplace la valeur manquante par la médiane de `X_train`.
   - `FunctionTransformer(np.log1p)` : Applique $\ln(1 + x)$ pour corriger la forte asymétrie à droite (*Skewness* de 4,37) et compresser les valeurs extrêmes jusqu'à 512,33 £.
   - `StandardScaler()` : Centre et réduit le résultat logarithmique (moyenne = `0`, écart-type = `1`).
3. **Branche `sexe` (`Sex`)** :
   - `SimpleImputer(strategy='most_frequent')` : Sécurité en cas de valeur manquante.
   - `OrdinalEncoder(categories=[['male', 'female']])` : Encode la variable binaire sur une seule colonne (`'male'` $\to 0.0$, `'female'` $\to 1.0$).
4. **Branche `categoriel` (`Pclass`, `Embarked`)** :
   - `SimpleImputer(strategy='most_frequent')` : Remplace les valeurs manquantes d'`Embarked` par le port majoritaire (`'S'`, Southampton).
   - `OneHotEncoder(drop='first', handle_unknown='ignore')` : Crée des variables indicatrices binaires (`0.0` ou `1.0`) en supprimant la première catégorie de chaque variable (`Pclass = 1` et `Embarked = 'C'` servent de catégories de référence lorsque les deux indicatrices valent `0.0`), ce qui évite toute redondance mathématique (*multicolinéarité*).

### 2.4. Étape 4 : Application sans fuite de données (`fit_transform` vs `transform`)

- Sur l'entraînement : `X_train_prepare = pipeline_final.fit_transform(X_train)` apprend les paramètres statistiques sur les 1 047 passagers et transforme `X_train`.
- Sur le test : `X_test_prepare = pipeline_final.transform(X_test)` applique exactement les formules et paramètres appris sur `X_train` aux 262 passagers de `X_test`.

### 2.5. Contrôle qualité des 9 variables finales et fichiers exportés

À la sortie du pipeline, les 7 colonnes d'entrée sont converties en **9 variables explicatives purement numériques** :

| N° | Nom de la colonne générée | Nature | Moyenne (`Train`) | Écart-type (`Train`) | Valeurs manquantes (`Train` / `Test`) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `numerique__Age` | Continue standardisée | 0,000 | 1,000 | 0 / 0 |
| 2 | `numerique__SibSp` | Discrète standardisée | 0,000 | 1,000 | 0 / 0 |
| 3 | `numerique__Parch` | Discrète standardisée | 0,000 | 1,000 | 0 / 0 |
| 4 | `fare__Fare` | Continue (`log1p` + standardisée) | 0,000 | 1,000 | 0 / 0 |
| 5 | `sexe__Sex` | Binaire (`0` = homme, `1` = femme) | 0,352 | 0,478 | 0 / 0 |
| 6 | `categoriel__Pclass_2` | Indicatrice One-Hot (2ème classe) | 0,213 | 0,410 | 0 / 0 |
| 7 | `categoriel__Pclass_3` | Indicatrice One-Hot (3ème classe) | 0,539 | 0,499 | 0 / 0 |
| 8 | `categoriel__Embarked_Q` | Indicatrice One-Hot (Queenstown) | 0,089 | 0,285 | 0 / 0 |
| 9 | `categoriel__Embarked_S` | Indicatrice One-Hot (Southampton) | 0,702 | 0,458 | 0 / 0 |

### 2.6. Fichiers générés pour la modélisation

Les ensembles préparés (incluant les 9 variables explicatives et la colonne cible `Survived` en 10ème colonne) sont sauvegardés dans le dossier `dataset/` :
- **[dataset/train.csv](/dataset/train.csv)** : **1 047 lignes × 10 colonnes**, prêt pour l'entraînement et la validation croisée du SVM.
- **[dataset/test.csv](/dataset/test.csv)** : **262 lignes × 10 colonnes**, prêt pour l'évaluation finale des performances du SVM.


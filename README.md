
# Projet SVM — Titanic

## 1. Présentation

Ce projet porte sur l'utilisation de l'algorithme **Support Vector Machine (SVM)** pour résoudre un problème de classification binaire à partir du dataset **Titanic**.

L'objectif est de mettre en œuvre les différentes étapes d'un projet d'apprentissage automatique :

- préparation des données ;
- traitement des valeurs manquantes ;
- encodage des variables catégorielles ;
- normalisation des variables numériques ;
- entraînement d'un modèle SVM ;
- optimisation des hyperparamètres ;
- évaluation des performances du modèle.

---

## 2. Dataset

Le dataset utilisé est **Titanic**.

Les données sont séparées en :

- **891 observations** pour l'entraînement ;
- **418 observations** pour le test.

La variable cible est :

```text
Survived
```

Elle indique si le passager a survécu ou non.

---

## 3. Prétraitement des données

Le prétraitement est réalisé dans :

```text
dataset/svm.py
```

Certaines colonnes sont supprimées, notamment :

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`
- `boat`
- `body`
- `home.dest`

### Variables numériques

Les variables :

- `Age`
- `SibSp`
- `Parch`

sont traitées par :

1. imputation de la médiane ;
2. standardisation avec `StandardScaler`.

### Variable `Fare`

`Fare` est traitée par :

1. imputation de la médiane ;
2. transformation logarithmique `log1p` ;
3. standardisation.

### Variable `Sex`

La variable `Sex` est imputée avec la valeur la plus fréquente puis encodée avec `OrdinalEncoder`.

### Variables `Pclass` et `Embarked`

Ces variables sont imputées avec la valeur la plus fréquente puis transformées avec `OneHotEncoder`.

---

## 4. Modélisation SVM

La modélisation est réalisée dans :

```text
m5_modelisation.py
```

Le modèle utilisé est :

```python
SVC()
```

Trois noyaux sont étudiés :

- `linear`
- `rbf`
- `poly`

---

## 5. Optimisation des hyperparamètres

La recherche des meilleurs paramètres est effectuée avec :

```text
GridSearchCV
```

Une validation croisée stratifiée à 5 plis est utilisée avec :

```text
StratifiedKFold
```

L'optimisation est réalisée selon le :

```text
F1-score
```

Les principaux hyperparamètres étudiés sont :

- `C`
- `gamma`
- `degree`
- `kernel`

---

## 6. Évaluation

Le modèle est évalué avec plusieurs métriques :

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Une **matrice de confusion** ainsi qu'un **rapport de classification** sont également produits.

---

## 7. Organisation du projet

```text
SVM/
│
├── dataset/
│   ├── svm.py
│   ├── titanic.csv
│   └── titanic.md
│
├── m5_modelisation.py
├── main.py
├── M5_rapport.md
├── README.md
└── .gitignore
```

### Rôle des fichiers

**`dataset/svm.py`**

Contient le chargement du dataset et le prétraitement des données.

**`m5_modelisation.py`**

Contient la modélisation SVM, la recherche des hyperparamètres et l'évaluation.

**`main.py`**

Permet de lancer le projet final.

**`M5_rapport.md`**

Contient le rapport relatif au travail de M5.

**`README.md`**

Présente le projet et explique son fonctionnement.

---

## 8. Installation

Installer les bibliothèques nécessaires :

```bash
python -m pip install pandas numpy scikit-learn
```

---

## 9. Exécution

Depuis le dossier principal du projet :

```bash
python main.py
```

Le programme lance le traitement du projet et affiche les résultats de la modélisation SVM.

---

## 10. Résultats

Les résultats obtenus sont affichés directement dans le terminal lors de l'exécution du programme.

Les principales informations affichées sont :

- meilleurs hyperparamètres ;
- meilleur F1-score obtenu lors de la validation croisée ;
- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- ROC-AUC ;
- rapport de classification ;
- matrice de confusion.

---

## 11. Conclusion

Ce projet permet de mettre en pratique l'utilisation d'un modèle SVM pour une tâche de classification sur les données du Titanic.

Le prétraitement des données, la standardisation, l'encodage des variables catégorielles ainsi que l'optimisation des hyperparamètres permettent de construire et d'évaluer un modèle de classification de manière structurée.
```

###  Pour les résultats

Je n'ai volontairement **pas inventé les valeurs d'Accuracy, F1, ROC-AUC, etc.** Elles doivent être celles affichées par votre exécution réelle de M5.

Une fois ce `README.md` enregistré, **on passe à la présentation PowerPoint** : je te donnerai les diapositives une par une, avec **le texte à mettre + ce que tu dois expliquer oralement**.
# Rapport M5 : Modélisation et évaluation du SVM

## 1. Introduction

Ce cinquième module est consacré à la modélisation et à l’évaluation d’un modèle de classification utilisant les machines à vecteurs de support (SVM).

L’objectif est de construire un modèle capable de prédire la survie des passagers du Titanic à partir des caractéristiques disponibles dans le jeu de données.

Après le travail de préparation des données effectué dans le module M4, plusieurs configurations du SVM sont étudiées afin de sélectionner les meilleurs hyperparamètres. Le modèle retenu est ensuite évalué sur un jeu de test indépendant.

## 2. Présentation des données

Le jeu de données Titanic initial contient 1 309 observations.

À l'issue du prétraitement réalisé dans le module M4, les données ont été réparties en deux ensembles :

- **Données d'entraînement :** 1 047 observations.
- **Données de test :** 262 observations.

La variable cible est `Survived` :
- `0` : le passager n'a pas survécu.
- `1` : le passager a survécu.

Les fichiers `train.csv` et `test.csv` contiennent chacun neuf variables explicatives préparées, ainsi que la variable cible `Survived`.

Les opérations de prétraitement réalisées dans M4 comprennent notamment :
- Le traitement des valeurs manquantes.
- L'encodage des variables catégorielles.
- La normalisation des variables numériques.
- La transformation de la variable `Fare`.

Les données préparées sont utilisées directement dans le module M5.

## 3. Méthodologie de modélisation

### 3.1. Chargement des données

Les fichiers `dataset/train.csv` et `dataset/test.csv` sont chargés avec la bibliothèque Pandas.

La variable cible `Survived` est séparée des variables explicatives afin de constituer les ensembles `X_train`, `y_train`, `X_test` et `y_test`.

Une vérification des dimensions et des valeurs manquantes est également effectuée.

### 3.2. Séparation des données d'entraînement

Les 1 047 observations du jeu d'entraînement sont divisées en deux sous-ensembles :

- **Entraînement interne :** 837 observations.
- **Validation :** 210 observations.

La séparation est réalisée avec `train_test_split`, en utilisant :
- `test_size=0.2` ;
- `random_state=42` ;
- `stratify=y_train`.

La stratification permet de conserver une répartition comparable des classes dans les deux sous-ensembles.

### 3.3. Construction du modèle

Le modèle est construit à l'aide de la classe `SVC` de Scikit-learn.

Une pipeline est utilisée pour regrouper le classificateur SVM. Les données ayant déjà été prétraitées dans M4, le pipeline de M5 ne contient plus d'étape de prétraitement.

## 4. Recherche et sélection du noyau

Le SVM propose plusieurs fonctions noyau permettant de traiter différents types de séparations entre les classes.

Trois noyaux sont étudiés dans la recherche des hyperparamètres :

- **Linéaire (`linear`) :** adapté aux séparations linéaires.
- **Radial (`rbf`) :** permet de modéliser des séparations non linéaires.
- **Polynomial (`poly`) :** permet de représenter des relations polynomiales entre les variables.

Ces trois noyaux sont intégrés dans la grille de recherche afin de comparer différentes configurations.

La recherche permet finalement de retenir le noyau radial (`rbf`) pour la meilleure configuration obtenue.

## 5. Optimisation des hyperparamètres

### 5.1. Recherche par GridSearchCV

La classe `GridSearchCV` est utilisée pour rechercher les meilleurs hyperparamètres du modèle.

La grille comporte 42 configurations différentes, réparties entre les trois noyaux étudiés.

La validation croisée stratifiée (`StratifiedKFold`) est réalisée avec cinq plis.

Le score F1 est utilisé comme métrique de sélection afin de tenir compte à la fois de la précision et du rappel.

Au total, 210 ajustements sont effectués (42 configurations × 5 plis).

### 5.2. Meilleurs hyperparamètres

La recherche retourne les paramètres suivants :

| Hyperparamètre | Valeur retenue |
|---|---|
| Noyau (`kernel`) | `rbf` |
| Paramètre C | `100` |
| Paramètre gamma | `0.01` |

Le meilleur score F1 obtenu en validation croisée est de **0,7273**.

### 5.3. Évaluation sur l'ensemble de validation

Le modèle optimisé est évalué sur les 210 observations de validation.

Les résultats obtenus sont :

| Métrique | Résultat |
|---|---:|
| Accuracy | 80,95 % |
| Précision | 77,78 % |
| Rappel | 70,00 % |
| F1-score | 73,68 % |

La matrice de confusion obtenue est :

| | Prédit : 0 | Prédit : 1 |
|---|---:|---:|
| Réel : 0 | 114 | 16 |
| Réel : 1 | 24 | 56 |

Ces résultats montrent que le modèle obtient une accuracy de 80,95 % sur l'ensemble de validation.

## 6. Évaluation finale sur le jeu de test

Après la sélection des meilleurs hyperparamètres, le modèle est réentraîné sur l'intégralité des 1 047 observations d'entraînement.

Il est ensuite évalué sur les 262 observations du jeu de test, qui n'ont pas été utilisées pour sélectionner les hyperparamètres.

### 6.1. Résultats obtenus

| Métrique | Résultat |
|---|---:|
| Accuracy | 81,30 % |
| Précision | 78,65 % |
| Rappel | 70,00 % |
| F1-score | 74,07 % |
| ROC-AUC | 83,94 % |

### 6.2. Matrice de confusion

La matrice de confusion obtenue sur le jeu de test est :

| | Prédit : 0 | Prédit : 1 |
|---|---:|---:|
| Réel : 0 | 143 | 19 |
| Réel : 1 | 30 | 70 |

Sur les 262 observations :
- 143 passagers n'ayant pas survécu sont correctement classés.
- 70 passagers ayant survécu sont correctement classés.
- 19 passagers n'ayant pas survécu sont prédits à tort comme survivants.
- 30 passagers ayant survécu sont prédits à tort comme non-survivants.

Le modèle réalise donc **213 prédictions correctes sur 262**, soit une accuracy de 81,30 %.

Le score ROC-AUC de 83,94 % indique également une capacité intéressante du modèle à distinguer les deux classes.

## 7. Analyse des résultats et limites

Les résultats obtenus montrent que le modèle SVM optimisé atteint une accuracy de 81,30 % sur le jeu de test, avec un F1-score de 74,07 %.

Le score F1 obtenu sur le jeu de test est légèrement supérieur à celui obtenu sur l'ensemble de validation (74,07 % contre 73,68 %).

Le rappel de 70 % indique que le modèle identifie correctement 70 des 100 passagers ayant survécu dans le jeu de test. Il reste donc des survivants qui ne sont pas correctement détectés.

Plusieurs limites doivent être prises en compte :

- Le modèle commet encore des erreurs de classification sur les deux classes.
- La recherche des hyperparamètres est limitée aux 42 configurations définies dans la grille.
- Les performances dépendent des variables disponibles et des choix de prétraitement effectués.

**Point méthodologique :** le prétraitement de M4 a été ajusté sur les 1 047 observations d'entraînement avant la séparation interne de M5 et la validation croisée. Les statistiques de prétraitement ont donc été calculées sans tenir compte des plis internes de M5. Cela peut introduire une légère fuite d'information dans la validation interne et la validation croisée. Pour une évaluation méthodologiquement plus stricte, le prétraitement devrait être intégré au pipeline évalué par validation croisée, afin d'être ajusté séparément dans chaque pli.

Le jeu de test final de 262 observations reste néanmoins distinct de la recherche des hyperparamètres de M5.

## 8. Conclusion

Ce module a permis de mettre en œuvre un modèle SVM pour prédire la survie des passagers du Titanic.

La recherche par GridSearchCV a permis d'identifier la configuration suivante : noyau radial (`rbf`), `C=100` et `gamma=0.01`.

Après réentraînement sur les 1 047 observations d'entraînement, le modèle a obtenu une accuracy de 81,30 %, un F1-score de 74,07 % et un ROC-AUC de 83,94 % sur le jeu de test.

Ces résultats montrent que le SVM constitue une approche intéressante pour cette tâche de classification. Toutefois, les erreurs de prédiction et les limites méthodologiques identifiées indiquent que des améliorations restent possibles.
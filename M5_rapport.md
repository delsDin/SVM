# Rapport M5 : Modélisation et évaluation du SVM

## 1. Introduction

Dans le cadre du projet de classification portant sur le dataset Titanic, cette étape est consacrée à la mise en œuvre et à l'évaluation de modèles de classification basés sur les machines à vecteurs de support (SVM).

L'objectif est de construire un modèle capable de prédire la survie des passagers à partir des caractéristiques disponibles dans le jeu de données.

Le travail de M5 intervient après l'étape de préparation des données réalisée par M4. Il consiste à entraîner plusieurs modèles SVM, à comparer leurs performances, à optimiser leurs hyperparamètres et à évaluer le modèle retenu sur les données de test.

Cette démarche permet d'identifier une configuration adaptée au problème étudié et d'analyser les performances ainsi que les limites du modèle obtenu.

## 2. Présentation des données

Le projet repose sur le dataset Titanic, qui regroupe des informations relatives à 1 309 passagers.

Les données sont réparties en deux ensembles :

- **Ensemble d'entraînement :** 891 observations, utilisées pour entraîner les modèles et effectuer la sélection des hyperparamètres.
- **Ensemble de test :** 418 observations, réservées à l'évaluation finale du modèle retenu.

La variable cible est `Survived`. Elle indique si un passager a survécu au naufrage (1) ou non (0).

La préparation des données, réalisée dans le module M4, comprend notamment le traitement des valeurs manquantes, l'encodage des variables catégorielles et la mise à l'échelle des variables numériques.

Ces opérations permettent de disposer de données adaptées à l'entraînement des modèles SVM.

Il est important de préciser que les 418 observations de test correspondent à la seconde partie du dataset fourni. Elles ne proviennent donc pas d'une séparation aléatoire effectuée sur les 1 309 observations.

## 3. Méthodologie de modélisation

### 3.1. Construction du pipeline

Pour la modélisation, nous avons utilisé la classe `Pipeline` de Scikit-learn afin de regrouper les étapes de prétraitement et de classification.

Le pipeline comprend deux éléments :

- Le prétraitement des données, réalisé à l'aide du transformateur `pretraitement` fourni par le module M4.
- Le classificateur SVM, implémenté avec la classe `SVC` de Scikit-learn.

Cette organisation permet d'appliquer automatiquement les transformations nécessaires avant chaque apprentissage et chaque prédiction.

### 3.2. Séparation des données d'entraînement

Les 891 observations de l'ensemble d'entraînement ont été divisées en deux sous-ensembles :

- 80 % pour l'apprentissage du modèle ;
- 20 % pour la validation de ses performances.

La fonction `train_test_split` a été utilisée avec `test_size=0.2`, `random_state=42` et `stratify=y_train`.

La stratification permet de conserver une répartition similaire des deux classes dans les ensembles d'apprentissage et de validation. La fixation de la graine aléatoire garantit la reproductibilité de cette séparation.

Cette méthode permet d'évaluer les modèles sur des observations qui n'ont pas servi à leur apprentissage.

## 4. Comparaison des noyaux SVM

### 4.1. Noyaux étudiés

Trois noyaux ont été testés dans le cadre de la modélisation :

- **Linéaire (`linear`) :** utilisé pour rechercher une frontière de décision linéaire entre les classes.
- **RBF (`rbf`) :** permet de modéliser des relations non linéaires entre les variables.
- **Polynomial (`poly`) :** permet de représenter des frontières de décision plus complexes à l'aide de fonctions polynomiales.

### 4.2. Comparaison des performances

Les trois modèles ont été évalués sur le même ensemble de validation à l'aide de quatre métriques : l'accuracy, la précision, le rappel et le F1-score.

| Métrique | Linéaire | RBF | Polynomial |
|---|---:|---:|---:|
| Accuracy | 77,65 % | 82,12 % | 80,45 % |
| Precision | 73,77 % | 83,64 % | 84,00 % |
| Recall | 65,22 % | 66,67 % | 60,87 % |
| F1-score | 69,23 % | 74,19 % | 70,59 % |

### 4.3. Analyse des résultats

Les résultats montrent que le noyau RBF obtient les meilleures performances en accuracy (82,12 %) et en F1-score (74,19 %).

Le noyau polynomial présente la meilleure précision (84,00 %), mais son rappel est plus faible (60,87 %).

Le noyau linéaire obtient les performances les moins élevées sur la majorité des métriques étudiées.

Au regard de ces résultats, le noyau RBF a été retenu comme candidat privilégié pour la suite du travail, notamment pour l'optimisation des hyperparamètres.

## 5. Optimisation des hyperparamètres

### 5.1. Principe de l'optimisation

Afin d'améliorer les performances du modèle SVM, une recherche des meilleurs hyperparamètres a été réalisée.

Les principaux paramètres étudiés sont :

- `C` : contrôle le compromis entre la maximisation de la marge et la pénalisation des erreurs de classification.
- `gamma` : détermine l'influence des observations d'entraînement dans la construction de la frontière de décision pour les noyaux concernés.
- `kernel` : définit le type de noyau utilisé par le modèle.

### 5.2. Recherche avec GridSearchCV

La classe `GridSearchCV` de Scikit-learn a été utilisée pour tester différentes combinaisons d'hyperparamètres.

La recherche a porté sur 42 configurations. Une validation croisée à 5 plis a été appliquée, avec le F1-score comme métrique de sélection.

Au total, 210 entraînements ont été réalisés (42 configurations × 5 plis).

### 5.3. Meilleure configuration obtenue

La recherche a identifié la configuration suivante :

- `kernel = 'rbf'`
- `C = 1`
- `gamma = 'auto'`

Le meilleur F1-score moyen obtenu lors de la validation croisée est de 0,7581.

Sur l'ensemble de validation de 179 observations, le modèle optimisé a obtenu les résultats suivants :

| Métrique | Résultat |
|---|---:|
| Accuracy | 82,68 % |
| Precision | 82,76 % |
| Recall | 69,57 % |
| F1-score | 75,59 % |

Ces résultats montrent que la recherche des hyperparamètres a permis d'identifier une configuration offrant un meilleur F1-score sur l'ensemble de validation que les modèles initialement comparés.

## 6. Évaluation finale du modèle

### 6.1. Entraînement et prédiction

Après l'optimisation des hyperparamètres, le meilleur estimateur a été réentraîné sur l'ensemble des 891 observations d'entraînement.

Le modèle obtenu a ensuite été utilisé pour effectuer des prédictions sur les 418 observations réservées au test final.

### 6.2. Résultats de l'évaluation

Les performances obtenues sont les suivantes :

| Métrique | Résultat |
|---|---:|
| Accuracy | 78,23 % |
| Precision | 71,90 % |
| Recall | 69,62 % |
| F1-score | 70,74 % |
| ROC-AUC | 79,52 % |

Le modèle a correctement classé 327 observations sur les 418 données de test.

### 6.3. Analyse de la matrice de confusion

La matrice de confusion obtenue est :

| | Prédit : 0 | Prédit : 1 |
|---|---:|---:|
| Réel : 0 | 217 | 43 |
| Réel : 1 | 48 | 110 |

On observe que :

- 217 non-survivants ont été correctement classés.
- 110 survivants ont été correctement identifiés.
- 43 non-survivants ont été prédits comme survivants.
- 48 survivants ont été prédits comme non-survivants.

### 6.4. Interprétation des performances

L'accuracy de 78,23 % indique que le modèle a correctement classé environ 78 % des observations de test.

La précision de 71,90 % signifie que parmi les passagers prédits comme survivants, environ 72 % ont effectivement survécu.

Le rappel de 69,62 % montre que le modèle a identifié environ 70 % des survivants réels.

Enfin, le F1-score de 70,74 % traduit un équilibre entre la précision et le rappel. Le ROC-AUC de 79,52 % indique également une capacité de discrimination intéressante entre les deux classes.

## 7. Limites du modèle

Malgré les performances obtenues, le modèle présente certaines limites.

Tout d'abord, les résultats de l'évaluation finale sont inférieurs à ceux obtenus sur l'ensemble de validation. L'accuracy passe de 82,68 % à 78,23 %, tandis que le F1-score passe de 75,59 % à 70,74 %. Cette différence montre que les performances observées pendant la validation ne se retrouvent pas entièrement sur les données de test.

Ensuite, le modèle commet encore des erreurs de classification, notamment 48 survivants prédits comme non-survivants et 43 non-survivants prédits comme survivants.

Par ailleurs, les performances dépendent des variables disponibles dans le dataset et des choix effectués lors du prétraitement. D'autres transformations ou combinaisons de variables pourraient éventuellement améliorer les résultats.

Enfin, la recherche des hyperparamètres s'est limitée aux configurations définies dans la grille de recherche. L'exploration d'autres paramètres ou la comparaison avec d'autres algorithmes pourrait permettre d'obtenir de meilleures performances.

## 8. Conclusion

Ce travail a permis de mettre en œuvre un modèle de classification basé sur les machines à vecteurs de support (SVM) afin de prédire la survie des passagers du Titanic.

La comparaison des différents noyaux a montré que le noyau RBF offrait les meilleurs résultats lors de la première évaluation. L'optimisation avec GridSearchCV a ensuite permis de retenir la configuration suivante : `kernel='rbf'`, `C=1` et `gamma='auto'`.

Sur les 418 observations réservées à l'évaluation finale, le modèle a obtenu une accuracy de **78,23 %** et un F1-score de **70,74 %**. Ces résultats montrent que le modèle parvient à effectuer des prédictions pertinentes, même si certaines erreurs restent présentes.

Cette expérience nous a également permis de comprendre l'importance du choix du noyau, du réglage des hyperparamètres et de l'utilisation de plusieurs métriques pour évaluer un modèle de classification.

Des améliorations pourraient être envisagées, notamment en testant d'autres combinaisons de paramètres, en travaillant davantage sur les variables utilisées et en comparant le SVM à d'autres algorithmes de classification.

En conclusion, cette étape a permis de construire, d'optimiser et d'évaluer un modèle SVM sur le jeu de données Titanic, tout en mettant en pratique les principales étapes d'un processus de modélisation en apprentissage automatique.



# Titanic Dataset 

Ce dépôt contient le jeu de données complet **Titanic** (`titanic.csv`), destiné à un problème de **classification binaire supervisée** : prédire si un passager a survécu (`Survived = 1`) ou non (`Survived = 0`) au naufrage du RMS Titanic le 15 avril 1912.



## 1. Structure du fichier de données

| Fichier | Lignes (hors en-tête) | Colonnes | Taille | Description |
| :--- | :---: | :---: | :---: | :--- |
| `titanic.csv` | **1 309** (`PassengerId` 1 à 1309) | **15** | 110 Ko | Jeu de données complet fusionnant les 12 colonnes au format Kaggle (`train` 1–891 + `test` 892–1309 avec la vraie cible `Survived` complète à 100 %) et les 3 colonnes historiques (`boat`, `body`, `home.dest`). |



## 2. Dictionnaire des variables (Features)

| Variable | Rôle | Type | Valeurs / Plage (Manquants) | Description & Intérêt pour la classification |
| :--- | :---: | :---: | :--- | :--- |
| **`PassengerId`** | Identifiant | Entier | `1` à `1309` (0 %) | Identifiant unique séquentiel (`1–891` : Train Kaggle, `892–1309` : Test Kaggle). **À exclure** des variables prédictives. |
| **`Survived`** | **Cible ($y$)** | Binaire | `0` = Non (809), `1` = Oui (500) (0 %) | Variable cible à prédire (`0` : Décédé — 61,8 %, `1` : Survivant — 38,2 %). Aucune valeur manquante. |
| **`Pclass`** | Explicative | Catégorielle ordinale | `1` = 1ère, `2` = 2ème, `3` = 3ème (0 %) | Classe du billet (proxy du statut socio-économique et de la proximité avec les ponts supérieurs). **Fort pouvoir discriminant**. |
| **`Name`** | Explicative | Texte | Chaîne de caractères (0 %) | Nom complet incluant la civilité (`Mr.`, `Mrs.`, `Miss.`, `Master.`, `Dr.`, etc.). Très utile pour extraire le **Titre (`Title`)** et regrouper les familles. |
| **`Sex`** | Explicative | Catégorielle binaire | `male` (843), `female` (466) (0 %) | Sexe du passager. **Variable la plus prédictive** en raison du protocole d'évacuation *« les femmes et les enfants d'abord »*. |
| **`Age`** | Explicative | Numérique continue | `0.17` à `80.00` ans (**20,1 %** / 263 manq.) | Âge en années (fractionnaire si `< 1` an, estimé si sous la forme `xx.5`). À imputer (ex: médiane par `Title` et `Pclass`). |
| **`SibSp`** | Explicative | Numérique discrète | `0` à `8` (0 %) | Nombre de frères, sœurs, époux ou épouses à bord. |
| **`Parch`** | Explicative | Numérique discrète | `0` à `9` (0 %) | Nombre de parents ou d'enfants à bord (`0` pour les enfants voyageant uniquement avec une nourrice). |
| **`Ticket`** | Explicative | Texte / Alphanum. | Ex: `A/5 21171`, `113803` (0 %) | Numéro de billet. Plusieurs passagers partageant le même billet voyagent en groupe ou en famille (permet de calculer le tarif réel par personne). |
| **`Fare`** | Explicative | Numérique continue | `0.00` à `512.33` (£) (**0,1 %** / 1 manq.) | Prix payé pour le billet (correspond souvent au **tarif total du billet/groupe**). Distribution très asymétrique à droite. |
| **`Cabin`** | Explicative | Texte / Alphanum. | Ex: `C85`, `B57 B59 B63` (**77,5 %** / 1014 manq.) | Numéro(s) de cabine. La première lettre (`A` à `G`, `T`) indique le **pont (`Deck`)**. L'indicateur de présence/absence de cabine est informatif. |
| **`Embarked`** | Explicative | Catégorielle nominale | `C`, `Q`, `S` (**0,2 %** / 2 manq.) | Port d'embarquement : `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton. Corrélé à la classe sociale (`Pclass`). |
| **`boat`** | **Fuite (Leakage)** | Texte / Alphanum. | Ex: `2`, `11`, `D` (**62,9 %** / 823 manq.) | Identifiant du canot de sauvetage dans lequel le passager a embarqué. **À supprimer impérativement lors du prétraitement** (fuite directe de la cible `Survived = 1`). |
| **`body`** | **Fuite (Leakage)** | Numérique discrète | `1` à `328` (**90,8 %** / 1188 manq.) | Numéro d'identification du corps repêché. **À supprimer impérativement lors du prétraitement** (fuite directe de la cible `Survived = 0`). |
| **`home.dest`** | Explicative | Texte | Ex: `New York, NY`, `Paris, France` (**43,1 %** / 564 manq.) | Domicile et/ou destination finale du passager. Nécessite un nettoyage textuel ou une extraction de pays/état si utilisée. |

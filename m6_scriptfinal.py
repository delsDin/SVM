
import os

# Chemin du dossier principal du projet
DOSSIER_PROJET = os.path.dirname(os.path.abspath(__file__))

# Aller temporairement dans le dossier dataset
# pour permettre à svm.py de trouver titanic.csv
os.chdir(os.path.join(DOSSIER_PROJET, "dataset"))

print("=" * 55)
print("PROJET SVM - TITANIC")
print("EXECUTION FINALE M6")
print("=" * 55)

print("\nLancement du traitement M4 + M5...\n")

# Import du fichier M5
# Le fichier M5 reste totalement inchangé.
import sys
sys.path.insert(0, DOSSIER_PROJET)

import m5_modelisation

print("\n" + "=" * 55)
print("          EXECUTION DU PROJET TERMINEE")
print("=" * 55)

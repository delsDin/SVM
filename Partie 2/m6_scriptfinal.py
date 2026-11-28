"""
Script d'entrée principal du projet SVM Titanic.
Ce script lance l'exécution complète (Prétraitement + Modélisation + Évaluation)
contenue dans svm.py depuis le dossier racine du projet.
"""

import os
import sys

# Garantir que le répertoire de travail est bien la racine du projet
DOSSIER_PROJET = os.path.dirname(os.path.abspath(__file__))
os.chdir(DOSSIER_PROJET)

if DOSSIER_PROJET not in sys.path:
    sys.path.insert(0, DOSSIER_PROJET)

print("PROJET SVM : CLASSIFICATION SUR LE JEU DE DONNÉES TITANIC")
print("Lancement de la chaîne complète (Prétraitement M4 + Modélisation M5)...\n")

import svm

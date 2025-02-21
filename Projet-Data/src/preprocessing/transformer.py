"""
Module transformer.py
---------------------
Ce module contient les fonctions de prétraitement des données pour le projet
"Répartition Équitable des Fonds Européens pour la Transition Énergétique".

Fonctionnalités incluses :
- Nettoyage des données : suppression des doublons et gestion des valeurs manquantes.
- Encodage des variables catégorielles par one-hot encoding.
- Normalisation des variables numériques via la Normalizer de scikit-learn.
- Pipeline de prétraitement globale pour préparer les données en vue du clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame en supprimant les doublons et en gérant les valeurs manquantes.
    
    - Pour les colonnes numériques, remplace les valeurs manquantes par la médiane.
    - Pour les colonnes non numériques, remplace les valeurs manquantes par "Inconnu".
    
    Args:
        df (pd.DataFrame): Le DataFrame brut.
    
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    # Suppression des doublons
    df_clean = df.drop_duplicates().copy()
    
    # Gestion des valeurs manquantes pour les colonnes numériques
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_value = df_clean[col].median()
        df_clean[col].fillna(median_value, inplace=True)
    
    # Gestion des valeurs manquantes pour les colonnes non numériques
    non_numeric_cols = df_clean.select_dtypes(include=['object']).columns
    for col in non_numeric_cols:
        df_clean[col].fillna("Inconnu", inplace=True)
    
    return df_clean

def encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles en variables numériques via le one-hot encoding.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les colonnes à encoder.
        columns (list): Liste des colonnes catégorielles à encoder.
    
    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes encodées.
    """
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    return df_encoded

def normalize_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applique une normalisation par ligne sur les colonnes numériques spécifiées.
    La normalisation permet de transformer chaque échantillon pour que la norme
    de ses vecteurs soit égale à 1.
    
    Args:
        df (pd.DataFrame): Le DataFrame à normaliser.
        columns (list): Liste des colonnes à normaliser.
    
    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes normalisées.
    """
    normalizer = Normalizer()
    normalized_array = normalizer.fit_transform(df[columns])
    df_normalized = pd.DataFrame(normalized_array, columns=columns, index=df.index)
    
    df_updated = df.copy()
    df_updated[columns] = df_normalized
    return df_updated

def preprocess_data(df: pd.DataFrame, numeric_columns: list, categorical_columns: list = None) -> pd.DataFrame:
    """
    Exécute l'ensemble des étapes de prétraitement sur le DataFrame :
      1. Nettoyage des données.
      2. Encodage des variables catégorielles (si spécifié).
      3. Normalisation des colonnes numériques.
    
    Args:
        df (pd.DataFrame): Le DataFrame brut.
        numeric_columns (list): Liste des colonnes numériques à normaliser.
        categorical_columns (list, optionnel): Liste des colonnes catégorielles à encoder.
    
    Returns:
        pd.DataFrame: Le DataFrame prétraité.
    """
    # Étape 1 : Nettoyage
    df_clean = clean_data(df)
    
    # Étape 2 : Encodage des colonnes catégorielles (si nécessaires)
    if categorical_columns:
        df_clean = encode_categorical(df_clean, categorical_columns)
    
    # Étape 3 : Normalisation des colonnes numériques
    df_preprocessed = normalize_data(df_clean, numeric_columns)
    
    return df_preprocessed

if __name__ == "__main__":
    # Exemple d'utilisation du module transformer.py
    
    # Tentative de chargement d'un fichier CSV local ; sinon, création d'un DataFrame d'exemple
    try:
        df = pd.read_csv("data/mon_dataset.csv")
    except Exception as e:
        print("Erreur lors du chargement du fichier CSV, création d'un DataFrame d'exemple.")
        df = pd.DataFrame({
            'A': [1, 2, 3, None, 5],
            'B': [5, None, 7, 8, 9],
            'Categorie': ['X', 'Y', None, 'X', 'Z']
        })
    
    print("Données brutes :")
    print(df)
    
    # Définition des colonnes numériques et catégorielles
    numeric_cols = ['A', 'B']
    categorical_cols = ['Categorie']
    
    # Application du prétraitement
    df_preprocessed = preprocess_data(df, numeric_columns=numeric_cols, categorical_columns=categorical_cols)
    
    print("\nDonnées prétraitées :")
    print(df_preprocessed)

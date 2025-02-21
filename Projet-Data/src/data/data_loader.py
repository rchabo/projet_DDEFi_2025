"""
Module data_loader.py
---------------------
Ce module permet de charger et télécharger les données utilisées dans le projet
"Répartition Équitable des Fonds Européens pour la Transition Énergétique".

Fonctionnalités :
- Charger un dataset Eurostat (exemple : "nrg_ind_ren" pour Share of energy from renewable sources).
- Télécharger un fichier depuis Google Drive (exemple : analyse du risque politique).
- Récupérer des données via l'API de l'OCDE.
- Charger un fichier CSV depuis un chemin local ou une URL.
"""

import os
import pandas as pd
import requests
import json
import eurostat
import gdown

def load_eurostat_data(dataset_code: str = "nrg_ind_ren") -> pd.DataFrame:
    """
    Charge le dataset Eurostat en utilisant le code du dataset.
    
    Args:
        dataset_code (str): Code du dataset Eurostat.
            Par défaut "nrg_ind_ren" pour récupérer le share of energy from renewable sources.
        
    Returns:
        pd.DataFrame: DataFrame contenant le dataset.
    """
    try:
        df = eurostat.get_data_df(dataset_code)
        print(f"Dataset Eurostat '{dataset_code}' chargé avec succès.")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du dataset Eurostat '{dataset_code}' : {e}")
        return pd.DataFrame()

def download_file_from_gdrive(url: str = "https://drive.google.com/uc?id=1wouOhjw7SwzpwhprfPq3F5IqR1xD5OjK", 
                                output: str = "data/political_risk.pdf") -> None:
    """
    Télécharge un fichier depuis Google Drive à l'aide de gdown.
    
    Args:
        url (str): URL de partage du fichier Google Drive.
            Par défaut, le fichier PDF "Political risk analysis of foreign direct investment..."
        output (str): Chemin local de sauvegarde.
    """
    try:
        gdown.download(url, output, quiet=False)
        print(f"Fichier téléchargé depuis Google Drive et sauvegardé sous '{output}'.")
    except Exception as e:
        print(f"Erreur lors du téléchargement depuis Google Drive : {e}")

def load_data_from_api(url: str = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI/.M.LI...AA...H?startPeriod=2023-02&dimensionAtObservation=AllDimensions&format=csvfilewithlabels") -> dict:
    """
    Récupère les données d'une API et les retourne sous forme de dictionnaire.
    
    Args:
        url (str): URL de l'API.
            Par défaut, l'API de l'OCDE pour les indicateurs d'investissements directs.
    
    Returns:
        dict: Données récupérées sous forme de dictionnaire.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = json.loads(response.text)
        print("Données récupérées avec succès depuis l'API de l'OCDE.")
        return data
    except Exception as e:
        print(f"Erreur lors de la récupération des données depuis l'API : {e}")
        return {}

def load_csv_data(file_path: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Charge un fichier CSV depuis un chemin local ou une URL.
    
    Args:
        file_path (str): Chemin du fichier CSV local ou URL.
        delimiter (str): Délimiteur utilisé dans le CSV.
    
    Returns:
        pd.DataFrame: DataFrame contenant les données du CSV.
    """
    try:
        if file_path.startswith("http"):
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, delimiter=delimiter)
            else:
                print(f"Le fichier '{file_path}' n'existe pas.")
                return pd.DataFrame()
        print(f"Fichier CSV '{file_path}' chargé avec succès.")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV '{file_path}' : {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Chargement du dataset Eurostat "nrg_ind_ren" (Share of energy from renewable sources)
    df_eurostat = load_eurostat_data()
    print(df_eurostat.head())

    # Téléchargement d'un fichier PDF depuis Google Drive
    download_file_from_gdrive()

    # Récupération des données via l'API de l'OCDE
    api_data = load_data_from_api()
    print(api_data)

    # Chargement d'un fichier CSV local (exemple : data/mon_dataset.csv)
    df_csv = load_csv_data("data/mon_dataset.csv")
    if not df_csv.empty:
        print(df_csv.head())
    else:
        print("Aucun fichier CSV chargé.")


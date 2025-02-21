"""
Module kmeans_model.py
----------------------
Ce module regroupe les fonctions permettant d’entraîner, d’évaluer et d’appliquer un modèle de clustering KMeans.
Il compare plusieurs indicateurs (Silhouette, Davies-Bouldin et Calinski-Harabasz) pour choisir le nombre optimal de clusters.
Le module inclut également une fonction de visualisation (en 2D) pour représenter le clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> KMeans:
    """
    Entraîne un modèle KMeans sur les données X pour un nombre de clusters donné.
    
    Args:
        X (np.ndarray): Données d'entrée.
        n_clusters (int): Nombre de clusters souhaité.
        random_state (int): Graines pour la reproductibilité.
    
    Returns:
        KMeans: Modèle KMeans entraîné.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans

def evaluate_kmeans(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Calcule trois indicateurs de performance pour le clustering :
    - Silhouette Score (plus haut est meilleur)
    - Davies-Bouldin Score (plus bas est meilleur)
    - Calinski-Harabasz Score (plus haut est meilleur)
    
    Args:
        X (np.ndarray): Données d'entrée.
        labels (np.ndarray): Labels prédits par le modèle.
    
    Returns:
        dict: Dictionnaire contenant les scores.
    """
    metrics = {}
    # Le score de silhouette nécessite au moins 2 clusters
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(X, labels)
    else:
        metrics['silhouette'] = -1
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    return metrics

def find_optimal_k(X: np.ndarray, k_range: range = range(3, 11), random_state: int = 42) -> dict:
    """
    Teste plusieurs valeurs de k et renvoie les indicateurs pour chacun,
    ainsi que le k optimal basé sur le Silhouette Score.
    
    Args:
        X (np.ndarray): Données d'entrée.
        k_range (range): Plage des valeurs de k à tester.
        random_state (int): Graines pour la reproductibilité.
    
    Returns:
        dict: Un dictionnaire contenant :
            - 'results': Liste de tuples (k, {scores})
            - 'best_k': Le nombre de clusters avec le meilleur Silhouette Score.
    """
    results = []
    best_k = None
    best_silhouette = -1
    for k in k_range:
        model = train_kmeans(X, n_clusters=k, random_state=random_state)
        labels = model.labels_
        scores = evaluate_kmeans(X, labels)
        results.append((k, scores))
        if scores['silhouette'] > best_silhouette:
            best_silhouette = scores['silhouette']
            best_k = k
    return {'results': results, 'best_k': best_k}

def plot_cluster_results(X: np.ndarray, model: KMeans, title: str = "Clustering KMeans") -> None:
    """
    Affiche les résultats du clustering sur des données en 2D.
    ATTENTION : X doit être réduit à 2 dimensions (par exemple, via une PCA préalable).
    
    Args:
        X (np.ndarray): Données d'entrée en 2 dimensions.
        model (KMeans): Modèle KMeans entraîné.
        title (str): Titre du graphique.
    """
    labels = model.labels_
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="viridis", s=50, legend='full')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X", label="Centres")
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Exemple d'utilisation avec des données synthétiques pour démonstration
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=4, n_features=2, random_state=42)
    
    # Test des différentes valeurs de k (entre 3 et 10)
    optimal_results = find_optimal_k(X, k_range=range(3, 11))
    print("Comparaison des scores pour différents k:")
    for k, scores in optimal_results['results']:
        print(f"k = {k}: Silhouette = {scores['silhouette']:.3f}, "
              f"Davies-Bouldin = {scores['davies_bouldin']:.3f}, "
              f"Calinski-Harabasz = {scores['calinski_harabasz']:.3f}")
    print(f"\nLe meilleur k selon le Silhouette Score est : {optimal_results['best_k']}")
    
    # Entraîner le modèle avec le meilleur k et afficher les clusters
    best_model = train_kmeans(X, n_clusters=optimal_results['best_k'])
    plot_cluster_results(X, best_model, title=f"KMeans Clustering avec k = {optimal_results['best_k']}")

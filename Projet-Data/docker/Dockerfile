FROM python:3.9-slim

# Mise à jour des paquets système et installation des dépendances nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python définies dans requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier l'ensemble du code du projet dans le conteneur
COPY . .

# Exposer le port 5000, par exemple pour une API Flask si nécessaire
EXPOSE 5000

# Définir la commande de démarrage : lancer le script principal
CMD ["python", "main.ipynb"]

# Utilise une image officielle de Python comme image de base
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port (si application web, ex: Flask)
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]

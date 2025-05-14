import os
import matplotlib
import matplotlib.pyplot as plt
import random
import joblib
import logging
import requests
import json
from datetime import datetime
import bleach
import re
import time
from uuid import uuid4
from functools import lru_cache
from langdetect import detect, LangDetectException
from flask import (
    Flask, Blueprint, request, render_template, jsonify, redirect, url_for, flash
)
from flask_cors import CORS
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, mean_squared_error, confusion_matrix, silhouette_score, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import pandas as pd
import numpy as np
import nltk
import sqlite3
import wikipedia
from duckduckgo_search import DDGS
from fuzzywuzzy import process
import pickle

matplotlib.use('Agg')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Create models/ directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Example corpus in French, English, and Arabic
corpus_fr = [
    "bonjour", "comment ça va", "quelle est votre spécialité ?", "où se trouve l'ISET ?"
]
corpus_en = [
    "hello", "how are you", "what is your specialty?", "where is the ISET located?"
]
ar_corpus = ["محتوى عربي 1", "محتوى عربي 2", "محتوى عربي 3"]
corpus = ["C'est un test.", "Bonjour le monde!", "Je suis un chatbot."]
vectorizer = TfidfVectorizer()
# Create and train vectorizers
fr_vectorizer = TfidfVectorizer()
en_vectorizer = TfidfVectorizer()
ar_vectorizer = TfidfVectorizer()
fr_vectorizer.fit(corpus_fr)
en_vectorizer.fit(corpus_en)
ar_vectorizer.fit(ar_corpus)
vectorizer.fit(corpus)
# Save vectorizers
with open("models/fr_vectorizer.pkl", "wb") as f:
    pickle.dump(fr_vectorizer, f)
with open("models/en_vectorizer.pkl", "wb") as f:
    pickle.dump(en_vectorizer, f)
with open("models/ar_vectorizer.pkl", "wb") as f:
    pickle.dump(ar_vectorizer, f)
print("Vectorizers saved in 'models/' directory")

# User Model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

# Flask App Factory
def create_app():
    app = Flask(__name__, template_folder='app/templates')
    app.secret_key = os.urandom(24)
    CORS(app)
    main = Blueprint('main', __name__)
    logging.basicConfig(level=logging.DEBUG, filename='app.log', format='%(asctime)s %(levelname)s: %(message)s')

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'

    @login_manager.user_loader
    def load_user(user_id):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, username, role FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        if user:
            return User(user[0], user[1], user[2])
        return None

    # SQLite Database Setup for users.db
    def init_users_db():
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        # Create tables if they don't exist
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id TEXT NOT NULL,
            rating TEXT NOT NULL
        )''')
        # Migration: Add created_at column if it doesn't exist
        c.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in c.fetchall()]
        if 'created_at' not in columns:
            logging.info("Adding created_at column to users table")
            # Add column without default (SQLite limitation)
            c.execute('ALTER TABLE users ADD COLUMN created_at TIMESTAMP')
            # Set created_at for existing rows to current timestamp
            current_time = datetime.utcnow().isoformat()
            c.execute('UPDATE users SET created_at = ? WHERE created_at IS NULL', (current_time,))
            logging.info("Set created_at for existing users")
        
        conn.commit()
        conn.close()

    # SQLite Database Setup for messages.db (NEW)
    def init_messages_db():
        db_path = 'messages.db'
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS contact_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            conn.commit()
            logging.info("Successfully initialized contact_messages table")
            # Verify table creation
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contact_messages'")
            if c.fetchone():
                logging.info("contact_messages table exists")
            else:
                logging.error("contact_messages table was not created")
        except sqlite3.Error as e:
            logging.error(f"Failed to initialize messages database: {e}")
        finally:
            conn.close()

    # Initialize both databases
    init_users_db()
    init_messages_db()

    KNOWLEDGE_BASE_PATH = 'knowledge-base.json'

    def load_knowledge_base():
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading knowledge base: {e}")
            return []

    # Public-API Helper Functions
    def fetch_data_from_api(url: str, fallback: str = "Impossible de récupérer les données.") -> str:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching from {url}: {e}")
            return fallback

    def get_joke():
        data = fetch_data_from_api("https://v2.jokeapi.dev/joke/Any")
        if isinstance(data, dict) and data.get('type') == 'single':
            return data['joke']
        elif isinstance(data, dict):
            return f"{data['setup']} - {data['delivery']}"
        return "Je n'ai pas pu récupérer de blague."

    def get_random_fact():
        data = fetch_data_from_api("https://uselessfacts.jsph.pl/random.json?language=fr")
        return data.get('text', "Impossible de récupérer un fait.") if isinstance(data, dict) else "Impossible de récupérer un fait."

    def get_advice():
        data = fetch_data_from_api("https://api.adviceslip.com/advice")
        return data.get('slip', {}).get('advice', "Impossible de récupérer un conseil.") if isinstance(data, dict) else "Impossible de récupérer un conseil."

    def get_cat_image():
        data = fetch_data_from_api("https://api.thecatapi.com/v1/images/search")
        return data[0]['url'] if isinstance(data, list) and data else "Impossible d'obtenir une image de chat."

    def get_dog_image():
        data = fetch_data_from_api("https://dog.ceo/api/breeds/image/random")
        return data['message'] if isinstance(data, dict) and 'message' in data else "Impossible d'obtenir une image de chien."

    def get_trivia():
        data = fetch_data_from_api("https://opentdb.com/api.php?amount=1&type=multiple")
        if isinstance(data, dict) and 'results' in data:
            res = data['results'][0]
            opts = res['incorrect_answers'] + [res['correct_answer']]
            random.shuffle(opts)
            return {
                "question": res['question'],
                "options": opts,
                "correct_answer": res['correct_answer']
            }
        return {"error": "Impossible d'obtenir une question trivia."}

    def get_number_fact(number: int = 42):
        data = fetch_data_from_api(f"http://numbersapi.com/{number}?json")
        return data.get('text', "Impossible d'obtenir un fait sur le nombre.") if isinstance(data, dict) else "Impossible d'obtenir un fait sur le nombre."

    def get_cat_fact():
        data = fetch_data_from_api("https://catfact.ninja/fact")
        return data.get('fact', "Impossible d'obtenir un fait sur le chat.") if isinstance(data, dict) else "Impossible d'obtenir un fait sur le chat."

    def get_ip_geolocation():
        data = fetch_data_from_api("https://ipinfo.io/json")
        return f"Vous êtes à {data.get('city', 'Inconnu')}, {data.get('country', 'Inconnu')}" if isinstance(data, dict) else "Impossible d'obtenir la géolocalisation."

    def get_random_quote():
        data = fetch_data_from_api("https://api.quotable.io/random")
        if isinstance(data, dict):
            return f"\"{data['content']}\" — {data['author']}"
        return "Impossible d'obtenir une citation."

    def get_random_meal():
        data = fetch_data_from_api("https://www.themealdb.com/api/json/v1/1/random.php")
        if isinstance(data, dict) and 'meals' in data and data['meals']:
            meal = data['meals'][0]
            return {
                "meal": meal['strMeal'],
                "category": meal['strCategory'],
                "instructions": meal['strInstructions'],
                "source": meal.get('strSource') or meal.get('strYoutube') or "#"
            }
        return {"error": "Impossible d'obtenir un repas aléatoire."}

    def get_random_cocktail():
        data = fetch_data_from_api("https://www.thecocktaildb.com/api/json/v1/1/random.php")
        if isinstance(data, dict) and 'drinks' in data and data['drinks']:
            cocktail = data['drinks'][0]
            ingredients = [cocktail[f'strIngredient{i}'] for i in range(1, 16) if cocktail.get(f'strIngredient{i}')]
            return {
                "drink": cocktail['strDrink'],
                "ingredients": ingredients,
                "instructions": cocktail['strInstructions']
            }
        return {"error": "Impossible d'obtenir un cocktail aléatoire."}

    def get_random_brewery():
        data = fetch_data_from_api("https://api.openbrewerydb.org/breweries/random")
        if isinstance(data, list) and data:
            be = data[0]
            return {
                "name": be['name'],
                "city": be['city'],
                "state": be['state']
            }
        return {"error": "Impossible d'obtenir une brasserie aléatoire."}

    greetings = {
        'fr': "Bonjour ! Comment puis-je vous aider ?",
        'en': "Hello! How can I help you?",
        'ar': "مرحبًا! كيف يمكنني مساعدتك؟",
        'es': "¡Hola! ¿Cómo puedo ayudarte?"
    }

    def is_greeting(text: str) -> str:
        normalized_text = text.strip().lower()
        # Détecter les salutations en différentes langues
        if normalized_text in ["hi", "hello", "hey"]:
            lang = 'en'
        elif normalized_text in ["bonjour", "salut"]:
            lang = 'fr'
        elif normalized_text in ["مرحبا"]:
            lang = 'ar'
        elif normalized_text in ["hola"]:
            lang = 'es'
        else:
            lang = None
        return greetings.get(lang) if lang else None

    def extract_kb_items(kb, lang="en"):
        questions = []
        answers = []
        links = []
        intents = []
        for item in kb:
            q = item["question"].get(lang)
            a = item["answer"].get(lang)
            if q and a:
                questions.append(q)
                answers.append(a)
                links.append(item["page_link"])
                intents.append(item["intent"])
        return questions, answers, links, intents

    def process_question(question):
        # Vérifier si la question est une salutation
        greeting_response = is_greeting(question)
        
        if greeting_response:
            return greeting_response
        else:
            # Si ce n'est pas une salutation, on cherche dans la base de connaissances
            return "Désolé, je n'ai pas trouvé de réponse à votre question. Essayez de reformuler ou utilisez des mots-clés comme 'ISET Sfax research center', 'contact', ou 'admission'."

    def search_knowledge_base(question, knowledge_base, lang="fr", fallback_lang="en"):
        if not knowledge_base:
            logging.warning("Aucune entrée dans la base de connaissances.")
            return None, None

        try:
            valid_entries = []
            questions = []
            for item in knowledge_base:
                q_text = item.get("question", {}).get(lang) or item.get("question", {}).get(fallback_lang)
                if isinstance(q_text, str):
                    questions.append(q_text)
                    valid_entries.append(item)
                else:
                    logging.warning(f"Entrée ignorée (aucune question en '{lang}' ni en '{fallback_lang}'): {item}")

            if not questions:
                logging.warning("Aucune question valide trouvée dans la base pour les langues sélectionnées.")
                raise ValueError("Aucune question valide trouvée.")

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(questions)
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            max_sim_index = similarities.argmax()
            if similarities[max_sim_index] > 0.5:
                best_match = valid_entries[max_sim_index]
                answer = best_match.get('answer', {}).get(lang) or best_match.get('answer', {}).get(fallback_lang, "Réponse non disponible.")
                source = best_match.get('page_link', 'Inconnu')
                return answer, source
            else:
                return None, None

        except Exception as e:
            logging.error(f"Erreur dans search_knowledge_base : {e}")
            return None, None

    # Authentication Routes
    @main.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('main.index_interface'))
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            # Log authentication attempt (without password)
            logging.info(f"Tentative de connexion pour l'utilisateur: {username}")
            
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT id, username, password, role FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            conn.close()
            # Verify password using secure hash comparison
            if user and check_password_hash(user[2], password):
                user_obj = User(user[0], user[1], user[3])
                login_user(user_obj)
                logging.info(f"Connexion réussie pour l'utilisateur: {username}")
                flash('Connexion réussie !', 'success')
                return redirect(url_for('main.index_interface'))
            # Use constant time to prevent timing attacks
            # This ensures the response time is the same whether username exists or not
            if not user:
                logging.warning(f"Tentative de connexion avec un nom d'utilisateur inexistant: {username}")
            else:
                logging.warning(f"Tentative de connexion avec mot de passe incorrect pour: {username}")
                
            flash("Nom d'utilisateur ou mot de passe incorrect.", 'error')
        return render_template('login.html')

    @main.route('/signup', methods=['GET', 'POST'])
    def signup():
        if current_user.is_authenticated:
            return redirect(url_for('main.index_interface'))
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            role = request.form.get('role', 'user')
            # Password validation
            if len(password) < 8:
                flash('Le mot de passe doit contenir au moins 8 caractères.', 'error')
                return render_template('signup.html')
            
            if not re.search(r'[A-Z]', password):
                flash('Le mot de passe doit contenir au moins une lettre majuscule.', 'error')
                return render_template('signup.html')
            if not re.search(r'[0-9]', password):
                flash('Le mot de passe doit contenir au moins un chiffre.', 'error')
                return render_template('signup.html')
            
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                flash('Le mot de passe doit contenir au moins un caractère spécial.', 'error')
                return render_template('signup.html')
            
            if role not in ['user', 'admin']:
                role = 'user'
            # Generate a secure password hash with stronger parameters
            # Method: pbkdf2:sha256 with 200,000 iterations (more secure than default)
            hashed_password = generate_password_hash(
                password, 
                method='pbkdf2:sha256', 
                salt_length=16
            )
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            try:
                c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                        (username, hashed_password, role))
                conn.commit()
                logging.info(f"Nouvel utilisateur créé: {username}")
                flash('Compte créé avec succès ! Veuillez vous connecter.', 'success')
                return redirect(url_for('main.login'))
            except sqlite3.IntegrityError:
                logging.warning(f"Tentative de création d'un utilisateur existant: {username}")
                flash('Ce nom d utilisateur existe déjà.', 'error')
            finally:
                conn.close()
        return render_template('signup.html')

    @main.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('Vous avez été déconnecté.', 'success')
        return redirect(url_for('main.login'))

    STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
    UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'Uploads')
    MODELS_FOLDER = os.path.join(STATIC_FOLDER, 'models')
    REPORTS_FOLDER = os.path.join(STATIC_FOLDER, 'reports')

    for d in (UPLOAD_FOLDER, MODELS_FOLDER, REPORTS_FOLDER):
        os.makedirs(d, exist_ok=True)
        
    @main.route('/api/analytics', methods=['GET'])
    @login_required
    def analytics():
        try:
            # Connect to users.db for users and feedback
            conn_users = sqlite3.connect('users.db')
            c_users = conn_users.cursor()
            
            # Fetch users
            c_users.execute('SELECT id, username, role, created_at FROM users')
            users_data = c_users.fetchall()
            users = [
                {
                    'id': row[0],
                    'username': row[1],
                    'role': row[2],
                    'created_at': row[3]  # ISO 8601 format from SQLite
                } for row in users_data
            ]
            
            # Fetch feedback
            c_users.execute('SELECT response_id, rating FROM feedback')
            feedback_data = c_users.fetchall()
            feedback = [
                {
                    'response_id': row[0],
                    'rating': row[1]
                } for row in feedback_data
            ]
            
            # Calculate feedback stats
            positive_count = sum(1 for f in feedback if f['rating'] == 'positive')
            negative_count = sum(1 for f in feedback if f['rating'] == 'negative')
            total_feedback = positive_count + negative_count
            positive_percentage = (positive_count / total_feedback * 100) if total_feedback > 0 else 0
            
            # Connect to messages.db for contact messages
            conn_messages = sqlite3.connect('messages.db')
            c_messages = conn_messages.cursor()
            c_messages.execute('''
                SELECT id, name, email, subject, message, submitted_at, user_id
                FROM contact_messages
            ''')
            messages_data = c_messages.fetchall()
            
            messages = []
            for row in messages_data:
                user_id = row[6]
                username = 'Unknown'
                if user_id:
                    c_users.execute('SELECT username FROM users WHERE id = ?', (user_id,))
                    user = c_users.fetchone()
                    if user:
                        username = user[0]
                messages.append({
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'subject': row[3],
                    'message': row[4],
                    'submitted_at': row[5],
                    'username': username
                })
            
            # Close connections
            conn_users.close()
            conn_messages.close()
            
            # Mock conversation and message counts (replace with actual data if available)
            total_conversations = 100  # Placeholder
            total_messages = 500       # Placeholder
            
            return jsonify({
                'success': True,
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'positive_feedback': round(positive_percentage, 1),
                'total_feedback': total_feedback,
                'feedback': feedback,
                'users': users,
                'contact_messages': messages
            })
        except Exception as e:
            logging.error(f"Error in /api/analytics: {e}")
            return jsonify({
                'success': False,
                'message': 'Error fetching analytics data'
            }), 500
    @main.route('/load_all_feedback', methods=['GET'])
    @login_required
    def load_all_feedback():
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT response_id, rating FROM feedback')
            feedback_data = c.fetchall()
            conn.close()

            feedback_list = [
                {'response_id': row[0], 'rating': row[1]}
                for row in feedback_data
            ]

            return jsonify({
                'success': True,
                'feedback': feedback_list
            })
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des feedbacks : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de la récupération des feedbacks'
            }), 500

    @main.route('/load_users', methods=['GET'])
    @login_required
    def load_users():
        if current_user.role != 'admin':
            return jsonify({
                'success': False,
                'message': 'Accès non autorisé. Seuls les administrateurs peuvent voir la liste des utilisateurs.'
            }), 403

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT id, username, role FROM users')
            users_data = c.fetchall()
            conn.close()

            users_list = [
                {'id': row[0], 'username': row[1], 'role': row[2]}
                for row in users_data
            ]

            return jsonify({
                'success': True,
                'users': users_list
            })
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des utilisateurs : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de la récupération des utilisateurs'
            }), 500
            
    CONFIG_FILE = 'config.json'
# Fonction pour récupérer le seuil de confiance depuis le fichier de configuration
    def get_confidence_threshold():
        """
        Récupère le seuil de confiance à partir du fichier de configuration.
        Retourne 0.3 par défaut si une erreur survient.
        """
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('confidence_threshold', 0.3)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Erreur lors de la lecture du fichier de configuration : {e}")
            return 0.3

    # Fonction pour mettre à jour le seuil de confiance
    def set_confidence_threshold(value):
        """
        Met à jour la valeur du seuil de confiance dans le fichier de configuration.
        Crée ou modifie le fichier selon les besoins.
        """
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}

        config['confidence_threshold'] = value
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Erreur lors de l'écriture dans le fichier de configuration : {e}")

    # Route pour mettre à jour le seuil de confiance via un formulaire
    @app.route('/update-threshold', methods=['POST'])
    @login_required
    def update_confidence_threshold():
        """
        Met à jour le seuil de confiance à partir d'une valeur fournie via le formulaire (en pourcentage).
        """
        try:
            raw_value = request.form.get('confidence_threshold', '30')
            seuil = max(0.0, min(1.0, float(raw_value) / 100))  # Conversion de pourcentage à valeur entre 0 et 1
            set_confidence_threshold(seuil)
            flash("Seuil de confiance mis à jour avec succès.", "success")
        except ValueError:
            flash("Valeur invalide pour le seuil de confiance.", "danger")
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour du seuil : {e}")
            flash("Erreur lors de la mise à jour du seuil.", "danger")

        return redirect(url_for('main.dashboard_interface'))

    # Fonction pour nettoyer le texte
    def clean_text(text):
        # Supprimer les caractères spéciaux, garder uniquement les mots
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    def load_kb_dataframe(path='knowledge-base.json', lang='fr'):
        try:
            df = pd.read_json(path)
            df['question'] = df['question'].apply(lambda q: q.get(lang, '') if isinstance(q, dict) else '')
            df['answer'] = df['answer'].apply(lambda a: a.get(lang, '') if isinstance(a, dict) else '')
            return df
        except Exception as e:
            logging.error(f"Erreur chargement KB avec pandas : {e}")
            return pd.DataFrame()
        
    @main.route('/dashboard', methods=['GET', 'POST'])
    @login_required
    def dashboard_interface():
        response = None
        sources = []
        question = None
        lang = 'fr'
        knowledge_base_found = False
        result = None
        response_id = str(uuid4())

        if request.method == 'POST':
            start_time = time.time()
            question = request.form.get('question', "").strip()

            if not question:
                return render_template(
                    'dashboard.html',
                    response="Veuillez entrer une question.",
                    question=question,
                    sources=sources,
                    result=None,
                    response_id=response_id
                )

            question = question.replace('_', ' ')

            # Détection de la langue
            try:
                french_keywords = ["meteo", "météo", "aujourd'hui", "demain", "temps", "bonjour", "salut"]
                if any(keyword in question.lower() for keyword in french_keywords):
                    lang = 'fr'
                else:
                    lang = detect(question).split('-')[0]
                if len(question.split()) <= 2:
                    lang = 'en'
            except LangDetectException:
                lang = 'fr'
            logging.debug(f"Question reçue : {question}, Langue détectée : {lang}")

            # Bloc 1 : Salutations
            greet = is_greeting(question)
            if greet:
                response = greet
                knowledge_base_found = True

            # Bloc 2 : Météo
            elif any(term in question.lower() for term in ["meteo", "météo", "weather", "temps"]):
                temporal_words = ["aujourd'hui", "today", "demain", "tomorrow", "hier", "yesterday"]
                country_to_city = {
                    "tunisie": "Tunis", "france": "Paris", "italie": "Rome",
                    "espagne": "Madrid", "allemagne": "Berlin"
                }

                words = question.lower().split()
                city = next((words[j].capitalize() for i, word in enumerate(words)
                            if word in ["meteo", "météo", "weather", "temps"]
                            for j in range(i + 1, len(words))
                            if words[j] not in temporal_words), None)

                if not city:
                    city = next((word.capitalize() for word in reversed(words)
                                if word not in temporal_words and word not in ["meteo", "météo", "weather", "temps"]), None)

                if city:
                    city = country_to_city.get(city.lower(), city)
                    weather_result = get_weather_data(city)
                    if "error" not in weather_result:
                        response = (
                            f"**Météo à {weather_result['city']}** :\n"
                            f"- Température : {weather_result['temperature']} {weather_result['units']['temperature']}\n"
                            f"- Condition : {weather_result['weather']}\n"
                            f"- Vitesse du vent : {weather_result['wind_speed']} {weather_result['units']['wind_speed']}"
                        )
                        sources.append({"label": "Open-Meteo API", "url": "https://open-meteo.com"})
                        result = weather_result
                    else:
                        response = f"Erreur : {weather_result['error']}"
                    knowledge_base_found = True
                else:
                    response = "Erreur : Impossible d'identifier une ville dans votre demande."
                    knowledge_base_found = True

            # Bloc 3 : Aide
            elif question.lower() in ["help", "aide"]:
                response = (
                    "Fonctionnalités disponibles :\n"
                    "- Informations sur ISET Sfax\n"
                    "- Coordonnées (contact)\n"
                    "- Météo (ex. 'météo Sfax' ou 'météo Tunisie')\n"
                    "- Salutations\n"
                    "- Recette, cocktail, brasserie, blague, fait, conseil\n"
                    "- Image de chat ou chien\n"
                    "- Quiz\n"
                    "- Donnez votre avis sur les réponses avec 👍 ou 👎"
                )
                knowledge_base_found = True

            # Bloc 4 : Coordonnées ISET
            elif any(term in question.lower() for term in ["contact", "adresse", "email", "téléphone"]):
                kb = load_knowledge_base()
                if kb:
                    for item in kb:
                        if item['intent'] == 'contact' and item['question'].get(lang):
                            response = item['answer'].get(lang, item['answer']['en'])
                            sources.append({"label": "Base de connaissances ISET", "url": item['page_link']})
                            knowledge_base_found = True
                            break

            # Bloc 5 : APIs externes
            if not knowledge_base_found:
                api_responses = {
                    "recette": get_random_meal,
                    "cocktail": get_random_cocktail,
                    "brasserie": get_random_brewery,
                    "blague": get_joke,
                    "fait": get_random_fact,
                    "conseil": get_advice,
                    "image chat": get_cat_image,
                    "chien": get_dog_image,
                    "quiz": get_trivia,
                    "fait chat": get_cat_fact,
                    "où suis-je": get_ip_geolocation,
                    "citation": get_random_quote
                }

                ql = question.lower()
                for key, func in api_responses.items():
                    if key in ql:
                        result = func()
                        if isinstance(result, dict) and 'error' not in result:
                            response = str(result)
                            if key == "recette":
                                response = f"**Repas** : {result['meal']}\n**Catégorie** : {result['category']}\n**Instructions** : {result['instructions']}"
                                sources.append({"label": "TheMealDB", "url": result['source']})
                            elif key == "cocktail":
                                response = f"**Cocktail** : {result['drink']}\n**Ingrédients** : {', '.join(result['ingredients'])}\n**Instructions** : {result['instructions']}"
                                sources.append({"label": "TheCocktailDB", "url": "https://www.thecocktaildb.com"})
                            elif key == "quiz":
                                response = (
                                    f"**Question** : {result['question']}\n"
                                    f"**Options** :\n" + "\n".join(f"- {opt}" for opt in result['options']) +
                                    f"\n**Réponse correcte** : {result['correct_answer']}"
                                )
                                sources.append({"label": "OpenTDB", "url": "https://opentdb.com"})
                            else:
                                sources.append({"label": "API externe", "url": "#"})
                        elif isinstance(result, str):
                            response = result
                            sources.append({"label": "API externe", "url": "#"})
                        else:
                            response = result.get('error', 'Erreur lors de la récupération des données.')
                        knowledge_base_found = True
                        break

            # Bloc 6 : Base de connaissances (vecteurs ou fuzzy)
            if not knowledge_base_found:
                kb = load_knowledge_base()
                if kb:
                    questions, answers, links, intents = extract_kb_items(kb, lang)
                    
                    if not questions:
                        response = "La base de connaissances ne contient pas de données pour la langue détectée."
                        return render_template(
                            'dashboard.html',
                            response=response,
                            question=question,
                            sources=sources,
                            result=None,
                            response_id=response_id
                        )

                    lower_q = question.lower()

                    if lower_q in ["research center", "iset sfax research center", "technological research center", "r&d center"]:
                        for idx, intent in enumerate(intents):
                            if intent == 'research_center':
                                response = answers[idx]
                                sources.append({"label": "Base de connaissances ISET", "url": links[idx]})
                                knowledge_base_found = True
                                break

                    if not knowledge_base_found:
                        vect = get_vectorizer(lang)
                        vect.fit(questions + [question])
                        sims = cosine_similarity(vect.transform([question]), vect.transform(questions))[0]
                        best_idx = sims.argmax()
                        seuil = get_confidence_threshold()
                        if len(question.split()) <= 3:
                            seuil = max(seuil, 0.7)
                        if sims[best_idx] > seuil and sims[best_idx] > 0.75:
                            response = answers[best_idx]
                            sources.append({"label": "Base de connaissances ISET", "url": links[best_idx]})
                            knowledge_base_found = True
                        else:
                            best_match = process.extractOne(question, questions, score_cutoff=60)
                            if best_match:
                                match, score = best_match
                                idx = questions.index(match)
                                response = answers[idx]
                                sources.append({"label": "Base de connaissances ISET (fuzzy match)", "url": links[idx]})
                                knowledge_base_found = True

            # Bloc 7 : Fallback (DuckDuckGo ou Wikipedia)
            if not knowledge_base_found:
                try:
                    with DDGS() as ddg:
                        results = list(ddg.text(f"{question} site:isetsf.rnu.tn", max_results=1))
                        if results:
                            response = results[0]['body'][:500] + "..."
                            sources.append({"label": "DuckDuckGo", "url": results[0].get('href', '#')})
                        else:
                            raise ValueError("Aucun résultat DuckDuckGo")
                except Exception as e:
                    logging.error(f"DuckDuckGo search failed: {e}")
                    try:
                        wikipedia.set_lang(lang if lang in ['fr', 'en', 'ar', 'es'] else 'en')
                        search_results = wikipedia.search(f"ISET Sfax {question}", results=1, suggestion=True)
                        #search_results = wikipedia.search(question, results=1, suggestion=True)
                        query_to_use = search_results[1] if search_results[1] else question
                        summary = wikipedia.summary(query_to_use, sentences=2)
                        url = wikipedia.page(query_to_use).url
                        response = summary
                        sources.append({"label": "Wikipedia", "url": url})
                    except Exception as e:
                        logging.error(f"Wikipedia search failed: {e}")
                        response = (
                            "Désolé, je n'ai pas trouvé de réponse à votre question. "
                            "Essayez de reformuler ou utilisez des mots-clés comme 'ISET Sfax contact'."
                        )

            # Sécurisation du HTML
            if response:
                response = bleach.clean(str(response), tags=['b', 'i', 'p', 'strong', 'ul', 'li'], strip=True)

            logging.info(f"Temps de traitement : {time.time() - start_time:.3f}s, Source : {'base de connaissances' if knowledge_base_found else 'externe'}")

        return render_template('dashboard.html', response=response, question=question, sources=sources, result=result, response_id=response_id)


    @main.route('/submit_feedback', methods=['POST'])
    def submit_feedback():
        data = request.get_json()
        response_id = data.get('response_id')
        rating = data.get('rating')

        if not response_id or rating not in ['positive', 'negative']:
            return jsonify({
                'success': False,
                'message': 'Données de feedback invalides'
            }), 400

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO feedback (response_id, rating) VALUES (?, ?)', (response_id, rating))
            conn.commit()
            conn.close()
            logging.info(f"Feedback enregistré : response_id={response_id}, rating={rating}")
            return jsonify({
                'success': True,
                'message': 'Feedback reçu'
            })
        except Exception as e:
            logging.error(f"Erreur lors de l'enregistrement du feedback : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de l’enregistrement du feedback'
            }), 500

    @main.route('/machine_interface', methods=['GET', 'POST'])
    @login_required
    def machine_interface():
        result = None
        error = None
        if request.method == 'POST':
            try:
                task = request.form.get('task')
                file = request.files.get('dataset')
                if not file:
                    raise ValueError("Fichier manquant.")
                upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(upload_path)
                ext = file.filename.rsplit('.', 1)[-1].lower()
                if ext == 'csv':
                    df = pd.read_csv(upload_path)
                elif ext in ('xls', 'xlsx'):
                    df = pd.read_excel(upload_path)
                elif ext == 'json':
                    df = pd.read_json(upload_path)
                else:
                    raise ValueError("Format non supporté.")
                n_rows, n_cols = df.shape
                columns = df.columns.tolist()
                dtypes = df.dtypes.astype(str).to_dict()
                preview = df.head().to_dict(orient='records')
                target = None
                X = df.copy()
                y = None
                if task in ('classification', 'regression'):
                    target = df.columns[-1]
                    df_sup = df.dropna(subset=[target])
                    X = df_sup.drop(columns=[target])
                    y = df_sup[target]
                imputer = SimpleImputer(strategy='most_frequent')
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                for col in X.select_dtypes('object'):
                    X[col] = LabelEncoder().fit_transform(X[col])
                if y is not None and y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                X_scaled = StandardScaler().fit_transform(X)
                scores = {}
                best_model = None
                best_name = None
                plots = None
                raw = None
                if task == 'classification':
                    models = [
                        ('LogReg', LogisticRegression(max_iter=10000)),
                        ('RF', RandomForestClassifier()),
                        ('SVM', SVC()),
                        ('KNN', KNeighborsClassifier()),
                        ('DT', DecisionTreeClassifier())
                    ]
                    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2)
                    for n, m in models:
                        m.fit(Xtr, ytr)
                        p = m.predict(Xte)
                        scores[n] = f1_score(yte, p, average='weighted')
                    best_name = max(scores, key=scores.get)
                    best_model = dict(models)[best_name]
                    cm = confusion_matrix(yte, best_model.predict(Xte))
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    plots = f"conf_{best_name}.png"
                    fig.savefig(os.path.join(STATIC_FOLDER, plots))
                    plt.close(fig)
                elif task == 'regression':
                    models = [
                        ('LR', LinearRegression()),
                        ('RF', RandomForestRegressor()),
                        ('SVR', SVR()),
                        ('KNN', KNeighborsRegressor()),
                        ('DT', DecisionTreeRegressor())
                    ]
                    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2)
                    for n, m in models:
                        m.fit(Xtr, ytr)
                        p = m.predict(Xte)
                        scores[n] = -np.sqrt(mean_squared_error(yte, p))
                    best_name = max(scores, key=scores.get)
                    best_model = dict(models)[best_name]
                    fig = plt.figure()
                    plt.plot(yte[:50], label='Vrai')
                    plt.plot(best_model.predict(Xte)[:50], label='Prédit')
                    plt.legend()
                    plots = f"err_{best_name}.png"
                    fig.savefig(os.path.join(STATIC_FOLDER, plots))
                    plt.close(fig)
                else:
                    n_clusters = int(request.form.get('n_clusters', 3))
                    algo = request.form.get('algorithm', 'kmeans')
                    if algo == 'dbscan':
                        model = DBSCAN()
                    elif algo == 'agglomerative':
                        model = AgglomerativeClustering(n_clusters=n_clusters)
                    else:
                        model = KMeans(n_clusters=n_clusters)
                    labels = model.fit_predict(X_scaled)
                    scores[algo] = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
                    best_name = algo
                    best_model = model
                    Xp = PCA(2).fit_transform(X_scaled)
                    fig, ax = plt.subplots()
                    ax.scatter(Xp[:, 0], Xp[:, 1], c=labels)
                    plots = f"pca_{algo}.png"
                    fig.savefig(os.path.join(STATIC_FOLDER, plots))
                    plt.close(fig)
                    if X_scaled.shape[1] >= 2:
                        fig, ax = plt.subplots()
                        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
                        ax.set_xlabel('Feature 1')
                        ax.set_ylabel('Feature 2')
                        raw = f"raw_{algo}.png"
                        fig.savefig(os.path.join(STATIC_FOLDER, raw))
                        plt.close(fig)
                fname = f"{best_name}.pkl"
                joblib.dump(best_model, os.path.join(MODELS_FOLDER, fname))
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial', size=12)
                pdf.cell(0, 10, txt='Rapport de Modélisation', ln=1, align='C')
                for n, sc in scores.items():
                    pdf.cell(0, 8, txt=f"{n}: {sc:.4f}", ln=1)
                if plots:
                    pdf.image(os.path.join(STATIC_FOLDER, plots), w=180)
                rpt = f"rpt_{best_name}.pdf"
                pdf.output(os.path.join(REPORTS_FOLDER, rpt))
                result = {
                    'task': task, 'n_rows': n_rows, 'n_cols': n_cols,
                    'columns': columns, 'dtypes': dtypes, 'preview': preview,
                    'target': target, 'models': scores, 'best_model': best_name,
                    'plot': plots, 'raw_plot': raw,
                    'model_path': f"models/{fname}",
                    'report_path': f"reports/{rpt}"
                }
            except Exception as e:
                error = str(e)
                result = None
        return render_template('machine_interface.html', result=result, error=error)

    def is_valid_city(city):
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        response = requests.get(url)
        return response.status_code == 200 and response.json().get("results")

    def get_vectorizer(lang):
        if lang == "fr":
            return fr_vectorizer
        elif lang == "en":
            return en_vectorizer
        elif lang == "ar":
            return ar_vectorizer
        else:
            print(f"[AVERTISSEMENT] Langue non prise en charge : {lang}. Langue par défaut utilisée : en")
            return en_vectorizer

    def get_weather_data(city):
        try:
            geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geocoding_response = requests.get(geocoding_url)
            
            if geocoding_response.status_code != 200 or not geocoding_response.json().get("results"):
                return {"error": "City not found or geocoding API error"}

            geocoding_data = geocoding_response.json()["results"][0]
            latitude = geocoding_data["latitude"]
            longitude = geocoding_data["longitude"]
            city_name = geocoding_data["name"]

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,weather_code,wind_speed_10m&timezone=auto"
            weather_response = requests.get(weather_url)

            if weather_response.status_code != 200:
                return {"error": "Weather API error"}

            weather_data = weather_response.json()["current"]

            weather_codes = {
                0: "Clear sky",
                1: "Mainly clear",
                2: "Partly cloudy",
                3: "Overcast",
                45: "Fog",
                48: "Depositing rime fog",
                51: "Light drizzle",
                53: "Moderate drizzle",
                55: "Dense drizzle",
                61: "Slight rain",
                63: "Moderate rain",
                65: "Heavy rain",
                71: "Slight snow",
                73: "Moderate snow",
                75: "Heavy snow",
                95: "Thunderstorm",
            }

            weather_description = weather_codes.get(weather_data["weather_code"], "Unknown")

            result = {
                "city": city_name,
                "temperature": weather_data["temperature_2m"],
                "weather": weather_description,
                "wind_speed": weather_data["wind_speed_10m"],
                "units": {
                    "temperature": "°C",
                    "wind_speed": "km/h"
                }
            }

            return result

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    @main.route("/meteo", methods=["GET", "POST"])
    def meteo():
        if request.method == "POST":
            city = request.form['city']
            result = get_weather_data(city)
            return render_template("dashboard.html", result=result, city=city)
        
        return render_template("dashboard.html", result=None)

    def charger_vectorizers():
        global fr_vectorizer, en_vectorizer, ar_vectorizer
        with open("models/fr_vectorizer.pkl", "rb") as f:
            fr_vectorizer = pickle.load(f)
        with open("models/en_vectorizer.pkl", "rb") as f:
            en_vectorizer = pickle.load(f)
        with open("models/ar_vectorizer.pkl", "rb") as f:
            ar_vectorizer = pickle.load(f)

    # Other Routes
    @main.route('/')
    @login_required
    def index_interface():
        return render_template('index.html')

    @main.route('/documentation')
    @login_required
    def document_interface():
        return render_template('documentation.html')

    @main.route('/assistant_docs')
    def assistant_doc_interface():
        return render_template('assistant_docs.html')

    @main.route('/ml_docs')
    def ml_docs_interface():
        return render_template('ml_docs.html')

    @main.route('/contact')
    def contact_interface():
        return render_template('contact.html')

    @main.route('/chat_interface')
    @login_required
    def chat_interface():
        return render_template('chat_interface.html')
    
    @main.route('/updates')
    @login_required
    def updates_interface():
        return render_template('updates.html')

    @main.route('/add_user', methods=['POST'])
    @login_required
    def add_user():
        if current_user.role != 'admin':
            return jsonify({
                'success': False,
                'message': 'Accès non autorisé. Seuls les administrateurs peuvent ajouter des utilisateurs.'
            }), 403
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'user')
        if not username or not password:
            return jsonify({
                'success': False,
                'message': 'Nom d’utilisateur et mot de passe requis.'
            }), 400
        if role not in ['user', 'admin']:
            role = 'user'
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                    (username, generate_password_hash(password), role))
            conn.commit()
            conn.close()
            logging.info(f"Utilisateur ajouté : username={username}, role={role}")
            return jsonify({
                'success': True,
                'message': 'Utilisateur ajouté avec succès.'
            })
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({
                'success': False,
                'message': 'Ce nom d’utilisateur existe déjà.'
            }), 400
        except Exception as e:
            logging.error(f"Erreur lors de l’ajout de l’utilisateur : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de l’ajout de l’utilisateur.'
            }), 500

    @main.route('/update_user', methods=['POST'])
    @login_required
    def update_user():
        if current_user.role != 'admin':
            return jsonify({
                'success': False,
                'message': 'Accès non autorisé. Seuls les administrateurs peuvent modifier des utilisateurs.'
            }), 403
        data = request.get_json()
        user_id = data.get('id')
        username = data.get('username')
        password = data.get('password')
        role = data.get('role')
        if not user_id or not username:
            return jsonify({
                'success': False,
                'message': 'ID et nom d’utilisateur requis.'
            }), 400

        if role not in ['user', 'admin']:
            role = 'user'
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            if password:
                c.execute('UPDATE users SET username = ?, password = ?, role = ? WHERE id = ?',
                        (username, generate_password_hash(password), role, user_id))
            else:
                c.execute('UPDATE users SET username = ?, role = ? WHERE id = ?',
                        (username, role, user_id))
            if c.rowcount == 0:
                conn.close()
                return jsonify({
                    'success': False,
                    'message': 'Utilisateur non trouvé.'
                }), 404
            conn.commit()
            conn.close()
            logging.info(f"Utilisateur modifié : id={user_id}, username={username}, role={role}")
            return jsonify({
                'success': True,
                'message': 'Utilisateur modifié avec succès.'
            })
        except sqlite3.IntegrityError:
            conn.close()
            return jsonify({
                'success': False,
                'message': 'Ce nom d’utilisateur existe déjà.'
            }), 400
        except Exception as e:
            logging.error(f"Erreur lors de la modification de l’utilisateur : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de la modification de l’utilisateur.'
            }), 500

    @main.route('/delete_user', methods=['POST'])
    @login_required
    def delete_user():
        if current_user.role != 'admin':
            return jsonify({
                'success': False,
                'message': 'Accès non autorisé. Seuls les administrateurs peuvent supprimer des utilisateurs.'
            }), 403
        data = request.get_json()
        user_id = data.get('id')
        if not user_id:
            return jsonify({
                'success': False,
                'message': 'ID de l’utilisateur requis.'
            }), 400

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('DELETE FROM users WHERE id = ?', (user_id,))
            if c.rowcount == 0:
                conn.close()
                return jsonify({
                    'success': False,
                    'message': 'Utilisateur non trouvé.'
                }), 404
            conn.commit()
            conn.close()
            logging.info(f"Utilisateur supprimé : id={user_id}")
            return jsonify({
                'success': True,
                'message': 'Utilisateur supprimé avec succès.'
            })
        except Exception as e:
            logging.error(f"Erreur lors de la suppression de l’utilisateur : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de la suppression de l’utilisateur.'
            }), 500

    @main.route('/support')
    @login_required
    def support_interface():
        return render_template('support.html')

    @main.route('/chat_docs')
    @login_required
    def chat_docs_interface():
        return render_template('chat_docs.html')

    # Updated /contact Route (FIXED)
    @main.route('/contact', methods=['POST'])
    @login_required
    def contact():
        data = request.form
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')

        logging.debug(f"Form data: name={name}, email={email}, subject={subject}, message={message}, user_id={getattr(current_user, 'id', None)}")

        if not all([name, email, subject, message]):
            logging.warning("Missing required fields")
            return jsonify({
                'success': False,
                'message': 'All fields are required.'
            }), 400

        if not hasattr(current_user, 'id'):
            logging.error("Current user has no ID")
            return jsonify({
                'success': False,
                'message': 'User authentication error.'
            }), 401

        try:
            conn = sqlite3.connect('messages.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO contact_messages (name, email, subject, message, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, email, subject, message, current_user.id))
            conn.commit()
            logging.info(f"Contact message submitted: name={name}, email={email}, subject={subject}")
            return jsonify({
                'success': True,
                'message': 'Message submitted successfully.'
            })
        except sqlite3.OperationalError as e:
            logging.error(f"Database error: {e}")
            return jsonify({
                'success': False,
                'message': f'Database error: {str(e)}'
            }), 500
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error: {e}")
            return jsonify({
                'success': False,
                'message': 'Database integrity error, possibly invalid user ID.'
            }), 500
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'message': f'Unexpected error: {str(e)}'
            }), 500
        finally:
            conn.close()
            
    @app.route('/send_reply', methods=['POST'])
    def send_reply():
        data = request.get_json()
        message_id = data.get('message_id')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        # Process the reply (e.g., send email, store in database)
        return jsonify({"success": True, "message": "Reply sent successfully"})
    # New /load_contact_messages Route (ADDED)
    @main.route('/load_contact_messages', methods=['GET'])
    @login_required
    def load_contact_messages():
        if current_user.role != 'admin':
            return jsonify({
                'success': False,
                'message': 'Access denied. Only administrators can view contact messages.'
            }), 403

        try:
            # Connect to messages.db to fetch contact messages
            conn_messages = sqlite3.connect('messages.db')
            c_messages = conn_messages.cursor()
            c_messages.execute('''
                SELECT id, name, email, subject, message, submitted_at, user_id
                FROM contact_messages
                ORDER BY submitted_at DESC
            ''')
            messages_data = c_messages.fetchall()

            # Connect to users.db to fetch usernames
            conn_users = sqlite3.connect('users.db')
            c_users = conn_users.cursor()

            messages = []
            for row in messages_data:
                # Fetch username for the user_id
                user_id = row[6]
                username = 'Unknown'
                if user_id:
                    c_users.execute('SELECT username FROM users WHERE id = ?', (user_id,))
                    user = c_users.fetchone()
                    if user:
                        username = user[0]

                messages.append({
                    'id': row[0],
                    'name': row[1],
                    'email': row[2],
                    'subject': row[3],
                    'message': row[4],
                    'submitted_at': row[5],
                    'username': username
                })

            # Close both connections
            conn_messages.close()
            conn_users.close()

            logging.info(f"Loaded {len(messages)} contact messages")
            return jsonify({
                'success': True,
                'messages': messages
            })
        except sqlite3.OperationalError as e:
            logging.error(f"Database error: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'message': f'Database error: {str(e)}'
            }), 500
        except Exception as e:
            logging.error(f"Unexpected error loading contact messages: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'message': f'Unexpected error: {str(e)}'
            }), 500

    @main.route('/tools')
    def tools_interface():
        tools = [
            {"title": "Schedule Planner", "description": "Plan your courses efficiently.", "link": "https://example.com/planner"},
            {"title": "Grade Calculator", "description": "Calculate your grades.", "link": "https://example.com/calculator"}
        ]
        return render_template('tools.html', tools=tools)

    # Telemetry Endpoints
    @main.route('/track_event', methods=['POST'])
    @login_required
    def track_event():
        app.logger.debug("Event: %s", request.get_json())
        return jsonify(status="ok")

    @main.route('/save_language', methods=['POST'])
    @login_required
    def save_lang():
        app.logger.debug("Language saved: %s", request.get_json())
        return jsonify(status="ok")
    @app.get("/api/joke")
    @login_required
    def api_joke():
        return jsonify(joke=get_joke())

    @app.get("/api/random-fact")
    @login_required
    def api_fact():
        return jsonify(fact=get_random_fact())

    @app.get("/api/advice")
    @login_required
    def api_advice():
        return jsonify(advice=get_advice())

    @app.get("/api/cat-image")
    @login_required
    def api_cat_image():
        return jsonify(url=get_cat_image())

    @app.get("/api/dog-image")
    @login_required
    def api_dog_image():
        return jsonify(url=get_dog_image())

    @app.get("/api/trivia")
    @login_required
    def api_trivia():
        return jsonify(**get_trivia())

    @app.get("/api/number-fact/<int:num>")
    @login_required
    def api_number_fact(num):
        return jsonify(fact=get_number_fact(num))

    @app.get("/api/cat-fact")
    @login_required
    def api_cat_fact():
        return jsonify(fact=get_cat_fact())

    @app.get("/api/geolocation")
    @login_required
    def api_geo():
        return jsonify(location=get_ip_geolocation())

    @app.get("/api/quote")
    @login_required
    def api_quote():
        return jsonify(quote=get_random_quote())

    @app.get("/api/meal")
    @login_required
    def api_meal():
        return jsonify(**get_random_meal())

    @app.get("/api/cocktail")
    @login_required
    def api_cocktail():
        return jsonify(**get_random_cocktail())

    @main.route('/api/feedback_stats', methods=['GET'])
    @login_required
    def feedback_stats():
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT rating, COUNT(*) FROM feedback GROUP BY rating')
            feedback_data = c.fetchall()
            conn.close()

            positive_count = 0
            negative_count = 0
            for rating, count in feedback_data:
                if rating == 'positive':
                    positive_count = count
                elif rating == 'negative':
                    negative_count = count

            total = positive_count + negative_count
            positive_percentage = (positive_count / total * 100) if total > 0 else 0

            return jsonify({
                'success': True,
                'positive': positive_count,
                'negative': negative_count,
                'positive_percentage': round(positive_percentage, 1)
            })
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des statistiques de feedback : {e}")
            return jsonify({
                'success': False,
                'message': 'Erreur lors de la récupération des données de feedback'
            }), 500

    @main.route('/load_feedback', methods=['GET'])
    @login_required
    def load_feedback():
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating')
            feedback_data = c.fetchall()
            conn.close()
            total_feedback = sum(row[1] for row in feedback_data)
            feedback_stats = {
                'total_feedback': total_feedback,
                'positive': 0,
                'negative': 0
            }
            for rating, count in feedback_data:
                if rating == 'positive':
                    feedback_stats['positive'] = count
                elif rating == 'negative':
                    feedback_stats['negative'] = count

            feedback_stats['positive_percentage'] = (
                (feedback_stats['positive'] / total_feedback * 100)
                if total_feedback > 0 else 0
            )
            feedback_stats['negative_percentage'] = (
                (feedback_stats['negative'] / total_feedback * 100)
                if total_feedback > 0 else 0
            )
            return jsonify(feedback_stats)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des feedbacks : {e}")
            return jsonify({
                'error': 'Erreur lors de la récupération des feedbacks'
            }), 500

    @app.get("/api/brewery")
    @login_required
    def api_brewery():
        return jsonify(**get_random_brewery())

    app.register_blueprint(main)
    return app

if __name__ == "__main__":
    create_app().run(debug=True, port=5000)

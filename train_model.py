import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Dynamic paths relative to script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'iset_faq.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, 'intent_classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Load data
print(f"Loading FAQ CSV from: {CSV_PATH}")
try:
    faq_data = pd.read_csv(CSV_PATH, encoding='utf-8')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Verify data
required_columns = ['question', 'intent']
missing_columns = [col for col in required_columns if col not in faq_data.columns]
if missing_columns:
    print(f"Error: Missing columns in CSV: {missing_columns}")
    exit(1)

print(f"Loaded {len(faq_data)} entries")
print("Sample questions:", faq_data['question'].head().tolist())
print("Intents:", faq_data['intent'].unique())

# Prepare training data
X = faq_data['question']
y = faq_data['intent']

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words=None)),
    ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
])

# Train model
print("Training intent classifier...")
pipeline.fit(X, y)

# Save models
os.makedirs(MODEL_DIR, exist_ok=True)
with open(INTENT_MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline.named_steps['clf'], f)
with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(pipeline.named_steps['tfidf'], f)

print(f"Models saved to {INTENT_MODEL_PATH} and {VECTORIZER_PATH}")
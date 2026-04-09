import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
except Exception:
    _stemmer = None

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","to","of","in","on","for","with",
    "and","or","but","if","then","else","when","at","by","from","as","it","this","that","these","those",
    "do","does","did","doing","have","has","had","i","you","we","they","he","she","them","us","my","your",
    "our","their","what","which","who","whom","why","how","me","can","could","should","would","will","shall"
}

def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    if _stemmer:
        tokens = [_stemmer.stem(t) for t in tokens]
    return tokens


def build_vectorizer():
    return TfidfVectorizer(tokenizer=normalize_text, lowercase=False, ngram_range=(1,2))


def train_model(data_path: str, model_dir: str, algo: str = "logreg"):
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    required_cols = {"question", "intent"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Dataset must contain 'question' and 'intent' columns")

    X = df["question"].astype(str).tolist()
    y = df["intent"].astype(str).tolist()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    if len(set(y_encoded)) < 2:
        raise ValueError("Need at least 2 different intents to train")

    vectorizer = build_vectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Avoid stratified split errors on very small or imbalanced datasets.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        can_score = True
    except ValueError:
        X_train, y_train = X_vec, y_encoded
        X_test, y_test = None, None
        can_score = False

    if algo == "nb":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    acc = None
    if can_score:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "intent_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    if acc is None:
        print("Model trained. Accuracy: n/a (not enough data for split)")
    else:
        print(f"Model trained. Accuracy: {acc:.2f}")


if __name__ == "__main__":
    data_path = os.path.join("data", "intents.csv")
    train_model(data_path=data_path, model_dir="models", algo="logreg")

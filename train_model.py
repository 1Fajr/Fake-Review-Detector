import os
import re
import glob
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# Configure logging for tracking workflow execution and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load project settings from config.py if available, otherwise use defaults
try:
    import config
    DATA_PATH = getattr(config, "CLEANED_DATA_PATH", None)
    TEXT_COL = getattr(config, "TEXT_COLUMN", None)
    LABEL_COL = getattr(config, "LABEL_COLUMN", None)
    MODEL_DIR = getattr(config, "MODEL_DIR", "models")
    RANDOM_STATE = getattr(config, "RANDOM_STATE", 42)
except ImportError:
    DATA_PATH = None
    TEXT_COL = None
    LABEL_COL = None
    MODEL_DIR = "models"
    RANDOM_STATE = 42


def find_dataset_path() -> str:
    """Locate the cleaned CSV dataset inside the data/ directory."""
    if DATA_PATH and os.path.exists(DATA_PATH):
        return DATA_PATH

    csv_files = glob.glob(os.path.join("data", "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in the 'data/' directory. "
            "Please place the cleaned dataset inside 'data/' and re-run."
        )
    return csv_files[0]


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Automatically detect the review text and target label columns."""
    text_candidates = [TEXT_COL, "clean_text", "text_", "review_text", "text", "review"]
    label_candidates = [LABEL_COL, "label", "Label", "class", "target"]

    found_text = next((c for c in text_candidates if c in df.columns), None)
    found_label = next((c for c in label_candidates if c in df.columns), None)

    if not found_text or not found_label:
        raise KeyError(
            f"Could not detect text or label columns. Available columns: {list(df.columns)}"
        )
    return found_text, found_label


def preprocess_text(text: str) -> str:
    """Normalize review text by lowercasing and removing URLs, HTML tags, and punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # Remove URLs
    text = re.sub(r"<.*?>", "", text)               # Remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)           # Keep English letters and spaces only
    text = re.sub(r"\s+", " ", text).strip()        # Collapse multiple spaces
    return text


def train_and_select_best_model(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """Train Logistic Regression and SVM models, compare weighted F1-scores, and select the best."""
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="weighted")
    logging.info(f"Logistic Regression -> Accuracy: {lr_acc:.4f} | F1-Score: {lr_f1:.4f}")

    svm_model = LinearSVC(random_state=RANDOM_STATE)
    svm_model.fit(X_train_tfidf, y_train)
    svm_preds = svm_model.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average="weighted")
    logging.info(f"SVM (LinearSVC)     -> Accuracy: {svm_acc:.4f} | F1-Score: {svm_f1:.4f}")

    if svm_f1 >= lr_f1:
        logging.info("Best Model Selected: SVM (LinearSVC)")
        return svm_model, lr_model, svm_model
    else:
        logging.info("Best Model Selected: Logistic Regression")
        return lr_model, lr_model, svm_model


def save_artifacts(tfidf, best_clf, lr_model, svm_model, X_test_tfidf, y_test, output_dir: str) -> None:
    """Serialize the best end-to-end pipeline, individual models, and test split for downstream use."""
    os.makedirs(output_dir, exist_ok=True)

    best_pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", best_clf)
    ])
    joblib.dump(best_pipeline, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(lr_model, os.path.join(output_dir, "lr_model.pkl"))
    joblib.dump(svm_model, os.path.join(output_dir, "svm_model.pkl"))
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump({"X_test": X_test_tfidf, "y_test": y_test}, os.path.join(output_dir, "test_data.pkl"))

    logging.info(f"All model artifacts saved successfully to '{output_dir}/'.")


def main() -> None:
    try:
        dataset_path = find_dataset_path()
        logging.info(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)

        text_col, label_col = detect_columns(df)
        logging.info(f"Detected text column: '{text_col}' | label column: '{label_col}'")

        df = df.dropna(subset=[text_col, label_col]).copy()
        df["processed_text"] = df[text_col].apply(preprocess_text)
        df = df[df["processed_text"].str.len() > 0]

        X = df["processed_text"]
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        logging.info(f"Data split complete: {len(X_train)} training samples, {len(X_test)} testing samples.")

        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        logging.info("TF-IDF vectorization completed.")

        best_clf, lr_model, svm_model = train_and_select_best_model(
            X_train_tfidf, y_train, X_test_tfidf, y_test
        )

        save_artifacts(tfidf, best_clf, lr_model, svm_model, X_test_tfidf, y_test, MODEL_DIR)

    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")


if __name__ == "__main__":
    main()

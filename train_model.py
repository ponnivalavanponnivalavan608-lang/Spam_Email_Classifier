import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)


# ==========================================
# STEP 2: REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()


# ==========================================
# STEP 3: CONVERT LABELS
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ==========================================
# STEP 4: SEPARATE INPUT AND OUTPUT
# ==========================================

X = df["message"]
y = df["label"]


# ==========================================
# STEP 5: TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# STEP 6: CREATE ML PIPELINE
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# ==========================================
# STEP 7: TRAIN MODEL
# ==========================================

print("Training model...")

model.fit(X_train, y_train)


# ==========================================
# STEP 8: MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# STEP 9: EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"]
))


# ==========================================
# STEP 10: SAVE MODEL
# ==========================================

joblib.dump(model, "model/spam_model.pkl")

print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")

print("\nLocation:")
print("model/spam_model.pkl")
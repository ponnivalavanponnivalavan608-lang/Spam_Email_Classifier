import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Original dataset:")
print(df.head())


# ==========================================
# STEP 2: REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nDataset after removing duplicates:")
print(df.shape)


# ==========================================
# STEP 3: CONVERT LABELS
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nLabels after conversion:")
print(df.head())


# ==========================================
# STEP 4: SEPARATE INPUT AND OUTPUT
# ==========================================

X = df["message"]
y = df["label"]

print("\nNumber of messages:", len(X))
print("Number of labels:", len(y))


# ==========================================
# STEP 5: SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# STEP 6: DISPLAY RESULTS
# ==========================================

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nData preparation completed successfully!")
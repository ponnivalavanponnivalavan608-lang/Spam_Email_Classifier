import pandas as pd

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print(df.head())

print("\nDataset size:")
print(df.shape)

print("\nMessage counts:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate messages:")
print(df.duplicated().sum())

df = df.drop_duplicates()

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nLabels after conversion:")
print(df.head())

print("\nDataset size after removing duplicates:")
print(df.shape)
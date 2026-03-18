import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/business_phishing_dataset.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns)

# Data types
print("\nInfo:")
print(df.info())

# Label distribution
print("\nLabel distribution:")
print(df["label"].value_counts())

# Attack types
print("\nAttack type distribution:")
print(df["attack_type"].value_counts())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Average email length
df["body_length"] = df["body"].apply(len)
print("\nAverage body length:", df["body_length"].mean())

# URL statistics
print("\nAverage number of links:", df["num_links"].mean())

# Urgent words ratio
print("\nUrgent emails ratio:", df["has_urgent_words"].mean())

# Phishing vs legit length
print("\nBody length by label:")
print(df.groupby("label")["body_length"].mean())


# Label distribution
df["label"].value_counts().plot(kind="bar")
plt.title("Phishing vs Legitimate Emails")
plt.xticks([0,1], ["Legitimate", "Phishing"])
plt.show()

# Attack types
df["attack_type"].value_counts().plot(kind="bar")
plt.title("Attack Type Distribution")
plt.xticks(rotation=45)
plt.show()
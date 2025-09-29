import pandas as pd
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X, y = [], []

# Check both data sources
folders = ["data", "uploads"]

for folder in folders:
    if not os.path.exists(folder):
        continue  # skip if folder not present

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        label = file[:-4]  # filename without .csv = gesture label
        df = pd.read_csv(f"{folder}/{file}", header=None)

        # Each row is one landmark set (21 points × 3 coords = 63 values)
        for row in df.values:
            if len(row) == 63:   # ✅ only full rows
                X.append(row)
                y.append(label)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save trained model
os.makedirs("model", exist_ok=True)
with open("model/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully with only valid rows.")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

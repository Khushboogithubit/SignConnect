import os
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = [], []

folder = "data"

if not os.path.exists(folder):
    raise Exception("❌ No 'data' folder found.")

for file in os.listdir(folder):
    if not file.endswith(".csv"):
        continue

    # ✅ label fix
    label = file.replace(".csv", "")

    with open(os.path.join(folder, file), "r") as f:
        reader = csv.reader(f)

        for row in reader:
            row = [float(x) for x in row]

            # pad if single hand
            if len(row) == 63:
                row.extend([0.0]*63)

            if len(row) == 126:
                X.append(row)
                y.append(label)

if len(X) < 10:
    raise Exception("❌ Not enough data. Collect more.")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Better model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Save
os.makedirs("model", exist_ok=True)
with open("model/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully")
print(f"Samples: {len(X)}")
print(f"🎯 Accuracy: {acc*100:.2f}%")
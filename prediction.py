# ================== Imports ==================
import json
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.datasets import load_breast_cancer

# ================== Load Dataset ==================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# ================== Train-Test Split ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ================== Feature Scaling ==================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== Train Model ==================
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train_scaled, y_train)

# ================== Predictions ==================
y_pred = model.predict(X_test_scaled)

# ================== Evaluation ==================
new_f1 = f1_score(y_test, y_pred)
print("New Model F1-score:", new_f1)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ================== Baseline Comparison ==================
with open("baseline_metrics.json", "r") as f:
    baseline = json.load(f)

old_f1 = baseline["f1_score"]
print("\nBaseline Model F1-score:", old_f1)

# ================== Conditional Model Promotion ==================
if new_f1 >= old_f1:
    print("✅ New model is better. Saving model.")

    with open("model_production.pkl", "wb") as f:
        pickle.dump(model, f)

    baseline["f1_score"] = float(new_f1)
    baseline["model_name"] = "random_forest_v2"

    with open("baseline_metrics.json", "w") as f:
        json.dump(baseline, f, indent=4)

else:
    print("❌ New model is worse. Model NOT saved.")

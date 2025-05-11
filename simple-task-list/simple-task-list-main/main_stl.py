import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import random

# ---------- CONFIGURATION ----------
DATA_FILE = "student_habits_performance.csv"  # Use your uploaded file
LABEL_COLUMN = "exam_score"  # Numerical, we'll categorize it for classification

# ---------- DATA LOADING ----------
if os.path.exists(DATA_FILE):
    data = pd.read_csv(DATA_FILE)
else:
    print(f"{DATA_FILE} not found.")
    exit()

# ---------- CATEGORIZE EXAM SCORE ----------
def categorize_score(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

data["performance"] = data[LABEL_COLUMN].apply(categorize_score)

# Drop unused or ID column
data = data.drop(columns=["student_id", LABEL_COLUMN])

# ---------- PREPROCESSING ----------
# Separate features and target
X = data.drop(columns=["performance"])
y = data["performance"]

# Encode categorical features
cat_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=cat_cols)

# Scale numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# ---------- MODEL TRAINING ----------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------- MAIN FUNCTIONS ----------
def add_entry(entry_dict, score):
    global data, model, scaler, X_encoded
    entry_df = pd.DataFrame([entry_dict])
    entry_df["performance"] = categorize_score(score)
    data = pd.concat([data, entry_df], ignore_index=True)

    # Retrain model
    X_new = data.drop(columns=["performance"])
    X_new_encoded = pd.get_dummies(X_new, columns=cat_cols)
    X_new_encoded = X_new_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    X_new_scaled = scaler.fit_transform(X_new_encoded)
    model.fit(X_new_scaled, data["performance"])

def recommend_entry(target_label):
    candidates = data[data["performance"] == target_label]
    if not candidates.empty:
        chosen = random.choice(candidates.to_dict(orient="records"))
        print(f"Recommended student habit profile (Performance: {target_label}):")
        for k, v in chosen.items():
            print(f"{k}: {v}")
    else:
        print(f"No entries found with performance label '{target_label}'.")

def list_entries():
    print(data)

def remove_entry(index):
    global data
    if index < 0 or index >= len(data):
        print("Invalid index.")
    else:
        data = data.drop(index=index).reset_index(drop=True)
        print("Entry removed.")

# ---------- MAIN LOOP ----------
while True:
    print("\nStudent Habit Classifier")
    print("1. Add Entry")
    print("2. Remove Entry")
    print("3. List Entries")
    print("4. Recommend by Performance Category")
    print("5. Exit")

    choice = input("Select an option: ").strip()

    if choice == "1":
        print("Enter student habit values:")
        entry = {}
        for col in X.columns:
            entry[col] = input(f"{col}: ")
        score = float(input("Enter exam_score: "))
        add_entry(entry, score)
        print("Entry added.")

    elif choice == "2":
        index = int(input("Enter index of entry to remove: "))
        remove_entry(index)

    elif choice == "3":
        list_entries()

    elif choice == "4":
        label = input("Enter performance category (High/Medium/Low): ")
        recommend_entry(label)

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Try again.")

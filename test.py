import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import sys # NEW: Import sys module

# --- NEW: Add this line at the very beginning ---
print(f"Executing script from: {os.path.abspath(__file__)}")

print("Hi, Michael! I am training the model now.")

try:
    df = pd.read_csv("Extracted_MALDI_Features.csv")
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nColumns in the dataset:", df.columns.tolist())
except FileNotFoundError:
    print("Error: 'Extracted_MALDI_Features.csv' not found. Please ensure it's in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

feature_columns = ['marker_2593', 'marker_2563', 'marker_2503', 'marker_2042']

missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print(f"Error: Missing feature columns in the dataset: {missing_features}")
    print("Please check your 'extract_features.py' script and the generated CSV.")
    exit()

X = df[feature_columns]
y = df['Species_Label']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")
print("Unique species labels:", y.unique())

if len(df) < 2:
    print("\nWarning: Not enough samples to perform a proper train/test split. Skipping split and training on all available data.")
    X_train, y_train = X, y
    X_test, y_test = X, y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split: {len(X_train)} samples for training, {len(X_test)} samples for testing.")


print("\nTraining the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test)

print("\nEvaluating model performance:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("DEBUG: Just finished Classification Report.")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model_filename = "hair_species_classifier.pkl"
print(f"\nAttempting to save the trained model to '{model_filename}'...")
try:
    joblib.dump(model, model_filename)
    print(f"✅ Trained model successfully saved to '{model_filename}'")
except Exception as e:
    print(f"\n❌ ERROR: Failed to save the model to '{model_filename}'.")
    print(f"Reason: {e}")
    print("Please check file permissions for the directory:")
    print(f"  {os.getcwd()}")
    print("And ensure there is enough disk space.")

print("\nModel training and evaluation process finished, Michael.")

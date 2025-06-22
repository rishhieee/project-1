import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load real dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
joblib.dump(label_encoder, 'label_encoder.pkl')

# Features and target
X = df.drop('label', axis=1)
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy: {acc:.4f}")

# Save model
joblib.dump(model, 'crop_model.pkl')
print("✅ Model saved.")

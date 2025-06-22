import pandas as pd

df = pd.read_csv("Crop_recommendation.csv")
print("📊 Columns:", df.columns.tolist())
print("\n🔍 Sample Data:\n", df.head())
print("\n📋 Unique Crops:\n", df['label'].value_counts())

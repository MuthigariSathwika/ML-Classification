"""
Generate complete test data CSV file from the Breast Cancer Wisconsin dataset
using the same train-test split as app.py (80/20 with random_state=42)
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split using same parameters as app.py
X = df.drop('target', axis=1)
y = df['target']

_, X_test, _, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Combine X_test and y_test
test_df = X_test.copy()
test_df['target'] = y_test

# Save to CSV
test_df.to_csv('test_data.csv', index=False)
print(f"✓ Created test_data.csv with {len(test_df)} rows")
print(f"  Columns: {list(test_df.columns)}")
print(f"  Shape: {test_df.shape}")
print(f"  Target distribution: {test_df['target'].value_counts().to_dict()}")

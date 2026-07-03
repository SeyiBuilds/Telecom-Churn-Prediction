# ============================================================================
# TELECOM CUSTOMER CHURN PREDICTION - COMPLETE PROJECT
# Google Colab Ready - Copy & Paste Each Section Into Colab Cells
# ============================================================================

# ============================================================================
# SECTION 1: IMPORT LIBRARIES (Run this first)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# ============================================================================
# SECTION 2: LOAD THE DATA (Your CSV file)
# ============================================================================

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Display basic info
print("Dataset Shape:", df.shape)  # How many rows & columns
print("\nFirst few rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================================
# SECTION 3: EXPLORE THE DATA (Understand what we're working with)
# ============================================================================

# Check the target variable (what we're predicting)
print("\n" + "="*50)
print("CHURN DISTRIBUTION (What we're predicting)")
print("="*50)
print(df['Churn'].value_counts())
print("\nChurn Percentage:")
print(df['Churn'].value_counts(normalize=True) * 100)

# Visualize churn distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
df['Churn'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Churn Distribution (Count)')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['Churn'].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red'])
plt.title('Churn Distribution (Percentage)')
plt.ylabel('')

plt.tight_layout()
plt.show()

# Statistical summary
print("\n" + "="*50)
print("NUMERICAL FEATURES SUMMARY")
print("="*50)
print(df.describe())


# ============================================================================
# SECTION 4: DATA PREPROCESSING (Clean & Prepare the data)
# ============================================================================

print("\n" + "="*50)
print("CLEANING & PREPARING DATA")
print("="*50)

# Step 1: Remove CustomerID (not useful for prediction)
df_clean = df.drop(['customerID'], axis=1)

# Step 2: Convert 'Churn' to 0s and 1s (0 = No, 1 = Yes)
df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})

# Step 3: Handle 'TotalCharges' - Convert to numeric (some might be spaces)
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Step 4: Fill any missing TotalCharges with 0 (usually means new customers)
df_clean['TotalCharges'].fillna(0, inplace=True)

print("Data cleaned!")
print(f"Final dataset shape: {df_clean.shape}")


# ============================================================================
# SECTION 5: FEATURE ENGINEERING (Convert text to numbers)
# ============================================================================

print("\n" + "="*50)
print("CONVERTING TEXT TO NUMBERS")
print("="*50)

# Create a copy for processing
df_encoded = df_clean.copy()

# Binary columns - convert Yes/No to 1/0
binary_cols = ['PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})

# Other categorical columns - one-hot encoding (create dummy variables)
categorical_cols = [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
print(f"\nCategorical columns to convert: {categorical_cols}")

df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

print(f"\nAll data converted to numbers!")
print(f"Final features shape: {df_encoded.shape}")
print(f"Total features: {df_encoded.shape[1] - 1}")  # -1 because we exclude target


# ============================================================================
# SECTION 6: PREPARE FOR MODELING (Split features and target)
# ============================================================================

print("\n" + "="*50)
print("PREPARING FOR MODELING")
print("="*50)

# Separate features (X) and target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")
print(f"Number of features: {X_train.shape[1]}")


# ============================================================================
# SECTION 7: BUILD MODEL 1 - LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

# Create and train the model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Calculate performance metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print(f"\nAccuracy:  {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"Precision: {lr_precision:.4f} ({lr_precision*100:.2f}%)")
print(f"Recall:    {lr_recall:.4f} ({lr_recall*100:.2f}%)")
print(f"F1-Score:  {lr_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn']))


# ============================================================================
# SECTION 8: BUILD MODEL 2 - RANDOM FOREST
# ============================================================================

print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST")
print("="*50)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Calculate performance metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"\nAccuracy:  {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"Precision: {rf_precision:.4f} ({rf_precision*100:.2f}%)")
print(f"Recall:    {rf_recall:.4f} ({rf_recall*100:.2f}%)")
print(f"F1-Score:  {rf_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No Churn', 'Churn']))


# ============================================================================
# SECTION 9: COMPARE MODELS
# ============================================================================

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison = pd.DataFrame({
    'Logistic Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1],
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

print(comparison)

# Visualize comparison
comparison.T.plot(kind='bar', figsize=(12, 5))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================================
# SECTION 10: CONFUSION MATRIX (Visual representation of predictions)
# ============================================================================

print("\n" + "="*50)
print("CONFUSION MATRICES")
print("="*50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
axes[0].set_xticklabels(['No Churn', 'Churn'])
axes[0].set_yticklabels(['No Churn', 'Churn'])

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Random Forest')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
axes[1].set_xticklabels(['No Churn', 'Churn'])
axes[1].set_yticklabels(['No Churn', 'Churn'])

plt.tight_layout()
plt.show()


# ============================================================================
# SECTION 11: FEATURE IMPORTANCE (Which factors matter most?)
# ============================================================================

print("\n" + "="*50)
print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
print("="*50)

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# Visualize top 10 features
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score')
plt.title('Top 10 Most Important Features for Churn Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================================
# SECTION 12: FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*50)
print("FINAL PROJECT SUMMARY")
print("="*50)

best_model = 'Random Forest' if rf_accuracy > lr_accuracy else 'Logistic Regression'
best_accuracy = max(rf_accuracy, lr_accuracy)

print(f"\nBEST MODEL: {best_model}")
print(f"ACCURACY: {best_accuracy*100:.2f}%")
print(f"This means the model correctly predicts churn {best_accuracy*100:.2f}% of the time")

print(f"\nKEY FINDINGS:")
print(f"   - Total customers analyzed: {len(df)}")
print(f"   - Churn rate in dataset: {(y.sum()/len(y))*100:.2f}%")
print(f"   - Top reason for churn: {feature_importance.iloc[0]['Feature']}")
print(f"   - Model tested on: {len(X_test)} customers")

print(f"\nWHAT THIS MEANS:")
print(f"   - The company can use this model to identify at-risk customers")
print(f"   - Focus retention efforts on customers with highest churn probability")
print(f"   - Monitor the top features identified for early warning signs")

print("\n" + "="*50)
print("PROJECT COMPLETE!")
print("="*50)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    accuracy_score
)

# --- 1. Data Loading ---
file_path = 'transaction_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    exit()

print("--- Initial Data Head ---")
print(df.head())
print("\n--- Initial Data Description ---")
print(df.describe())

# --- 2. Initial Cleaning & Preprocessing ---
print("\n--- Starting Preprocessing ---")

# Drop irrelevant index-like columns
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
if 'Index' in df.columns:
    df = df.drop(columns=['Index'])

# Strip whitespace from column names
df.columns = df.columns.str.strip()
print("\nColumn names stripped.")

# Separate features (X) and target (y)
if 'FLAG' not in df.columns:
    print("Error: Target column 'FLAG' not found in the dataset.")
    exit()
X = df.drop(columns=['FLAG'])
y = df['FLAG']

# Drop 'Address' column and categorical ERC20 token type columns
columns_to_drop_explicitly = ['Address', 'ERC20 most sent token type', 'ERC20_most_rec_token_type']
for col in columns_to_drop_explicitly:
    if col in X.columns:
        X = X.drop(columns=[col])
        print(f"Dropped column: {col}")

# Impute remaining missing values in features with 0
X.fillna(0, inplace=True)
print("\nMissing values in features imputed with 0.")

# Identify and drop zero-variance columns (after imputation)
# These were identified as problematic from EDA (e.g., ERC20 avg time columns, ERC20 contract value columns)
# A more robust way is to check variance:
cols_before_variance_drop = X.shape[1]
variances = X.var()
zero_variance_cols = variances[variances == 0].index.tolist()

if zero_variance_cols:
    X = X.drop(columns=zero_variance_cols)
    print(f"\nDropped {len(zero_variance_cols)} zero-variance columns: {zero_variance_cols}")
    print(f"Number of features remaining: {X.shape[1]}")
else:
    print("\nNo zero-variance columns found to drop.")

# --- 3. Feature Scaling ---
# Select only numeric columns for scaling (should be all at this point if object columns were handled)
numeric_cols = X.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print("\nNumerical features scaled using MinMaxScaler.")
print("\n--- Preprocessed Features Head ---")
print(X.head())

# --- 4. Train-Test Split ---
# Stratify by y to ensure proportional representation of classes
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split into training and testing sets.")
    print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Fraud cases in training set: {sum(y_train)}")
    print(f"Fraud cases in testing set: {sum(y_test)}")
except ValueError as e:
    print(f"Error during train-test split. This might happen if features are not purely numeric: {e}")
    print("Ensure all features in X are numeric before splitting.")
    exit()


# --- 5. Model Training and Evaluation ---
print("\n--- Starting Model Training and Evaluation ---")

# Define models
# For imbalanced datasets, class_weight='balanced' or scale_pos_weight can be helpful.
# scale_pos_weight = count(negative instances) / count(positive instances)
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight_val = neg_count / pos_count

models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_val, random_state=42)
}

results = {}
trained_models = {}

plt.figure(figsize=(10, 8)) # For ROC curves

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision (Fraud)": precision,
        "Recall (Fraud)": recall,
        "F1-Score (Fraud)": f1,
        "AUC-ROC": auc_roc
    }
    
    print(f"Results for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (Fraud): {precision:.4f}")
    print(f"  Recall (Fraud): {recall:.4f}")
    print(f"  F1-Score (Fraud): {f1:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud (0)', 'Fraud (1)']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(6,4)) # New figure for each CM
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show() # Show CM plot immediately
    
    # For collective ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.2f})')

# Finalize collective ROC curve plot
plt.plot([0, 1], [0, 1], 'k--') # Diagonal dashed line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("\n--- Model Performance Summary ---")
results_df = pd.DataFrame(results).T
print(results_df)


# --- 6. Feature Importance ---
print("\n--- Feature Importance Analysis ---")

# Random Forest
if "Random Forest" in trained_models:
    rf_model = trained_models["Random Forest"]
    importances_rf = rf_model.feature_importances_
    feature_names = X_train.columns
    forest_importances = pd.Series(importances_rf, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=forest_importances.head(20), y=forest_importances.head(20).index) # Display top 20
    plt.title('Top 20 Feature Importances - Random Forest')
    plt.xlabel('Mean decrease in impurity')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    print("\nTop 10 Features (Random Forest):")
    print(forest_importances.head(10))

# XGBoost
if "XGBoost" in trained_models:
    xgb_model = trained_models["XGBoost"]
    importances_xgb = xgb_model.feature_importances_
    xgb_importances = pd.Series(importances_xgb, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=xgb_importances.head(20), y=xgb_importances.head(20).index) # Display top 20
    plt.title('Top 20 Feature Importances - XGBoost')
    plt.xlabel('Feature Importance Score (e.g., gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    print("\nTop 10 Features (XGBoost):")
    print(xgb_importances.head(10))

print("\n--- Analysis Complete ---")
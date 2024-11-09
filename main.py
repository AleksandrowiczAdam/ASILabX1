import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make a directory for charts
os.makedirs("charts", exist_ok=True)

# Load the dataset
df = pd.read_csv('CollegeDistance.csv')

df = df.drop(columns=['rownames'])

df = df.round(2)

# Group the 'score' column into 3 categories (0, 1, 2) based on percentiles
percentiles = [0, 0.33, 0.66, 1]
df['score_category'] = pd.qcut(df['score'], q=percentiles, labels=[0, 1, 2])

# Drop the original 'score' column as we will use 'score_category' for prediction
df = df.drop(columns=['score'])

# Handle categorical columns: We will one-hot encode them
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define the target variable (score_category) and features (X)
X = df_encoded.drop(columns=['score_category'])  # Features
y = df_encoded['score_category']  # Target variable

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join("charts", "confusion_matrix.png"))

# Calculate AUC ROC (For multi-class classification, we need to use One-vs-Rest approach)
y_prob = rf_clf.predict_proba(X_test)
auc_roc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f"AUC ROC: {auc_roc:.4f}")

# Plot ROC curve
plt.figure(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)  # We will plot for class 1
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join("charts", "roc.png"))

# Print feature importances
feature_importances = rf_clf.feature_importances_
feature_names = X.columns

# Print and plot the feature importances
print("Feature Importances:")
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importances - Random Forest')
plt.savefig(os.path.join("charts", "feature_importance.png"))

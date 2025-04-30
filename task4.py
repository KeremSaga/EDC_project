import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Load the 30s data
data = pd.read_csv("data/GenreClassData_30s.txt", sep="\t")

# Split into train/test
train_data = data[data['Type'] == 'Train']
test_data = data[data['Type'] == 'Test']

# Features (remove non-features)
feature_cols = [col for col in data.columns if col not in ['Track ID', 'File', 'GenreID', 'Genre', 'Type']]

X_train = train_data[feature_cols]
y_train = train_data['Genre']

X_test = test_data[feature_cols]
y_test = test_data['Genre']

# Scale the features (Random Forest technically doesn't need scaling, but it doesn't hurt)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=200,    # number of trees
    random_state=25,     # for reproducibility
    n_jobs=-1,            # use all CPU cores
    # tuning for redusing overfitting while keeping good accuracy
    max_depth=7,
    max_features=10,
    min_samples_leaf=10
)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
y_train_pred = rf.predict(X_train_scaled)
acc_train = accuracy_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))
print("Training Accuracy: {:.2f}%".format(acc_train * 100))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Random Forest Confusion Matrix for 30s segments")
plt.grid(False)
plt.tight_layout()

# Feature importance
importances = pd.Series(rf.feature_importances_*100, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nTop 5 most important features:")
print(importances.head(5))
print(f"\nSum of feature importances: {importances.sum()}")

# Print used features
used_features = importances[importances > 0].index.tolist()
print(f"Number of features used: {len(used_features)}")
# print("Used features:")
# print(used_features)

end_time = time.time()
print("\nTotal time to run task 4 30s: {:.2f} seconds".format(end_time - start_time))

plt.show()
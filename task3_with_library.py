import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from itertools import combinations
import time
import numpy as np

start_time = time.time()

# Load the data
data = pd.read_csv("data/GenreClassData_30s.txt", sep="\t")

# Split the dataset
train_data = data[data['Type'] == 'Train']
test_data = data[data['Type'] == 'Test']

# Select relevant features
features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo'
    ]

all_features = list(data.columns)

for not_feature_col in ['Track ID', 'File' , 'GenreID', 'Genre', 'Type']:
    all_features.remove(not_feature_col)

accuracies = []

## Try the four features

X_train = train_data[features]
y_train = train_data['Genre']

X_test = test_data[features]
y_test = test_data['Genre']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate the classifier
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
accuracies.append(accuracy)

# Current best option
best_accuracy = accuracy
best_features = features

## Try all other combinations

for feature_combos in combinations(features, 3):
    remaining_features = [f for f in all_features if f not in features]
      
    for extra_feature in remaining_features:
        current_features = list(feature_combos) + [extra_feature]

        X_train = train_data[current_features]
        y_train = train_data['Genre']

        X_test = test_data[current_features]
        y_test = test_data['Genre']

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train the k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)

        # Evaluate the classifier
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Update best option
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = current_features
            # Training accuracy to check for overfitting (for reflection, can't do much about it since overfitting is caused by k, which is fixed to k=5 in this task)
            y_train_pred = knn.predict(X_train_scaled)
            training_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\nBest Feature Set: {best_features}")
print("\nBest test accuracy: {:.2f}%".format(best_accuracy * 100))

print("Best sets training accuracy: {:.2f}%".format(training_accuracy * 100))

# Plot the accuracies and mark the best one
plt.figure()
plt.plot(range(len(accuracies)), accuracies, label='Accuracies')
best_index = np.argmax(accuracies)
plt.scatter(best_index, accuracies[best_index], color='red', label='Best Accuracy')
plt.xlabel("All possible set indices")
plt.ylabel("Accuracies")
plt.title("Accuracies of all feature set combinations")
plt.legend()
plt.grid()

## Generate confusion matrix for the best features

X_train = train_data[best_features]
y_train = train_data['Genre']

X_test = test_data[best_features]
y_test = test_data['Genre']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Evaluate the classifier
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix for k-NN Genre Classifier (k=5)")
plt.grid(False)
plt.tight_layout()

end_time = time.time()
print("\nRuntime task 3 w/ library: {:.2f} seconds\n".format(end_time - start_time))

plt.show()
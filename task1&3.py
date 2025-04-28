import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from itertools import combinations

def load_data(filepath, features):
    # Load dataset from file and split into training and testing sets  
    data = pd.read_csv(filepath, sep="\t")
    train_data = data[data['Type'] == 'Train']
    test_data = data[data['Type'] == 'Test']
    return data, train_data, test_data

def scale_features(X_train, X_test):
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def knn_classifier(X_train, y_train, X_test, k=5):
    predictions = []
    for x in X_test:
        # Calculate Euclidean distance from x to all training points
        distances = np.linalg.norm(X_train - x, axis=1)
        
        # Find the indices of the k nearest neighbors
        nearest_indices = distances.argsort()[:k]
        
        # Get the labels of the nearest neighbors
        nearest_labels = y_train[nearest_indices]
        
        # Majority vote: pick the most common label
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    
    return np.array(predictions)

def evaluate_model(y_true, y_pred, labels, k):
    # Evaluate the model by calculating accuracy and displaying the confusion matrix
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix for k-NN Classifier (k={k})")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def find_best_feature_set(data, train_data, test_data, base_features, k=5):
    accuracies = []
    all_features = list(data.columns)

    # Remove non-feature columns
    for not_feature_col in ['Track ID', 'File', 'GenreID', 'Genre', 'Type']:
        all_features.remove(not_feature_col)

    # Start with initial four features
    X_train = train_data[base_features]
    y_train = train_data['Genre'].values
    X_test = test_data[base_features]
    y_test = test_data['Genre'].values

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    y_pred = knn_classifier(X_train_scaled, y_train, X_test_scaled, k=k)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    best_accuracy = accuracy
    best_features = base_features.copy()

    # Try all combinations: 3 features from base, 1 new feature
    for feature_combos in combinations(base_features, 3):
        remaining_features = [f for f in all_features if f not in base_features]
        for extra_feature in remaining_features:
            current_features = list(feature_combos) + [extra_feature]

            X_train = train_data[current_features]
            X_test = test_data[current_features]

            X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
            y_pred = knn_classifier(X_train_scaled, y_train, X_test_scaled, k=k)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = current_features

    return accuracies, best_accuracy, best_features

def plot_accuracies(accuracies):
    # Plot accuracies of all feature combinations
    best_index = np.argmax(accuracies)
    plt.figure()
    plt.plot(range(len(accuracies)), accuracies, label='Accuracies')
    plt.scatter(best_index, accuracies[best_index], color='red', label='Best Accuracy')
    plt.xlabel("Feature set index")
    plt.ylabel("Accuracy")
    plt.title("Accuracies of Feature Set Combinations")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    filepath = "data/GenreClassData_30s.txt"
    features = [
        'spectral_rolloff_mean',
        'mfcc_1_mean',
        'spectral_centroid_mean',
        'tempo'
    ]
    k = 5

    ### TASK 1 ###
    print("Task 1:")
    data, train_data, test_data = load_data(filepath, features)

    X_train = train_data[features].values
    y_train = train_data['Genre'].values
    X_test = test_data[features].values
    y_test = test_data['Genre'].values

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    y_pred = knn_classifier(X_train_scaled, y_train, X_test_scaled, k=k)
    evaluate_model(y_test, y_pred, labels=np.unique(y_train), k=k)

    ### TASK 3 ###
    print("\nTask 3:")
    accuracies, best_accuracy, best_features = find_best_feature_set(data, train_data, test_data, features, k=k)

    print("Best Accuracy: {:.2f}%".format(best_accuracy * 100))
    print("Best Feature Set:", best_features)

    # Plot accuracy for all feature combinations
    plot_accuracies(accuracies)

    # Train and evaluate model with the best feature set
    X_train = train_data[best_features]
    y_train = train_data['Genre'].values
    X_test = test_data[best_features]
    y_test = test_data['Genre'].values

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    y_pred = knn_classifier(X_train_scaled, y_train, X_test_scaled, k=k)

    evaluate_model(y_test, y_pred, labels=np.unique(y_train), k=k)

if __name__ == "__main__":
    main()

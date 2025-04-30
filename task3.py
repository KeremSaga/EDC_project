from task1 import load_data, scale_features, knn_classifier, evaluate_model
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time

start_time = time.time()

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
    data, train_data, test_data = load_data(filepath, features)
    accuracies, best_accuracy, best_features = find_best_feature_set(data, train_data, test_data, features, k=k)

    print("\nBest Feature Set:", best_features)
    plot_accuracies(accuracies)

    # Extract features from dataset
    X_train = train_data[best_features].values
    y_train = train_data['Genre'].values
    X_test = test_data[best_features].values
    y_test = test_data['Genre'].values

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    y_pred = knn_classifier(X_train_scaled, y_train, X_test_scaled, k=k)
    y_train_pred = knn_classifier(X_train_scaled, y_train, X_train_scaled, k=k)

    end_time = time.time()
    print("\nRuntime: {:.2f} seconds".format(end_time - start_time))
    
    evaluate_model(y_test, y_pred, y_train, y_train_pred, labels=np.unique(y_train), k=k)

if __name__ == "__main__":
    main()

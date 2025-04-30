import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import time

start_time = time.time()

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
    for i, x in enumerate(X_test):
        # Calculate Euclidean distance from x to all training points
        distances = np.linalg.norm(X_train - x, axis=1)
        
        # If predicting on training set itself, ignore the same point
        if np.array_equal(X_train, X_test):
            distances[i] = np.inf
        
        # Find the indices of the k nearest neighbors
        nearest_indices = distances.argsort()[:k]

        # Get the labels of the nearest neighbors
        nearest_labels = y_train[nearest_indices]

        # Majority vote: pick the most common label
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    
    return np.array(predictions)

def evaluate_model(y_true, y_pred, y_train, y_train_pred, labels, k):
    # Evaluate the model by calculating accuracy and displaying the confusion matrix
    accuracy = accuracy_score(y_true, y_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix for k-NN Classifier (k={k})")
    plt.grid(False)
    plt.tight_layout()
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

    # Extract features from dataset
    X_train = train_data[features].values
    y_train = train_data['Genre'].values
    X_test = test_data[features].values
    y_test = test_data['Genre'].values

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    y_pred = knn_classifier(X_train_scaled, y_train, X_test_scaled, k=k)
    y_train_pred = knn_classifier(X_train_scaled, y_train, X_train_scaled, k=k)

    end_time = time.time()
    print("\nRuntime: {:.2f} seconds".format(end_time - start_time))
    
    evaluate_model(y_test, y_pred, y_train, y_train_pred, labels=np.unique(y_train), k=k)

if __name__ == "__main__":
    main()

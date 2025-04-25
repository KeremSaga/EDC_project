import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
data = pd.read_csv("GenreClassData_30s.txt", sep="\t")

# Select relevant features
features = [
    'spectral_rolloff_mean',
    'mfcc_1_mean',
    'spectral_centroid_mean',
    'tempo'
    ]

# Split the dataset
train_data = data[data['Type'] == 'Train']
test_data = data[data['Type'] == 'Test']

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
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)

print("Accuracy: {:.2f}%".format(accuracy * 100))
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix for k-NN Genre Classifier (k=5)")
plt.grid(False)
plt.tight_layout()
plt.show()

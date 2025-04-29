import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np

start_time = time.time()

# Load the data
data = pd.read_csv("data/GenreClassData_30s.txt", sep="\t")

# Split the dataset
train_data = data[data['Type'] == 'Train']
test_data = data[data['Type'] == 'Test']

# 
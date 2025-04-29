import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

start_time = time.time()

# Load the data
data = pd.read_csv('data/GenreClassData_30s.txt', sep='\t')

# Filter data for the four genres
selected_genres = ['pop', 'disco', 'metal', 'classical']
data_filtered = data[data['Genre'].isin(selected_genres)]

# Define features and colors for each genre
features = ['spectral_rolloff_mean', 'mfcc_1_mean', 
           'spectral_centroid_mean', 'tempo']
colors = {'pop': 'blue', 'disco': 'green', 
         'metal': 'red', 'classical': 'purple'}

first_plot = True

# Create histogram plots for each feature
for feature in features:
    plt.figure(figsize=(10, 6))
    
    # Plot histograms for each genre
    for genre in selected_genres:
        subset = data_filtered[data_filtered['Genre'] == genre]
        sns.histplot(subset[feature], 
                    color=colors[genre],
                    label=genre.capitalize(),
                    alpha=0.4,
                    bins=20)
    
    plt.title(f'{feature} distribution', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Genre')
    plt.grid(alpha=0.3)
    if first_plot:
        end_time = time.time()
        print("\nTotal time to run task 2: {:.2f} seconds".format(end_time - start_time))
        first_plot = False
    plt.show()
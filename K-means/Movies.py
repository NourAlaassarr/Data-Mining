import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import pandas as pd
from itertools import combinations
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(filename, percentage):
    df = pd.read_csv(filename)
    num_rows = int(len(df) * (percentage / 100))  
    df = df.head(num_rows)
    return df



# as K-means, are sensitive to outliers because they optimize cluster centers based on distances between data points. 
# Outliers can significantly affect the centroids and hence the resulting clusters. In such cases, it's often recommended to handle outliers before clustering.


def Pre_Processing(movie_data):
    
    # Select only the 'Movie Name' and 'IMDB Rating' columns
    Movies_DataFrame = movie_data[['Movie Name', 'IMDB Rating']].copy()
    
    # Check for missing values in IMDB Rating
    Movies_DataFrame.dropna(subset=['IMDB Rating'], inplace=True)
    return Movies_DataFrame
    

def Pre_Processing_Outliers(Movies_DataFrame):
    # Handle outliers in IMDB Rating
    if not Movies_DataFrame.empty:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = np.percentile(Movies_DataFrame['IMDB Rating'], 25)
        Q3 = np.percentile(Movies_DataFrame['IMDB Rating'], 75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = Movies_DataFrame[(Movies_DataFrame['IMDB Rating'] < lower_bound) | (Movies_DataFrame['IMDB Rating'] > upper_bound)]
        return outliers
    else:
        return pd.DataFrame()



def RandomCentroidsIntialization(data, k):
    # Randomly select k data points as initial centroids
    centroidsRand = np.random.choice(len(data), size=k, replace=False)
    centroids = data.iloc[centroidsRand]
    return centroids

def euclidean_distance(point1, point2):
    # Ensure both points have the same length
    point1_length = len(point1)
    point2_length = len(point2)
    assert point1_length == point2_length, "Points must have the same dimensionality"
    
    # Calculate the squared differences for each dimension
    squared_diffs = []
    for i in range(point1_length):
        squared_diff = (point1[i] - point2[i]) ** 2
        squared_diffs.append(squared_diff)
    
    # Sum up the squared differences
    sum_of_squares = 0
    for diff in squared_diffs:
        sum_of_squares += diff
    
    # Take the square root of the sum
    distance = sum_of_squares ** 0.5
    
    return distance

# def euclidean_distance(point1, point2):
#     # Ensure both points have the same length
#     assert len(point1) == len(point2), "Points must have the same dimensionality"
    
#     # Calculate the squared differences for each dimension
#     squared_diffs = [(x - y) ** 2 for x, y in zip(point1, point2)]
    
#     # Sum up the squared differences
#     sum_of_squares = 0
#     for diff in squared_diffs:
#         sum_of_squares += diff
    
#     # Take the square root of the sum
#     distance = sum_of_squares ** 0.5
    
#     return distance



def argmin(distances):
    min_distance = float('inf')
    min_index = None
    for i, distance in enumerate(distances):
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index

# def assign_points_to_clusters(data, centroids):
#     clusters = [[] for _ in range(len(centroids))]
#     for index, row in data.iterrows():
#         movie_name = row['Movie Name']
#         imdb_rating = row['IMDB Rating']
        
#         # Calculate distances
#         distances = [euclidean_distance([imdb_rating], [centroid]) for centroid in centroids]
        
#         # Find the index of the centroid with the minimum distance
#         cluster_index = argmin(distances)
        
#         # Assign the data point to the corresponding cluster
#         clusters[cluster_index].append((movie_name, imdb_rating))
#     return clusters
def assign_points_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    # iterates over each row (movie) in the data
    for index, row in data.iterrows():
        movie_name = row['Movie Name']
        imdb_rating = row['IMDB Rating']
        
        distances = []
        for centroid in centroids:
            
            distance = euclidean_distance([imdb_rating], [centroid])
            distances.append(distance)
        cluster_index = argmin(distances)
        clusters[cluster_index].append((movie_name, imdb_rating))
    return clusters





# def update_centroids(clusters):
#     centroids = []
#     for cluster in clusters:
#         if cluster:
#             ratings = [rating for _, rating in cluster]
#             centroids.append(np.mean(ratings))
#         else:
#             # If a cluster is empty, keep its centroid unchanged
#             centroids.append(np.nan)
#     return np.array(centroids)



def calculate_mean(values):
    if not values:
        return None  # Return None if the list is empty
    return sum(values) / len(values)

def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            ratings = [rating for _, rating in cluster]# Extract IMDb ratings from movies in the cluster
            cluster_mean = calculate_mean(ratings)
            centroids.append(cluster_mean)
        else:
            centroids.append(None)  # Keep centroid unchanged if the cluster is empty
    return centroids


def clusters_equal(clusters1, clusters2):
    # Check if the cluster assignments are the same between two sets of clusters
    if len(clusters1) != len(clusters2):
        return False
    
    for i in range(len(clusters1)):
        cluster1_set = set(clusters1[i])
        cluster2_set = set(clusters2[i])
        
        if len(cluster1_set.symmetric_difference(cluster2_set)) != 0:
            return False
    
    return True

def kmeans(data, k):
    # Initialize centroids
    centroids = RandomCentroidsIntialization(data['IMDB Rating'], k)
    clusters = [[] for _ in range(k)]

    # Keep track of previous clusters
    prev_clusters = None

    while not prev_clusters or not clusters_equal(prev_clusters, clusters):
        # Assign data points to clusters
        prev_clusters = clusters
        clusters = assign_points_to_clusters(data, centroids)

        # Update centroids
        centroids = update_centroids(clusters)

    return clusters, centroids


# def kmeans(data, k):
#     # Initialize centroids
#     centroids = RandomCentroidsIntialization(data['IMDB Rating'], k)
#     clusters = [[] for _ in range(k)]

#     # Keep track of whether centroids have changed
#     centroids_changed = True

#     while centroids_changed:
#         # Assign data points to clusters
#         clusters = assign_points_to_clusters(data, centroids)

#         # Update centroids
#         new_centroids = update_centroids(clusters)

#         # Check if centroids have changed
#         centroids_changed = not np.array_equal(centroids, new_centroids)

#         centroids = new_centroids

#     return clusters, centroids


def format_clusters(clusters, centroids, outliers):
    formatted_output = ""
    cluster_movie_counts = []  # List to store the count of movies in each cluster
    
    for i, centroid in enumerate(centroids):
        formatted_output += f"Cluster {i+1} centroid: {centroid:.2f}\n"
        formatted_output += f"Cluster {i+1} contents:\n"
        
        cluster_movies = clusters[i]  # Get movies in the current cluster
        cluster_movie_counts.append(len(cluster_movies))  # Store count of movies in the current cluster
        
        for movie_name, imdb_rating in cluster_movies:
            formatted_output += f"- {movie_name}: {imdb_rating}\n"
        formatted_output += "\n"
    
    # Calculate total number of outliers
    total_outliers = len(outliers)
    formatted_output += f"Total number of outliers: {total_outliers}\n"
    
    # Calculate total number of movies in each cluster
    for i, count in enumerate(cluster_movie_counts):
        formatted_output += f"Total number of movies in Cluster {i+1}: {count}\n"
    
    return formatted_output



import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import pandas as pd
import numpy as np

# Other functions and imports remain the same

def browse_file():
    filename = filedialog.askopenfilename(title="Select CSV file")
    if filename:
        # Ask user for the percentage of data to be used
        percentage = float(percentage_entry.get())
        
        movies_data = load_data(filename, percentage)
        preprocessed_data = Pre_Processing(movies_data)
        outliers = Pre_Processing_Outliers(preprocessed_data)
        data_without_outliers = preprocessed_data[~preprocessed_data.index.isin(outliers.index)]
        clusters_number = int(entry_clusters.get())
        randcentroids = RandomCentroidsIntialization(data_without_outliers, clusters_number)
        
        # Update the label with the randomly initialized centroids
        centroids_label.config(text=f"Randomly Initialized Centroids:\n{randcentroids.to_string(index=False)}")
        
        clusters, centroids = kmeans(data_without_outliers, clusters_number)
        formatted_output = format_clusters(clusters, centroids, outliers)
        
        # Append outliers to the formatted output
        formatted_output += "\n\nOutliers:\n"
        formatted_output += outliers.to_string(index=False)
        
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, formatted_output)
        text_output.config(state=tk.DISABLED)

# Create main application window
root = tk.Tk()
root.title("Movie Clustering")
root.geometry("800x600")

# Update background color
background_color = "#4A90E2"  # Blue
root.config(bg=background_color)

# Update font style and size
font_style = ('Helvetica', 12)

# Create widgets
label_clusters = ttk.Label(root, text="Enter the number of clusters (k):", font=font_style)
label_clusters.pack(pady=10)

entry_clusters = ttk.Entry(root)
entry_clusters.pack(pady=5)

label_percentage = ttk.Label(root, text="Enter the percentage of data to use:", font=font_style)
label_percentage.pack(pady=10)

percentage_entry = ttk.Entry(root)
percentage_entry.pack(pady=5)

browse_button = ttk.Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=10)

# Add a label to display the randomly initialized centroids
centroids_label = ttk.Label(root, text="", font=font_style)
centroids_label.pack(pady=10)

# Add scrollbar to the text area
scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL)
text_output = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED, yscrollcommand=scrollbar.set, font=font_style)
scrollbar.config(command=text_output.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_output.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()





# filename = 'imdb_top_2000_movies.csv'  
# percentage = 3
# movies_data = load_data(filename, percentage)

# # Print the loaded data
# print("Loaded Data:")
# print(movies_data)

# # Preprocess the data
# preprocessed_data = Pre_Processing(movies_data)

# # Print the preprocessed data
# print("\nPreprocessed Data:")
# print(preprocessed_data)


# # Find outliers
# outliers = Pre_Processing_Outliers(preprocessed_data)



# data_without_outliers = preprocessed_data[~preprocessed_data.index.isin(outliers.index)]
# print("Data Without Outliers")
# print(data_without_outliers)

# Clusters_number = int(input("Enter the number of clusters (k): "))

# randcentroids = RandomCentroidsIntialization(data_without_outliers, Clusters_number)

# # Print centroids
# print("Centroids:")
# print(randcentroids)

# # Perform k-means clustering


# clusters, centroids = kmeans(data_without_outliers, Clusters_number)

# formatted_output = format_clusters(clusters, centroids)
# print("K-Means Clustering Results:\n")
# print(formatted_output)

# print("......................................")
# # Output outliers
# print("Outliers:")
# print(outliers)

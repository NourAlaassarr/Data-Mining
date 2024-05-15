import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    data_copy = data.copy()
    data_copy.dropna(inplace=True)
    label_encoder = LabelEncoder()
    categorical_cols = ['gender', 'smoking_history']
    for col in categorical_cols:
        data_copy[col] = label_encoder.fit_transform(data_copy[col])
    return data_copy


def LoadData_Preprocess(filename, percentage):
    data = pd.read_csv(filename)
    num_rows = int(len(data) * percentage / 100)
    preprocessed_data = preprocess_data(data.iloc[:num_rows])
    return preprocessed_data


def split_data(data, train_percentage, test_percentage):
    num_samples = len(data)
    train_size = int(num_samples * train_percentage / 100)
    test_size = int(num_samples * test_percentage / 100)
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    return train_data, test_data


#----------------------------------------------------------------------------------->Naive Bayes
def gaussian_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# For each class label, it calculates summary statistics (mean, standard deviation, count) for each feature
def Class_Summary(dataset):
    separated = {}
    # Iterate through each row in the dataset
    for i in range(len(dataset)):
        vector = dataset[i]
        # Extract the class label from the last column
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        # Append the current row to the list corresponding to its class label
        separated[class_value].append(vector)
    summaries = {}
    for class_value, rows in separated.items():
        # Calculate summary statistics (mean, std, count) for each feature
        summaries[class_value] = [(np.mean(column), np.std(column), len(column)) for column in zip(*rows)]
        # Remove the summary statistic for the class label (last column
        del summaries[class_value][-1]
    return summaries

def Class_ProbabilityCalculation(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        #Calculates the prior probability of the class by dividing the count of samples in that class by the total number of rows.
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, stdev)
    return probabilities

def predict(summaries, row):
    probabilities = Class_ProbabilityCalculation(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def naive_bayes(train, test):
    summaries = Class_Summary(train)
    #for row in test iterates over each row in the test dataset, applying the predict function to each row.
    predictions = [predict(summaries, row) for row in test]
    return predictions

def calculate_accuracy(actual, predicted):
    correct = sum(1 for i in range(len(actual)) if actual[i] == predicted[i])
    return correct / float(len(actual)) * 100.0


#--------------------------------------------------------------------------->decision tree
def decision_tree_classifier(X_train, y_train, max_depth=3):
    tree = Build_Tree(X_train, y_train, max_depth)
    return tree

def information_gain(parent, left_child, right_child):
    p = len(left_child) / len(parent)
    entropy_parent = entropy(parent)
    # Entropy measures the impurity or disorder

    entropy_children = p * entropy(left_child) + (1 - p) * entropy(right_child)
    return entropy_parent - entropy_children

def entropy(y, bins=None):
    if bins is None:
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    else:
        discretized_y = np.digitize(y, bins)
        _, counts = np.unique(discretized_y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def Build_Tree(X, y, depth=0, max_depth=None):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))

    if (max_depth is not None and depth >= max_depth) or n_labels == 1 or n_samples < 2:
        return {'label': np.argmax(np.bincount(y)), 'depth': depth}

    best_gain = 0
    best_feature = None
    best_threshold = None
    
    for feature_index in range(n_features):
        sorted_indices = np.argsort(X[:, feature_index])
        sorted_X = X[sorted_indices, feature_index]
        sorted_y = y[sorted_indices]
        
        for i in range(1, len(sorted_X)):
            if sorted_X[i] == sorted_X[i-1]:
                continue
            
            threshold = (sorted_X[i] + sorted_X[i-1]) / 2
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold
            left_y = y[left_indices]
            right_y = y[right_indices]
            
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
                
    if best_gain == 0:
        return {'label': np.argmax(np.bincount(y)), 'depth': depth}

    left_indices = X[:, best_feature] <= best_threshold
    right_indices = X[:, best_feature] > best_threshold

    left_tree = Build_Tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right_tree = Build_Tree(X[right_indices], y[right_indices], depth + 1, max_depth)

    return {'feature_index': best_feature, 'threshold': best_threshold, 'left': left_tree, 'right': right_tree, 'depth': depth}

# _predict_tree function for each input sample in X.
def predict_tree(tree, X):
    return [_predict_tree(tree, x) for x in X]

def _predict_tree(tree, x):
    if 'label' in tree:
        return tree['label']
    if x[tree['feature_index']] <= tree['threshold']:
        return _predict_tree(tree['left'], x)
    else:
        return _predict_tree(tree['right'], x)

def custom_accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / float(len(y_true)) * 100

def decision_tree_classifier_accuracy(X_train, y_train, X_test, y_test, max_depth=3):
    tree = decision_tree_classifier(X_train, y_train, max_depth)
    predicted_labels = predict_tree(tree, X_test)
    acc = custom_accuracy(y_test, predicted_labels)
    return acc








def train_and_test():
    global train_data, test_data
    filename = filedialog.askopenfilename()
    if filename:
        data = LoadData_Preprocess(filename, float(read_percentage_entry.get()))
        train_percentage = float(train_percentage_entry.get())
        test_percentage = float(test_percentage_entry.get())
        
        # Calculate the sizes of train, test, and read data
        train_size = int(len(data) * train_percentage / 100)
        test_size = int(len(data) * test_percentage / 100)
        read_size = len(data) - train_size - test_size
        
        # Split the data
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:train_size+test_size]
        read_data = data.iloc[train_size+test_size:]
        test_rows = len(test_data)
        nb_predictions = naive_bayes(train_data.values, test_data.iloc[:, :-1].values)
        dt_predictions = predict_tree(decision_tree_classifier(train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values), test_data.iloc[:, :-1].values)
        actual = test_data.iloc[:, -1].values
        nb_accuracy = calculate_accuracy(actual, nb_predictions)
        dt_accuracy = custom_accuracy(actual, dt_predictions)
        accuracy_label.config(text=f"Naive Bayes Accuracy: {nb_accuracy:.2f}%\nDecision Tree Accuracy: {dt_accuracy:.2f}%")
        
        # Update the label to show the number of rows loaded
        rows_loaded_label.config(text=f"Rows loaded: {len(data)}\nRows for testing: {test_rows}")
        
        # Dictionary to map numerical gender values to their categorical labels
        gender_map = {0: 'female', 1: 'male'}
        
        # Display results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Actual Label\t\t\t\tNaiveBayesPredictedLabel\t\t\t\tDecisionTreePredictedLabel\t\t\t\t\t\tFeatures\n")
        for i in range(len(actual)):
            actual_label = "Diabetes" if actual[i] == 1.0 else "No Diabetes"
            nb_predicted_label = "Diabetes" if nb_predictions[i] == 1.0 else "No Diabetes"
            dt_predicted_label = "Diabetes" if dt_predictions[i] == 1.0 else "No Diabetes"
            # Replace numerical gender values with categorical labels
            features = ', '.join(str(gender_map.get(x, x)) if test_data.columns[j] == 'gender' else str(x) for j, x in enumerate(test_data.iloc[i, :-1]))
            result_text.insert(tk.END, f"{actual_label}\t\t\t\t{nb_predicted_label}\t\t\t\t{dt_predicted_label}\t\t\t\t[ {features} ]\n")

# Print test data
        print("\nTest Data:")
        print(test_data)


root = tk.Tk()
root.title("Naive Bayes and Decision Tree Classifier")

# Set background color
root.configure(bg='dark blue')

# Create labels and entry fields for input
tk.Label(root, text="Percentage of data for training set:").grid(row=0, column=0, padx=5, pady=5)
train_percentage_entry = tk.Entry(root)
train_percentage_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Percentage of data for testing set:").grid(row=1, column=0, padx=5, pady=5)
test_percentage_entry = tk.Entry(root)
test_percentage_entry.grid(row=1, column=1, padx=5, pady=5)

# New label and entry field for read percentage
tk.Label(root, text="Percentage of data for reading:").grid(row=2, column=0, padx=5, pady=5)
read_percentage_entry = tk.Entry(root)
read_percentage_entry.grid(row=2, column=1, padx=5, pady=5)

# Label to display the number of rows loaded
rows_loaded_label = tk.Label(root, text="")
rows_loaded_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

# Create button to load data and train/test the classifiers
train_test_button = tk.Button(root, text="Train and Test", command=train_and_test)
train_test_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

# Accuracy label
accuracy_label = tk.Label(root, text="")
accuracy_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

# Create a frame for the scrollable text widget
result_frame = ttk.Frame(root)
result_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

# Create a scrollable text widget
result_text = tk.Text(result_frame, height=30, width=150)  # Adjust height and width here
result_text.pack(side="left", fill="both", expand=True)

# Add a vertical scrollbar to the text widget
scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=result_text.yview)
scrollbar.pack(side="right", fill="y")
result_text.config(yscrollcommand=scrollbar.set)

root.mainloop()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("diabetes.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def euclidean_distance(x1, x2):
    """This function calculates the Euclidean distance between two points"""
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist

class KNearestNeighbors:
    """This is the class for K Nearest Neighbors Algorithm"""
    def __init__(self, k):
        """This function just stores the K value"""
        self.k = k

    def fit(self, X, y):
        """This function is used to fit the model"""
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        """This function is used to predict the model"""
        y_pred = []
        for x in X:
            print(f"\nPredicting for test point: {x}")
            prediction = self._predict(x)
            y_pred.append(prediction)
        return np.array(y_pred)

    def _predict(self, x):
        """This is the main function on how the function is going to predict"""
        distances = []
        for x_train in self.x_train:
            dist = euclidean_distance(x, x_train)
            distances.append(dist)
            print(f"Distance to {x_train}: {dist:.4f}")

        k_indexes = self._find_k_smallest_indices(distances, self.k)
        print(f"Indices of the {self.k} nearest neighbors: {k_indexes}")

        k_nearest_labels = [self.y_train[i] for i in k_indexes]
        print(f"Labels of the {self.k} nearest neighbors: {k_nearest_labels}")

        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        most_common_label = max(label_counts, key=label_counts.get)
        print(f"Most common label: {most_common_label}")

        return most_common_label

    def _find_k_smallest_indices(self, distances, k):
        k_smallest_indices = []
        distances_copy = distances.copy()

        for _ in range(k):
            min_index = None
            min_value = float('inf')

            for i, dist in enumerate(distances_copy):
                if dist < min_value:
                    min_value = dist
                    min_index = i

            k_smallest_indices.append(min_index)
            distances_copy[min_index] = float('inf')

        return k_smallest_indices

k = int(input("Enter the number of k: "))
knn = KNearestNeighbors(k)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
for i in range(len(predictions)):
    print(f"\nPredicted class: {predictions[i]}","\t",":", "\t", f"True label: {y_test[i]}")
    if predictions[i] != y_test[i]:
        print("This instance is Wrongly labelled")
    else:
        print("This instance is Correctly labelled")

accuracy = np.mean(predictions == y_test)
print(f"\nAccuracy: {accuracy:.2f}")

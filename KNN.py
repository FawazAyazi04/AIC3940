import numpy as np
import pandas as pd
from collections import Counter

dataset = pd.read_csv('IRIS.csv')

X_train = dataset.iloc[:121][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y_train = dataset.iloc[:121]['species'].values

X_test = dataset.iloc[121:][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y_test = dataset.iloc[121:]['species'].values

def euclidean_distance(x1, x2):
    """This function calculates the Euclidean distance between two points"""
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist

class KNearestNeighbors:
    """This is the class for K Nearest Neighbours Algorithm"""
    def __init__(self, k):
        """This function just stores the K value"""
        self.k = k

    def fit(self, X, y):
        """"This function is used to fit the model"""
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        """This fucntion is used to predict the model"""
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

        k_indexes = np.argsort(distances)[:self.k]
        print(f"Indices of the {self.k} nearest neighbors: {k_indexes}")

        k_nearest_labels = [self.y_train[i] for i in k_indexes]
        print(f"Labels of the {self.k} nearest neighbors: {k_nearest_labels}")

        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        print(f"Most common label: {most_common_label}")

        return most_common_label

k = int(input("Enter the number of k: "))
knn = KNearestNeighbors(k)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
for i in range(len(predictions)):
    print(f"\nPredicted classe: {predictions[i]}","\t",":", "\t" ,f"True label: {y_test[i]}")
    if predictions[i] != y_test[i]:
        print("This instance is Wrongly labelled")
    else:
        print("This instance is Correctly labelled")


accuracy = np.mean(predictions == y_test)
print(f"\nAccuracy: {accuracy:.2f}")
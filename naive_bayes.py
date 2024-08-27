import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv("diabetes.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class Naive_Bayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

            # Print intermediate steps in fit
            print(f"\nClass {c}:")
            print(f"Mean: {self._mean[c, :]}")
            print(f"Variance: {self._var[c, :]}")
            print(f"Prior: {self._priors[c]}")

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = self._predict(x)
            y_pred.append(prediction)
        return y_pred

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._prob_distribution_func(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

            print(f"\nTest point {x}:")
            print(f"Class {c}:")
            print(f"Prior (log): {prior}")
            print(f"Class Conditional (log): {class_conditional}")
            print(f"Posterior: {posterior}")

        return self._classes[np.argmax(posteriors)]

    def _prob_distribution_func(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        probability = numerator / denominator

        print(f"\nProbability Distribution for Class {self._classes[class_idx]} with mean {mean} and var {var}:")
        print(f"Numerator: {numerator}")
        print(f"Denominator: {denominator}")
        print(f"Probability: {probability}")

        return probability


nb = Naive_Bayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print(f"\nNaive Bayes accuracy: {accuracy(y_test, predictions):.2f}")

import numpy as np


class SoftmaxRegression:
    """
    A simple Softmax Regression model implemented using Gradient Descent.
    """

    def __init__(self, n_features: int, n_classes: int, n_samples: int
                 , learning_rate: float = 0.1
                 , n_iteration: int = 100):
        """
        Initializes the SoftmaxRegression model.

        Args:
            n_features(int): number of features
            n_classes(int): number of classes
            n_samples(int): number of samples
            learning_rate (float): The step size for updating weights and bias during gradient descent.
                                   A smaller learning rate requires more iterations but can prevent overshooting.
            n_iterations (int): The number of times to iterate through the training data to update the weights and bias.
        """
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_samples = n_samples

    def _softmax(self, z: np.array) -> np.array:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _predict_proba(self, x_: np.array) -> np.array:
        logits = np.dot(x_, self.weights) + self.bias
        return self._softmax(logits)

    def predict(self, x_: np.array) -> np.array:
        return np.argmax(self._predict_proba(x_), axis=1)

    def _loss(self, x_: np.array, y_onehot: np.array) -> float:
        probs = self._predict_proba(x_)
        return -np.sum(y_onehot * np.log(probs + 1e-9)) / self.n_samples

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Fits the softmax regression model to the training data using Gradient Descent.

        Args:
            x_train (np.array): The input features.
            y_train (np.array): The target values.
        """
        y_onehot = np.eye(self.n_classes)[y_train]

        for _ in range(self.n_iteration):
            probs = self._predict_proba(x_train)

            dW = np.dot(x_train.T, (probs - y_onehot)) / self.n_samples
            db = np.sum(probs - y_onehot, axis=0, keepdims=True) / self.n_samples

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db


if __name__ == "__main__":
    model = SoftmaxRegression(4, 3, 6, learning_rate=8e-4, n_iteration=10000)

    X = np.array([
        [1, 2, 3, 4],
        [4, 4, 4, 4],
        [10, 0, 6, 9],
        [1, 7, 5, 1],
        [4, 0, 0, 4],
        [1, 10, 6, 3]
    ])
    y = np.array([0, 2, 1, 2, 0, 1])

    preds_before = model.predict(X)
    probs_before = model._predict_proba(X)
    loss_before = model._loss(X, np.eye(3)[y])

    print(probs_before)
    print(preds_before)
    print(loss_before)

    model.fit(X, y)

    preds_after = model.predict(X)
    probs_after = model._predict_proba(X)
    loss_after = model._loss(X, np.eye(3)[y])

    print(probs_after)
    print(preds_after)
    print(loss_after)

    print(y)

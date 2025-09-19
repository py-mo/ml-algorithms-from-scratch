import numpy as np


class RidgeRegression:
    """
    Ridge Regression (Linear Regression with L2 regularization) implemented using Gradient Descent.
    """

    def __init__(self, n_features: int, n_samples: int,
                 learning_rate: float = 0.1,
                 n_iteration: int = 100,
                 l2_lambda: float = 0.1):
        """
        Initializes the RidgeRegression model.

        Args:
            n_features (int): number of features
            n_samples (int): number of samples
            learning_rate (float): Step size for gradient descent
            n_iterations (int): Number of iterations
            l2_lambda (float): Regularization strength
        """
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = 0
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.n_samples = n_samples
        self.n_features = n_features
        self.l2_lambda = l2_lambda

    def predict(self, x_: np.array) -> np.array:
        return (x_.dot(self.weights) + self.bias).flatten()

    def _loss(self, y_true: np.array, y_predicted: np.array) -> np.array:
        mse = np.mean((y_true - y_predicted) ** 2)
        reg = self.l2_lambda * np.sum(self.weights ** 2)
        return mse + reg

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Fits the ridge regression model using Gradient Descent.
        """
        for _ in range(self.n_iteration):
            y_predicted = self.predict(x_train)
            error = y_predicted - y_train.flatten()

            dW = (x_train.T.dot(error.reshape(-1, 1)) / self.n_samples) + (2 * self.l2_lambda * self.weights)
            dB = np.sum(error) / self.n_samples

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB


if __name__ == "__main__":
    rr = RidgeRegression(4, 3, n_iteration=250, learning_rate=8e-4, l2_lambda=0.1)
    xs = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [10, 0, 6, 9]])
    ys = np.array([6, 10, 15])

    preds = rr.predict(xs)
    print(rr._loss(ys, preds))
    print(preds)

    rr.fit(xs, ys)

    preds = rr.predict(xs)
    print(rr._loss(ys, preds))
    print(preds)

    print(ys)

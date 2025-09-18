import numpy as np


class LinearRegression:
    """
    A simple Linear Regression model implemented using Gradient Descent.
    """

    def __init__(self, n_features: int, n_samples: int
                 , learning_rate: float = 0.1
                 , n_iteration: int = 100):
        """
        Initializes the LinearRegression model.

        Args:
            n_features(int): number of features
            learning_rate (float): The step size for updating weights and bias during gradient descent.
                                   A smaller learning rate requires more iterations but can prevent overshooting.
            iterations (int): The number of times to iterate through the training data to update the weights and bias.
        """
        self.weights = np.ones(shape=(n_features, 1))
        self.bias = -5
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.n_samples = n_samples
        self.n_features = n_features

    def predict(self, x_: np.array) -> np.array:
        return (x_.dot(self.weights) + self.bias).flatten()

    def _loss_mse(self, y_true: np.array, y_predicted: np.array) -> np.array:
        return np.mean((y_true - y_predicted) ** 2)

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Fits the linear regression model to the training data using Gradient Descent.

        Args:
            x_train (np.array): The input features.
            y_train (np.array): The target values.
        """

        for _ in range(self.n_iteration):
            y_predicted = self.predict(x_train)
            error = y_predicted - y_train.flatten()

            dW = x_train.T.dot(error.reshape(-1, 1)) / self.n_samples
            dB = np.sum(error) / self.n_samples

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB


if __name__ == "__main__":
    lr = LinearRegression(4, 3, n_iteration=250, learning_rate=8e-4)
    xs = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [10, 0, 6, 9]])
    ys = np.array([6, 10, 15])

    preds: np.array = lr.predict(xs)
    print (lr._loss_mse(ys, preds))
    print (preds)

    lr.fit(xs, ys)

    preds = lr.predict(xs)
    print (lr._loss_mse(ys, preds))
    print (preds)

    print (ys)
    
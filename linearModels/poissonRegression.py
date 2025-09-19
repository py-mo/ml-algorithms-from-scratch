import numpy as np


class PoissonRegression:
    """
    Poisson Regression implemented with Gradient Descent.
    """

    def __init__(self, n_features: int, n_samples: int,
                 learning_rate: float = 0.01,
                 n_iteration: int = 1000):
        """
        Initializes the PoissonRegression model.

        Args:
            n_features(int): number of features
            n_samples(int): number of samples
            learning_rate (float): The step size for updating weights and bias during gradient descent.
                                   A smaller learning rate requires more iterations but can prevent overshooting.
            n_iterations (int): The number of times to iterate through the training data to update the weights and bias.
        """
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.n_samples = n_samples
        self.n_features = n_features

    def _linear_predictor(self, X: np.array) -> np.array:
        return (X.dot(self.weights) + self.bias).flatten()

    def predict(self, X: np.array) -> np.array:
        return np.exp(self._linear_predictor(X))

    def _loss_nll(self, y_true: np.array, mu: np.array) -> float:
        return -np.mean(y_true * np.log(mu + 1e-12) - mu)

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Train Poisson regression using Gradient Descent.
        
        Args:
            x_train (np.array): The input features.
            y_train (np.array): The target values.
        """
        for _ in range(self.n_iteration):
            mu = self.predict(x_train)
            error = mu - y_train.flatten()

            dW = x_train.T.dot(error.reshape(-1, 1)) / self.n_samples
            dB = np.mean(error)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB


if __name__ == "__main__":
    X = np.array([[1, 2],
                  [2, 1],
                  [3, 4],
                  [4, 3]])
    y = np.array([2, 1, 6, 5])

    pr = PoissonRegression(n_features=2, n_samples=4,
                           learning_rate=0.01, n_iteration=5000)

    preds = pr.predict(X)
    print(pr._loss_nll(y, preds))
    print(preds)

    pr.fit(X, y)

    preds = pr.predict(X)
    print(pr._loss_nll(y, preds))
    print(preds)
    print(y)
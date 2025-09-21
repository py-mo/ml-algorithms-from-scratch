import numpy as np


class ElasticNet:
    """
    ElasticNet implemented using Gradient Descent.
    """

    def __init__(self, n_features: int, n_samples: int,
                 learning_rate: float = 0.1,
                 n_iteration: int = 100,
                 l1_lambda: float = 0.1,
                 l2_lambda: float = 0.1):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.n_samples = n_samples
        self.n_features = n_features
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def predict(self, x_: np.array) -> np.array:
        return (x_.dot(self.weights) + self.bias).flatten()

    def _loss(self, y_true: np.array, y_predicted: np.array) -> float:
        mse = np.mean((y_true - y_predicted) ** 2)
        reg = self.l1_lambda * np.sum(np.abs(self.weights))
        reg += self.l2_lambda * np.sum(self.weights ** 2)
        return mse + reg

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Fits the elastic net model using Gradient Descent.
        """
        for _ in range(self.n_iteration):
            y_pred = self.predict(x_train)
            error = y_pred - y_train.flatten()

            dW = (x_train.T.dot(error.reshape(-1, 1)) / self.n_samples) 
            + (self.l1_lambda * np.sign(self.weights))
            + (2 * self.l2_lambda * self.weights)
            dB = np.sum(error) / self.n_samples

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB


if __name__ == "__main__":
    lr = ElasticNet(4, 3, n_iteration=500, learning_rate=8e-4
                           , l1_lambda=0.1, l2_lambda=0.1)
    xs = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [10, 0, 6, 9]])
    ys = np.array([6, 10, 15])

    preds = lr.predict(xs)
    print(lr._loss(ys, preds))
    print(preds)

    lr.fit(xs, ys)

    preds = lr.predict(xs)
    print(lr._loss(ys, preds))
    print(preds)
    print(ys)

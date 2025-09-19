import numpy as np


class GAModel:
    """
    A simple Generalized Additive Model using polynomial basis functions.
    """

    def __init__(self, n_features: int, n_samples: int, degree: int = 3
                 , learning_rate: float = 1e-3, n_iterations: int = 500):
        self.n_features = n_features
        self.n_samples = n_samples
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = [np.zeros(degree + 1) for _ in range(n_features)]
        self.bias = 0.0

    def _basis_expand(self, x_: np.array):
        expanded = []
        for j in range(self.n_features):
            col = x_[:, j]
            basis = np.vstack([col ** d for d in range(self.degree + 1)]).T
            expanded.append(basis)
        return expanded

    def predict(self, x_: np.array):
        x_ = (x_ - x_.mean(axis=0)) / x_.std(axis=0)
        expanded = self._basis_expand(x_)
        y_pred = self.bias + sum(expanded[j].dot(self.weights[j]) for j in range(self.n_features))
        return y_pred

    def _loss_mse(self, y_true: np.array, y_pred: np.array):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, x_train: np.array, y_train: np.array):
        for _ in range(self.n_iterations):
            y_pred = self.predict(x_train)
            error = y_pred - y_train

            expanded = self._basis_expand(x_train)
            for j in range(self.n_features):
                dW = expanded[j].T.dot(error) / self.n_samples
                self.weights[j] -= self.learning_rate * dW

            dB = np.sum(error) / self.n_samples
            self.bias -= self.learning_rate * dB


if __name__ == "__main__":
    xs = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [10, 0, 6, 9]])
    ys = np.array([27, 42, 55])

    gam = GAModel(4, 3, 3, learning_rate=5e-6, n_iterations=25000)
    gam.fit(xs, ys)

    preds = gam.predict(xs)
    print(gam._loss_mse(ys, preds))
    print(preds)
    print(ys)
import numpy as np


class CoxPH:
    """
    Cox Proportional Hazards Model using partial likelihood.
    """

    def __init__(self, n_features: int, n_samples: int
                 , learning_rate: float = 0.01, n_iterations: int = 1000):
        self.n_features = n_features
        self.n_samples = n_samples
        self.weights = np.zeros(n_features)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _risk_score(self, x_):
        return np.exp(x_.dot(self.weights))

    def _partial_likelihood(self, x_, T, E):
        """
        T: time of event or censoring
        E: event occurred (1) or censored (0)
        """
        risk_scores = self._risk_score(x_)
        log_likelihood = 0.0
        for i in range(len(T)):
            if E[i] == 1:
                log_likelihood += x_[i].dot(self.weights) - np.log(np.sum(risk_scores[T >= T[i]]))
        return log_likelihood

    def _gradient(self, x_, T, E):
        risk_scores = self._risk_score(x_)
        grad = np.zeros_like(self.weights)
        for i in range(len(T)):
            if E[i] == 1:
                risk_set = risk_scores[T >= T[i]]
                x_risk_set = x_[T >= T[i]]
                grad += x_[i] - np.sum(x_risk_set.T * risk_set, axis=1) / np.sum(risk_set)
        return grad

    def fit(self, x_train, T, E):
        for _ in range(self.n_iterations):
            dW = self._gradient(x_train, T, E)
            self.weights += self.learning_rate * dW

    def predict_risk(self, x_):
        x_ = (x_ - x_.mean(axis=0)) / x_.std(axis=0)
        return self._risk_score(x_)

    def c_index(self, T, E, x_):
        risk = self.predict_risk(x_)
        n = 0
        n_concordant = 0

        for i in range(len(T)):
            for j in range(len(T)):
                if i == j:
                    continue
                if T[i] < T[j] and E[i] == 1:
                    n += 1
                    if risk[i] > risk[j]:
                        n_concordant += 1
                    elif risk[i] == risk[j]:
                        n_concordant += 0.5

        if n == 0:
            return 0
        return n_concordant / n


if __name__ == "__main__":
    xs = np.array([
        [50, 130],
        [60, 120],
        [45, 150],
        [70, 110],
        [55, 140]
    ])

    T = np.array([10, 12, 8, 15, 9])
    E = np.array([1, 1, 0, 1, 0])

    model = CoxPH(2, 5, learning_rate=0.01, n_iterations=250)
    model.fit(xs, T, E)

    risks = model.predict_risk(xs)
    print(risks)
    print(model.c_index(T, E, xs))

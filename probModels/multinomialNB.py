import numpy as np


class MultinomialNB:
    """
    Multinomial Naive Bayes classifier.
    """

    def __init__(self, n_features: int, n_classes: int, alpha: float = 1.0):
        assert alpha >= 0.0, "alpha must be non-negative"
        self.n_features = n_features
        self.n_classes = n_classes
        self.alpha = alpha

    def fit(self, x_train: np.array, y_train: np.array):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        self.classes_, y_idx = np.unique(y_train, return_inverse=True)

        self.class_count_ = np.zeros(self.n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((self.n_classes, self.n_features), dtype=np.float64)

        for i, c in enumerate(self.classes_):
            X_c = x_train[y_train == c]
            self.class_count_[i] = X_c.shape[0]
            self.feature_count_[i, :] = X_c.sum(axis=0)

        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_norm = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_norm)

        return self

    def _joint_log_likelihood(self, x_: np.array):
        return x_.dot(self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob_norm = jll - jll.max(axis=1, keepdims=True)
        proba = np.exp(log_prob_norm)
        return proba / proba.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    xs = np.array([[1.0, 1.8], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0], [1.1, 1.7], [1.2, 1.6]])
    ys = np.array([0, 0, 1, 1, 1, 0])

    gnb = MultinomialNB(n_features=2, n_classes=2)
    gnb.fit(xs, ys)

    print (gnb.predict(xs))
    print (gnb.predict_proba(xs))

    print (ys)
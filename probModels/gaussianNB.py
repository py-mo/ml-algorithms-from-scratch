import numpy as np


class GaussianNB:
    """
    Gaussian Naive Bayes classifier.
    Assumes each feature is normally distributed per class.
    """

    def __init__(self, n_features: int, n_classes: int, var_smoothing=1e-9):
        self.n_features = n_features
        self.n_classes = n_classes
        self.var_smoothing = var_smoothing

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        self.classes_, y_idx = np.unique(y_train, return_inverse=True)
        self.theta_ = np.zeros((self.n_classes, self.n_features))
        self.var_ = np.zeros((self.n_classes, self.n_features))
        self.class_prior_ = np.zeros(self.n_classes)

        for i, c in enumerate(self.classes_):
            X_c = x_train[y_idx == i]
            self.theta_[i, :] = X_c.mean(axis=0)
            self.var_[i, :] = X_c.var(axis=0) + self.var_smoothing
            self.class_prior_[i] = X_c.shape[0] / x_train.shape[0]

        return self

    def _joint_log_likelihood(self, x_):
        x_ = np.asarray(x_)
        jll = []

        for i in range(self.n_classes):
            log_prob = -0.5 * np.sum(
                np.log(2. * np.pi * self.var_[i])
                + ((x_ - self.theta_[i]) ** 2) / self.var_[i],
                axis=1
            )
            jll.append(np.log(self.class_prior_[i]) + log_prob)

        return np.array(jll).T

    def predict(self, x_):
        jll = self._joint_log_likelihood(x_)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, x_):
        jll = self._joint_log_likelihood(x_)
        log_prob_norm = jll - jll.max(axis=1, keepdims=True)
        proba = np.exp(log_prob_norm)
        return proba / proba.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    xs = np.array([[1.0, 1.8], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0], [1.1, 1.7], [1.2, 1.6]])
    ys = np.array([0, 0, 1, 1, 1, 0])

    gnb = GaussianNB(n_features=2, n_classes=2)
    gnb.fit(xs, ys)

    print (gnb.predict(xs))
    print (gnb.predict_proba(xs))

    print (ys)
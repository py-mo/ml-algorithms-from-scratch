import numpy as np


class LogisticRegression:
    """
    A simple Logistic Regression model.
    """

    def __init__(self, n_features: int, n_samples: int
                 , learning_rate: float = 0.1
                 , n_iteration: int = 100):
        """
        Initializes the LosgisticRegression model.

        Args:
            n_features(int): number of features
            n_samples(int): number of samples
            learning_rate (float): The step size for updating weights and bias during gradient descent.
                                   A smaller learning rate requires more iterations but can prevent overshooting.
            n_iterations (int): The number of times to iterate through the training data to update the weights and bias.
        """
        self.n_iteration = n_iteration
        self.n_features = n_features
        self.n_samples = n_samples
        self.weights = np.zeros(shape=(n_features, 1))
        self.bias = 0
        self.learning_rate = learning_rate
    
    def predict(self, x_: np.array) -> np.array:
        proba = self._predict_proba(x_)
        return (proba >= 0.5).astype(int)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _predict_proba(self, x_: np.array) -> np.array:
        linear_output = np.dot(x_, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def _loss_log(self, y_true: np.array, y_predicted: np.array) -> float:
        eps = 1e-15
        y_predicted = np.clip(y_predicted, eps, 1 - eps)

        lg_loss = -np.mean(
            y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted)
        )
        return lg_loss

    def fit(self, x_train: np.array, y_train: np.array):
        """
        Fits the linear regression model to the training data using Gradient Descent.

        Args:
            x_train (np.array): The input features.
            y_train (np.array): The target values.
        """
        self.weights = np.zeros(self.n_features)
        self.bias = 0.0

        for _ in range(self.n_iteration):
            linear_output = np.dot(x_train, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_output)

            dw = (1 / self.n_samples) * np.dot(x_train.T, (y_predicted - y_train))
            db = (1 / self.n_samples) * np.sum(y_predicted - y_train)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

if __name__ == "__main__":
    lr = LogisticRegression(4, 3, n_iteration=1000, learning_rate=8e-4)
    xs = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [10, 0, 6, 9]])
    ys = np.array([0, 1, 1])

    preds: np.array = lr.predict(xs)
    print (lr._loss_log(ys, preds))
    print (preds)

    lr.fit(xs, ys)

    preds = lr.predict(xs)
    print (lr._loss_log(ys, preds))
    print (preds)

    print (ys)

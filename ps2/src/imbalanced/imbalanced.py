import numpy as np
import util
import sys
from random import random

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    model = LogisticRegression(verbose=False)
    model.fit(x_train, y_train)
    
    x_validation, y_validation = util.load_dataset(validation_path, add_intercept=True)
    validation_preds = model.predict(x_validation)
    np.savetxt(output_path_naive, validation_preds)
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    
    for i in range(y_validation.shape[0]):
        if y_validation[i] == 1 and validation_preds[i] > 0.5:
            TP = TP + 1
        if y_validation[i] == 1 and validation_preds[i] < 0.5:
            FN = FN + 1
        if y_validation[i] == 0 and validation_preds[i] > 0.5:
            FP = FP + 1
        if y_validation[i] == 0 and validation_preds[i] < 0.5:
            TN = TN + 1
    print(f"TN, TP, FN, FP = {TN, TP, FN, FP}")
    print(f"the total number of sample is {TN + TP + FN + FP}")
    classifier_accuracy = (TN + TP) / (TN + TP + FN + FP)
    A0 = TN / (TN + FP)
    A1 = TP / (TP + FN)
    balanced_accuracy =  1 / 2 * (A0 + A1)
    print(f"classifier_accuracy {classifier_accuracy}")
    print(f"A0 {A0}")
    print(f"A1 {A1}")
    print(f"balanced_accuracy {balanced_accuracy}")
    print("=============================================================")
    util.plot(x_validation, y_validation, model.theta, "ps2::q4::(b).png")

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    model = LogisticRegression_Upsampling(verbose=False)
    model.fit(x_train, y_train)
    x_validation, y_validation = util.load_dataset(validation_path, add_intercept=True)
    validation_preds = model.predict(x_validation)
    np.savetxt(output_path_upsampling, validation_preds)
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    
    for i in range(y_validation.shape[0]):
        if y_validation[i] == 1 and validation_preds[i] > 0.5:
            TP = TP + 1
        if y_validation[i] == 1 and validation_preds[i] < 0.5:
            FN = FN + 1
        if y_validation[i] == 0 and validation_preds[i] > 0.5:
            FP = FP + 1
        if y_validation[i] == 0 and validation_preds[i] < 0.5:
            TN = TN + 1
    print("=============================================================")
    print(f"TN, TP, FN, FP = {TN, TP, FN, FP}")
    print(f"the total number of sample is {TN + TP + FN + FP}")
    classifier_accuracy = (TN + TP) / (TN + TP + FN + FP)
    A0 = TN / (TN + FP)
    A1 = TP / (TP + FN)
    balanced_accuracy =  1 / 2 * (A0 + A1)
    print(f"classifier_accuracy {classifier_accuracy}")
    print(f"A0 {A0}")
    print(f"A1 {A1}")
    print(f"balanced_accuracy {balanced_accuracy}")
    util.plot(x_validation, y_validation, model.theta, "ps2::q4::(d).png")
    # *** END CODE HERE

class LogisticRegression_Upsampling:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True, regularization=0):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        # *** START CODE HERE ***
        self.regularization = regularization
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1]) if self.theta is None else self.theta
        positive_y = 0
        for i in range (y.shape[0]):
            if abs(y[i] - 1) < self.eps:
                positive_y = positive_y + 1
        n = x.shape[0]
        negative_y = n - positive_y
        kappa = positive_y / negative_y
        weight = np.where(y == 0, 1, y * (1 / kappa))
        weight = weight.reshape(1, -1)
        print(f"y_transformed.shape = {weight.shape}")
        print(f"{weight}")

        n = x.shape[0]
        for i in range(self.max_iter):
            h = 1 / (1 + np.exp(-(x @ self.theta))) # (100,)
            #grad = 1 / n * np.dot(x.T, (y - h)) - self.regularization * self.theta 
            grad = 1 / n * np.dot(x.T * weight, (y - h))
            self.theta = self.theta + self.learning_rate * grad
            if np.linalg.norm(grad) < self.eps:
                print(f"LR converge at i = {i}")
                break
            if self.verbose and i % 10000 == 0:
                print(f"grad = {grad}")
                print(f"self.theta = {self.theta}")
                print(f'Iteration {i}, Loss: {self._compute_loss(x, y)}')
        print(f'Loss: {self._compute_loss(x, y)}')
        print(f"Final theta = {self.theta}")
        # *** END CODE HERE ***

    def _compute_loss(self, x, y):
        n = x.shape[0]
        h = self.predict(x)
        return -(1/n) * np.sum(y * np.log(h + self.eps) + (1 - y) * np.log(1 - h + self.eps))

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(x @ self.theta)))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')

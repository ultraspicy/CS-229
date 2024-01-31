import numpy as np
import util


def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # print(f"x_train.shape = {x_train.shape}")
    # print(f"y_train.shape = {y_train.shape}")
    model = LogisticRegression(verbose=False)
    model.fit(x_train, y_train)
    preds = model.predict(x_train)
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, preds)
    if save_path == 'logreg_pred_a.txt':
        util.plot(x_train, y_train, model.theta, "ps2::q3::(a)", correction=1.0)
    else:
        util.plot(x_train, y_train, model.theta, "ps2::q3::(b)", correction=1.0)

    model = LogisticRegression(verbose=False, regularization=0.01)
    model.fit(x_train, y_train)
    preds = model.predict(x_train)
    np.savetxt(save_path + '_reg', preds)
    if save_path == 'logreg_pred_a.txt':
        util.plot(x_train, y_train, model.theta, "ps2::q3::(d)_ds1_a_reg", correction=1.0)
    else:
        util.plot(x_train, y_train, model.theta, "ps2::q3::(d)_ds1_b_reg", correction=1.0)
    # *** END CODE HERE ***


class LogisticRegression:
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
        #print(f"theta = {self.theta}") # theta.shape = (3,)
        n = x.shape[0]
        for i in range(self.max_iter):
            h = 1 / (1 + np.exp(-(x @ self.theta))) # (100,)
            grad = 1 / n * np.dot(x.T, (y - h)) - self.regularization * self.theta 
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
    print('==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a.txt')

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b.txt')

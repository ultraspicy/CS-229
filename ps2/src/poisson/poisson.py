import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    model = PoissonRegression()
    model.fit(x_train, y_train)
    
    # Run on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    # and use np.savetxt to save outputs to save_path
    np.savetxt(save_path, y_pred)
    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_eval, y_pred, alpha=0.5)
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.xlabel('True Counts')
    plt.ylabel('Predicted Counts')
    plt.title('True Counts vs. Predicted Counts')
    plt.savefig("ps2::q1::(d).png")
    plt.show() 
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

        ############################
        self.learning_rate=0.01

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n = x.shape[0]
        d = x.shape[1]
        if self.theta is None:
            self.theta = np.zeros(d)

        print(f"self.theta.shape = {self.theta.shape}")
        for i in range(self.max_iter): #range(100): #
            # for jth sample
            grad = np.zeros(d)
            for j in range(n):
                # x_jth is a vector, y is a scalar
                # x_jth = x[j,:]
                # print(f"x[]j,: = {x[j,:].shape}")
                # print(f"{np.exp(self.theta.T @ x[j,:])}")
                grad = grad + y[j] * (1 / np.exp(self.theta.T @ x[j,:])) * x[j,:] - x[j,:] 
            grad = grad / n
            self.theta = self.theta + self.learning_rate * grad
            # print(f"grad = {grad}")
            if np.linalg.norm(grad) < self.eps:
                print(f"LR converge at i = {i}")
                break
            if self.verbose and i % 1000 == 0:
                print(f"grad = {grad}")
                print(f"self.theta = {self.theta}")
                print(f"np.linalg.norm(grad) = {np.linalg.norm(grad)}")
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        # print(f"theta = {self.theta}")
        # print(f"{self.theta.T.shape}")
        # print(f"{x.shape}")
        return np.exp(x @ self.theta.T)
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')

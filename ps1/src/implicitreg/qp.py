import numpy as np
import matplotlib.pyplot as plt
import util

# quadratic parameterized model
class QP:
    def __init__(self, dim, beta = None):
        self.dim = dim
        if beta is None:
            self.theta = np.ones(dim)
            self.phi = np.ones(dim)
        else:
            self.theta = beta.copy()
            self.phi = beta.copy()
            
    def train_GD(self, X, Y, eta = 8e-2, max_step = 1000, 
                 verbose = False, X_test = None, Y_test = None):
        """Train the QP model using gradient descent

        Args:
            X: shape (n, d) matrix representing the input
            Y: shape (n) vector representing the label
            eta: learning rate
            max_step: maximum training steps
            verbose: return training/test logs
            X_test: test input
            Y_test: test output
        """
        if verbose:
            log_steps = []
            log_tests = []
            log_trains = []

        for t in range(max_step):
            # *** START CODE HERE ***
            # Compute the gradient of self.theta, self.phi using data X, Y
            grad_theta, grad_phi = self.gradient(X, Y)
            # and update self.theta, self.phi
            self.theta = self.theta - eta * grad_theta
            self.phi = self.phi - eta * grad_phi
            # *** END CODE HERE ***
            if verbose:
                log_steps.append(t)
                log_tests.append(self.test(X_test, Y_test))
                log_trains.append(self.test(X, Y))
        if verbose:
            return (log_steps, log_trains, log_tests)
            
    def train_SGD(self, X, Y, eta = 8e-2, max_step = 1000, batch_size = 1, 
                  verbose = False, X_test = None, Y_test = None):
        """Train the QP model using stochastic gradient descent

        Args:
            X: shape (n, d) matrix representing the input
            Y: shape (n) vector representing the label
            eta: learning rate
            max_step: maximum training steps
            batch_size: batch size of the SGD algorithm
            verbose: return training/test logs
            X_test: test input
            Y_test: test output
        """
        if verbose:
            log_steps = []
            log_tests = []
            log_trains = []

        np.random.seed(0)
        n = X.shape[0]
        idx = np.arange(n)

        for t in range(max_step):
            idx_batch = np.random.choice(idx, size = batch_size, replace = False)
            X_batch = X[idx_batch]
            Y_batch = Y[idx_batch]

            # *** START CODE HERE ***
            # Compute the gradient of self.theta, self.phi using data X_batch, Y_batch
            grad_theta, grad_phi = self.gradient(X_batch, Y_batch)
            # and update self.theta, self.phi
            self.theta = self.theta - eta * grad_theta
            self.phi = self.phi - eta * grad_phi
            # *** END CODE HERE ***
            if verbose:
                log_steps.append(t)
                log_tests.append(self.test(X_test, Y_test))
                log_trains.append(self.test(X, Y))
        if verbose:
            return (log_steps, log_trains, log_tests)
    
    def predict(self, X):
        return X.dot(self.theta ** 2 - self.phi ** 2)
    
    def gradient(self, X, Y):
        """Return the gradient w.r.t. theta and phi
        """
        # *** START CODE HERE ***
        n = X.shape[0]
        error = X @ (self.theta ** 2 - self.phi ** 2) - Y
        #print(f"error = {error}")
        # print(f"error.shape = {error.shape}")

        # Gradient w.r.t. theta
        grad_theta = 1 / n * (X * (error[:, None] * self.theta[None, :])).mean(axis=0)
        # Gradient w.r.t. phi
        grad_phi = -1 / n * (X * (error[:, None] * self.phi[None, :])).mean(axis=0)
       
        return grad_theta, grad_phi
        # *** END CODE HERE ***
    
    def test(self, X, Y):
        #print(f"self.predict(X) = {self.predict(X)}")
        return np.average(0.25 * (self.predict(X) - Y) ** 2)

def QP_model_initialization(train_path, test_path):
    X, Y = util.load_dataset(train_path)
    X_test, Y_test = util.load_dataset(test_path)
    d = X.shape[1]
    
    save_path = "implicitreg_quadratic_initialization"

    # Use gradient descent to train a quadratically parameterized 
    # model with different initialization. Plot the curves of test
    # error against the number of gradient steps in a single figure.
    log = []
    labels = []
    alphas = [0.1, 0.03, 0.01]
    for i in range(len(alphas)):
        alpha = alphas[i]
        model = QP(d, np.ones(d) * alpha)
        log.append(model.train_GD(X, Y, verbose = True, X_test = X_test, Y_test = Y_test))
        labels.append("init. = {}".format(alphas[i]))
        print("initialization: {:.3f}".format(alpha), "final test error: ", model.test(X_test, Y_test))
    util.plot_training_and_test_curves(log, save_path, label = labels)

def QP_model_batchsize(train_path, test_path):
    X, Y = util.load_dataset(train_path)
    X_test, Y_test = util.load_dataset(test_path)
    d = X.shape[1]
    
    save_path = "implicitreg_quadratic_batchsize"

    # Use SGD to train a quadratically parameterized model with
    # different batchsize. Plot the curves of test
    # error against the number of gradient steps in a single figure.
    log = []
    labels = []
    bs = [1, 5, 40]
    for i in range(len(bs)):
        model = QP(d, np.ones(d) * 0.1)
        log.append(model.train_SGD(X, Y, eta = 0.08, batch_size = bs[i], verbose = True, 
                        X_test = X_test, Y_test = Y_test))
        labels.append("batch size = {}".format(bs[i]))
        print("batchsize: ", bs[i], "final test error: ", model.test(X_test, Y_test))
    util.plot_training_and_test_curves(log, save_path, label = labels)

def implicitreg_main():
    train_path = 'ir2_train.csv'
    test_path = 'ir2_test.csv'
    QP_model_initialization(train_path, test_path)
    QP_model_batchsize(train_path, test_path)

if __name__ == '__main__':
    implicitreg_main()

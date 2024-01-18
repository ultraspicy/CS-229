import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # a = X.T @ X
        # b = X.T @ y
        # # if inputs.ndim == 1:
        # # inputs = np.expand_dims(inputs, -1)
        # print(f"a.shape {a.shape}")
        # print(f"b.shape {b.shape}")

        self.theta = np.linalg.solve(X.T @ X, X.T @ y)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n = X.shape[0]
        poly_features = np.ones((n, k + 1))

        for i in range(1, k + 1):
            poly_features[:, i] = X[:, 1] ** i
        
        return poly_features
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n = X.shape[0]
        poly_features_with_sin = np.ones((n, k + 2))

        for i in range(1, k + 1):
            poly_features_with_sin[:, i] = X[:, 1] ** i
        poly_features_with_sin[:, k + 1] = np.sin(X[:, 1])
        
        #util.print_matrix(poly_features_with_sin)
        return poly_features_with_sin

        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n = X.shape[0]
        y_pred = np.zeros(n)
        for i in range(n):
            y_pred[i] = self.theta.T @ X[i, :]
        return y_pred

        # *** END CODE HERE ***


def run_exp(train_path, eval_path, sine=False, ks=[0, 1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    # *** START CODE HERE ***

    # Load validation set and use validation poly features
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    print(f"x_eval.shape =  {x_eval.shape}")
    x_eval_smooth = np.linspace(min(x_eval[:, 1]), max(x_eval[:, 1]), 1000).reshape(-1, 1)
    x_eval_smooth = np.hstack([np.ones_like(x_eval_smooth), x_eval_smooth])
    print(f"x_eval_smooth.shape =  {x_eval_smooth.shape}")

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        print(f"k = {k}")
        # Load training set
        x_train, y_train = util.load_dataset(train_path, add_intercept=True)
        print(f"shape of x_train {x_train.shape}")
        print(f"shape of y_train {y_train.shape}")
        # Fit a LWR model
        model = LinearModel()
        # convert the x_train into poly features
        if not sine:
            x_train =  model.create_poly(k, x_train)
        else:
            x_train =  model.create_sin(k, x_train)
        model.fit(x_train, y_train)
        if not sine:
            x_eval_smooth =  model.create_poly(k, x_eval_smooth)
        else:
            x_eval_smooth =  model.create_sin(k, x_eval_smooth)

        print(f"x_eval_smooth.shape =  {x_eval_smooth.shape}")
        y_pred = model.predict(x_eval_smooth)
    
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        if k == 1:
            plt.plot(x_eval_smooth[:, 1], y_pred, 'c-', label='k=%d' % k)
        elif k == 2:
            plt.plot(x_eval_smooth[:, 1], y_pred, 'g-', label='k=%d' % k)
        elif k == 3:
            plt.plot(x_eval_smooth[:, 1], y_pred, 'r-', label='k=%d' % k)
        elif k == 5:
            plt.plot(x_eval_smooth[:, 1], y_pred, 'k-', label='k=%d' % k)
        elif k == 10:
            plt.plot(x_eval_smooth[:, 1], y_pred, 'b-', label='k=%d' % k)
        elif k == 20:
            plt.plot(x_eval_smooth[:, 1], y_pred, 'm-', label='k=%d' % k)

    # *** END CODE HERE ***
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # this is for (b)
    #run_exp_k_equals_3(train_path, small_path, eval_path)
    # this is for (c)
    #run_exp(train_path, eval_path)
    # this is for (d)
    run_exp(train_path, eval_path, True)

    

def run_exp_k_equals_3 (train_path, small_path, eval_path):
    # *** START CODE HERE ***
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    print(f"shape of x_train {x_train.shape}")
    print(f"shape of y_train {y_train.shape}")
    # Fit a LWR model
    model = LinearModel()

    # convert the x_train into poly features then train the model
    x_train_poly =  model.create_poly(3, x_train)
    model.fit(x_train_poly, y_train)
    # we need to plot the learnt hypothesis as a smooth curve over it
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    print(f"shape of x_eval {x_eval.shape}")
    x_smooth = np.linspace(min(x_eval[:, 1]), max(x_eval[:, 1]), 1000).reshape(-1, 1)
    x_smooth = np.hstack([np.ones_like(x_smooth), x_smooth])
    print(f"shape of x_smooth {x_smooth.shape}")
    x_poly_smooth = model.create_poly(3, x_smooth)
    y_pred_smooth = model.predict(x_poly_smooth)
    
    # create a scatter plot of the training data
    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', label='training data')
    # we need to plot the learnt hypothesis as a smooth curve over it
    plt.plot(x_poly_smooth[:, 1], y_pred_smooth, 'r-', label='learnt hypothesis')
    plt.legend()
    plt.show()
    # *** END CODE HERE ***    

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')

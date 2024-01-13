import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import util

def generate_plot(betas, X, Y, X_test, Y_test, save_path):
    """Generate a scatter plot of test error vs. norm

    Args:
        betas: list of numpy arrays, indicating different solutions
        X, Y: training dataset 
        X_test, Y_test: test dataset
        save_path: path to save the plot
    """
    # check if the test error is zero
    for b in betas:
        assert(np.allclose(X.dot(b), Y))

    # compute the norm and the test error of all the solutions in list beta
    test_err = []
    norms = []
    for i in range(len(betas)):
        test_err.append(np.mean((X_test.dot(betas[i]) - Y_test) ** 2))
        norms.append(np.linalg.norm(betas[i]))

    # plot the test error against norm of the solution
    util.plot_points(norms, test_err, save_path)

def linear_model_main():
    save_path_linear = "implicitreg_linear"
    
    train_path = 'ir1_train.csv'
    test_path = 'ir1_test.csv'
    X, Y = util.load_dataset(train_path)
    X_test, Y_test = util.load_dataset(test_path)
    
    beta_0 = None
    # *** START CODE HERE ***
    # find the min norm solution of the training dataset
    # store the results to beta_0
    # *** END CODE HERE ***
    
    assert(np.allclose(X.dot(beta_0), Y))
    
    # ns[i] is orthogonal to all the inputs in the training dataset
    # to help you understand the starter code, check the dimension 
    # of ns before you use it
    ns = null_space(X).T

    # *** START CODE HERE ***
    # find 3 different solutions and generate a scatter plot
    # your plot should include the min norm solution and 3 different solutions
    # you can use the function generate_plot()
    # *** END CODE HERE ***

if __name__ == '__main__':
    linear_model_main()

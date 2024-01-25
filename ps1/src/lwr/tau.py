import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***

    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # Load validation set
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    # Load test set
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    
    # Initialize variables to store the best tau and its MSE
    best_tau = None
    best_mse = float('inf')

    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        # Fit LWR model
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        # Make predictions on validation set
        y_valid_pred = model.predict(x_valid)
        # Calculate MSE on validation set
        mse = np.mean((y_valid - y_valid_pred) ** 2)
        # Update best tau if this is the lowest MSE so far
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
        plt.figure(f"tau = {tau}")
        plt.plot(x_train[:, 1], y_train, 'bx', label='Training Data')
        plt.plot(x_valid[:, 1], y_valid, 'go', label='Validation Data')
        lable = f'LWR Predictions tau = {tau}'
        plt.plot(x_valid[:, 1], y_valid_pred, 'ro', label=lable)
        plt.legend()
        filename = f'ps1_q2_(c)_tau_{tau}.png'
        plt.savefig(filename)
        plt.clf()
            
    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    y_test_pred = model.predict(x_test)
    mse = np.mean((y_test - y_test_pred) ** 2)
    print(f"best_tau = {best_tau}, MSE on the test split using this τ value is {mse}")
    plt.figure(f"best_tau = {best_tau}")
    plt.plot(x_train[:, 1], y_train, 'bx', label='Training Data')
    plt.plot(x_test[:, 1], y_test, 'go', label='Test Data')
    plt.plot(x_test[:, 1], y_test_pred, 'ro', label='LWR Predictions on test data')
    plt.legend()
    filename = f'ps1_q2_(c)_best_tau_{best_tau}.png'
    plt.savefig(filename)
    plt.clf()
    # Save predictions to pred_path  def plot(x, y, theta, save_path, correction=1.0):
    # Plot data
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')

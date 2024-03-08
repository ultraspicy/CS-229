import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group

    # print(f"x.shape = {x.shape}") #(980, 2)
    n = x.shape[0]
    # np.random.seed(seed) is used to set the random seed for NumPy's random number generator.
    # np.random.seed(trial_num)  # For reproducibility
    # np.random.choice(a, size=None, replace=True, p=None) is a function from 
    # the NumPy library that generates a random sample from a given 1-D array or integer.
    
    assignments = np.random.choice(K, n) 
    # print(f"assignments.shape = {assignments.shape}") # (980,)
    
    mu = [np.mean(x[assignments == k, :], axis=0) for k in range(K)]
    # print(f"{mu[0]}")
    # print(f"{mu[1]}")
    # print(f"{mu[2]}")
    # print(f"{mu[3]}")
    # The np.cov function expects each row to represent a variable (feature) 
    # and each column to represent an observation. 
    # However, the data points in x are arranged such that each row is an 
    # observation and each column is a variable. 
    # To align with np.cov's expectation, the data matrix is transposed using .T. 
    sigma = [np.cov(x[assignments == k, :].T) for k in range(K)]
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, 1 / K)
    # print(f"phi.shape = {phi.shape}")  (4,)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n, K), 1 / K)

    # print(f"sigma.len = {len(sigma)}")
    # print(f"phi.len = {len(phi)}")
    # print(f"mu.len = {len(mu)}")

    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        # print(f"it = {it}")
        for j in range(K):
            w[:, j] = phi[j] * multivariate_normal.pdf(x, mean=mu[j], cov=sigma[j])
            # if j == 1:
            #     print(f"w[:, j] = {w[:, j]}")
            # print(f"w.shape = {w.shape}") (980, 4)
        # print(f"w = {np.sum(w, axis=1, keepdims=True)}")
        print(f"w before = {w}")
        w /= np.sum(w, axis=1, keepdims=True)
        print(f"w before = {w}")

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            responsibility = w[:, j]
            # print(f"responsibility.shape = {responsibility.shape}")
            total_responsibility = np.sum(responsibility)
            # print(f"total_responsibility = {total_responsibility}")
            mu[j] = np.sum(responsibility[:, np.newaxis] * x, axis=0) / total_responsibility
            # print(f"mu[j].shape = {mu[j].shape}") # (2,)
            # print(f"x.shape = {x.shape}") # (980, 2)
            # print(f"np.sum(responsibility * np.dot((x - mu[j]), (x - mu[j]).T)) = {np.sum(responsibility * np.dot((x - mu[j]), (x - mu[j]).T)).shape}")
            sigma[j] = (np.dot(responsibility * (x - mu[j]).T, (x - mu[j]))) / total_responsibility
            print(f"sigma[j] = {sigma[j]}")
            phi[j] = total_responsibility / x.shape[0]

        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        # print(w)
        ll = np.sum(np.sum(np.log(w)))
        # print(f"prev_ll = {prev_ll}, ll ={ll}")
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
    
        it = it + 1
        
        # *** END CODE HERE ***
    print(f"run_em, converge at it = {it}, ll = {ll}")
    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        for j in range(K):
            w[:, j] = phi[j] * multivariate_normal.pdf(x, mean=mu[j], cov=sigma[j])
            # print(f"w.shape = {w.shape}") (980, 4)
        w /= np.sum(w, axis=1, keepdims=True)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            # For unlabeled data
            responsibility_unlabeled = w[:, j]
            total_responsibility_unlabeled = np.sum(responsibility_unlabeled)
            
            # For labeled data
            responsibility_labeled = (z_tilde == j).flatten()
            total_responsibility_labeled = np.sum(responsibility_labeled)
            
            # Combine labeled and unlabeled
            total_responsibility = total_responsibility_unlabeled + alpha * total_responsibility_labeled
            mu[j] = (np.sum(responsibility_unlabeled[:, np.newaxis] * x, axis=0) + alpha * np.sum(responsibility_labeled[:, np.newaxis] * x_tilde, axis=0)) / total_responsibility
            sigma[j] = (np.dot((responsibility_unlabeled * (x - mu[j]).T), (x - mu[j])) + alpha * np.dot((responsibility_labeled * (x_tilde - mu[j]).T), (x_tilde - mu[j]))) / total_responsibility
            phi[j] = total_responsibility / (x.shape[0] + alpha * x_tilde.shape[0])


        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll_unlabeled = np.sum(np.sum(np.log(w)))
        ll_labeled = np.sum(np.log(phi[z_tilde.flatten().astype(int)]) + [multivariate_normal.logpdf(x_tilde[i], mean=mu[int(z_tilde[i])], cov=sigma[int(z_tilde[i])]) for i in range(len(z_tilde))])
        ll = ll_unlabeled + alpha * ll_labeled
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        it = it + 1
        # *** END CODE HERE ***
    print(f"run_semi_supervised_em, converge at it = {it}")
    return w


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***

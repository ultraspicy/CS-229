import numpy as np
import matplotlib.pyplot as plt
import argparse

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_x
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1 / (1 + np.exp(-x))
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    # The function `np.random.randn`` from NumPy generates samples from 
    # the standard normal distribution, which is a specific type of 
    # normal (or Gaussian) distribution N(0,1)
    W1 = np.random.randn(input_size, num_hidden)
    b1 = np.zeros((num_hidden,))
    W2 = np.random.randn(num_hidden, num_output)
    b2 = np.zeros((num_output,))
    
    # Return the parameters as a dictionary
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # *** END CODE HERE ***

def forward_prop(data, one_hot_labels, params, verbose = False):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    W1 = params['W1']
    b1 = params['b1'].reshape([-1, 1])
    W2 = params['W2']
    b2 = params['b2'].reshape([-1, 1])

    z1 = W1.T @ data.T + b1 # 300, 1000
    a1 = sigmoid(z1)
    # at this point, z2 is 2D matrix, each column is a score vector
    # score vector has 10 number, for 10 classes
    # z2 has dimension of (#classes, batch_size) 
    z2 = W2.T @ a1 + b2 #(10, 1000)
    # probability_of_classes has dimension of (batch_size, # of classes)
    # each row is a row vector, each element of it is a probability of corresponding class
    probability_of_classes = softmax(z2.T) # (1000, 10)
    J = - np.mean(np.sum(one_hot_labels * np.log(probability_of_classes), axis = 1))

    if verbose:
        # print(f"a1 = {a1}")
        print(f"z2.shape = {z2.shape}")
        # print(f"z2 = {z2}")
        print(f"probability_of_classes = {probability_of_classes.shape}")
        
    # print(f"J = {J}")
    return a1, probability_of_classes, J
    # *** END CODE HERE ***

def backward_prop(data, one_hot_labels, params, forward_prop_func, verbose = False):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    batch_size = data.shape[0]
    # get all specs of forward propagation 
    a1, h, loss = forward_prop_func(data, one_hot_labels, params)
    # z_2 is the score vector that has dimension of (# of classes, batch_size)
    # patial_loss_over_z_2 and one_hot_labels has dimension (batch_size, # of classes)
    # each row is a loss vector, with each element as the loss for its class
    patial_loss_over_z_2 = h - one_hot_labels

    # gradient of the previous activated input which will be used for computing 
    # the gradient of the previous unactivated input
    patial_loss_over_a_1 = W2 @ patial_loss_over_z_2.T # dimension [# of feature, # of sample]

    ### =================================================================
    # num_of_sample = data.shape[0]
    # patial_loss_over_w_2 = np.zeros(W2.shape) # 300, 10
    # for i in range(num_of_sample):
    #     # print(f"forward_prop['a1'][:, i] = {forward_prop['a1'][:, i].shape}")
    #     # print(f"patial_loss_over_z_2[i, :] = {patial_loss_over_z_2[i, :].shape}")
    #     patial_loss_over_w_2 = patial_loss_over_w_2 + a1[:, i].reshape((-1, 1)) @ patial_loss_over_z_2[i, :].reshape((1, -1))
    # patial_loss_over_w_2 = 1 / num_of_sample * patial_loss_over_w_2
    patial_loss_over_w_2 = a1 @ patial_loss_over_z_2 / batch_size # 300, 10
    ### =================================================================
    patial_loss_over_b_2 = np.sum(patial_loss_over_z_2, axis = 0) / batch_size # (10, )
    patial_loss_over_z_1 = a1 * (1 - a1) * patial_loss_over_a_1
    # print(f"patial_loss_over_z_1.shape = {patial_loss_over_z_1.shape}") # 300, 1000
    ### =================================================================
    # patial_loss_over_w_1 = np.zeros(W1.shape) # (784, 300)
    # for i in range(num_of_sample):
    #     patial_loss_over_w_1 = patial_loss_over_w_1 + data[i,:].reshape((-1, 1)) @ patial_loss_over_z_1[:, i].reshape((1, -1))
    # patial_loss_over_w_1 = 1 / num_of_sample * patial_loss_over_w_1
    patial_loss_over_w_1 = patial_loss_over_z_1 @ data / batch_size
    # print(f"patial_loss_over_w_1.shape = {patial_loss_over_w_1.shape}") # 300, 784
    ### =================================================================
    patial_loss_over_b_1 = np.sum(patial_loss_over_z_1, axis = 1)  / batch_size # (300,)
    
    if verbose:
        # print(f"a1 = {a1}") # 1000, 10
        # print(f"a1.shape = {a1.shape}") # 300, 1000
        print(f"patial_loss_over_z_2[1,:] = {patial_loss_over_z_2[1,:]}") # 1000, 10
        print(f"patial_loss_over_z_2.shape = {patial_loss_over_z_2.shape}") # 1000, 10
        print(f"patial_loss_over_a_1.shape = {patial_loss_over_a_1.shape}")
        print(f"patial_loss_over_w_2 = {patial_loss_over_w_2.shape}") 
        print(f"patial_loss_over_b_2 = {patial_loss_over_b_2.shape}")
        print(f"patial_loss_over_z_1 = {patial_loss_over_z_1.shape}") # 300, 1000
        print(f"patial_loss_over_b_1 = {patial_loss_over_b_1.shape}")
    return {'W1': patial_loss_over_w_1.T, 'b1': patial_loss_over_b_1, 'W2': patial_loss_over_w_2, 'b2':patial_loss_over_b_2}
    # *** END CODE HERE ***


def backward_prop_regularized(data, one_hot_labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        one_hot_labels: A 2d numpy array containing the the one-hot embeddings of the labels e_y.
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    grad_params = backward_prop(data, one_hot_labels, params, forward_prop_func)
    
    grad_params['W2'] += 2 * reg * params['W2']
    grad_params['W1'] += 2 * reg * params['W1']
    
    return grad_params
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, 
                           one_hot_train_labels, 
                           learning_rate, 
                           batch_size, 
                           params, forward_prop_func,
                           backward_prop_func,
                           epoch,
                           verbose = False):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        one_hot_train_labels: A numpy array containing the one-hot embeddings of the training labels e_y.
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    # create mini-batches sequentially
    iteration =  train_data.shape[0] // batch_size
    for i in range(iteration):
        train_data_mini_batch = train_data[i*batch_size : (i + 1)*batch_size,:]
        one_hot_train_labels_mini_batch = one_hot_train_labels[i*batch_size : (i + 1)*batch_size,:]
        # run backward propagation to get the gradiant
        grad = backward_prop_func(train_data_mini_batch, 
                                  one_hot_train_labels_mini_batch, 
                                  params, 
                                  forward_prop_func)
        ## update gradient
        params['W1'] = params['W1'] - learning_rate * grad['W1'] 
        params['b1'] = params['b1'] - learning_rate * grad['b1'] 
        params['W2'] = params['W2'] - learning_rate * grad['W2'] 
        params['b2'] = params['b2'] - learning_rate * grad['b2'] 
    # print out the result  
    if verbose:
        print(f"W1 = {grad['W1']}")
        print(f"b1 = {grad['b1']}")
        print(f"W2 = {grad['W2']}")
        print(f"b2 = {grad['b2']}")
    # *** END CODE HERE ***

    # This function does not return anything
    return
'''
params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )
'''
def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func, epoch)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

'''
compute the accuracy of NN using test set
return value is a float number < 1
'''
def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy
'''
racial of the max number in `output` is the corresponding label
the expression * 1. is used to ensure that the result of the division 
is a floating-point number (float) rather than an integer (int).
'''
def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy
'''
covert vector of labels into a list of one-hot row vectors
each row is a one-hot presentation
'''
def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels
'''
np.loadtxt() is used to load data from a text file, with each row in the file corresponding 
to a row in the resulting array, and each number in a row being separated by a specific 
delimiter (which is whitespace by default).
'''
def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

'''
train the model 
save the plot
baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
'''
def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    # train the NN model
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    # save the learnt parameters (i.e., all the weights and biases)
    # into a file, so that next time you can directly initialize the 
    # parameters with these values from the file
    filenames = {'W1': f"nn_w1_{name}.txt", 'b1': f"nn_b1_{name}.txt", 
                 'W2': f"nn_w2_{name}.txt", 'b2': f"nn_b2_{name}.txt"}
    np.savetxt(filenames['W1'], params['W1'])
    np.savetxt(filenames['b1'], params['b1'])
    np.savetxt(filenames['W2'], params['W2'])
    np.savetxt(filenames['b2'], params['b2'])

    # The np.arange function in NumPy is used to generate arrays containing
    # evenly spaced values within a given interval. 
    #When you use np.arange(num_epochs), it creates an array that starts from 0 and 
    # ends at num_epochs - 1, with a step size of 1 by default.
    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    # convert labels to one-hot embeddings e_y.
    train_labels = one_hot_labels(train_labels)
    # split 10000 samples randomly from 60000 samples in training data
    # process shuffle, then first 10k is dev_data, later 50k is training data 
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]
    # normalization
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    # convert labels to one-hot embeddings e_y.
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
        
    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()

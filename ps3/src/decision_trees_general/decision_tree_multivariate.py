import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DecisionTree:
    def __init__(self, max_depth=None):
        # Initialize the Decision Tree with an optional max depth parameter.
        self.max_depth = max_depth
        
    def fit(self, X, y):
        # Fit the decision tree model to the training data.
        self.n_classes_ = len(np.unique(y))  # Calculate the number of unique classes.
        self.n_features_ = X.shape[1]  # Number of features.
        self.tree_ = self._grow_tree(X, y)  # Build the decision tree. Return the root node.
        
    def predict(self, X):
        # Predict class labels for samples in X.
        return [self._predict(inputs) for inputs in X]
    '''
    given a vector, the most frequent element is the "correct classification"
    all other classes are misclassified
    the return value is # of the misclassified 
    ''' 
    def _misclassification_loss(self, y):
        # TODO: Implement the misclassification loss function.
        loss = None
        # *** START CODE HERE ***
        # Find unique values and their frequencies
        _, value_counts = np.unique(y, return_counts=True)
        # Find the index of the most frequent item
        max_freq_index = np.argmax(value_counts)
        # compute loss
        loss = len(y) - value_counts[max_freq_index]
        # *** END YOUR CODE ***
        # Return the misclassification loss.
        return loss

    '''
    X - training data
    y - training labels
    return the index of best feature from all d features, and the threshold 
    ''' 
    def _best_split(self, X, y):
        # TODO: Find the best split for a node.
        # Hint: Iterate through all features and calculate the best split threshold based on the misclassification loss.
        # Hint: You might want to loop through all the unique values in the feature to find the best threshold.
        best_idx, best_thr = None, None
        # *** START CODE HERE ***        
        # Calculate the parent's loss to compare with splits
        best_loss = np.inf
        for feature_idx in range(self.n_features_):
            # print(f"{X[:, feature_idx]}")
            thresholds = self.find_midpoints(np.unique(X[:, feature_idx]))
            for threshold in thresholds:
                # print(f"current threshold {threshold}")
                # indices that LE to threshold
                left_indices = [i for i, x in enumerate(X[:, feature_idx]) if x <= threshold]
        
                # indices that greater than the threshold
                right_indices = [i for i, x in enumerate(X[:, feature_idx]) if x > threshold]
            
                left_loss = self._misclassification_loss(y[left_indices])
                right_loss = self._misclassification_loss(y[right_indices])
                total_loss = left_loss + right_loss
                
                # all classes are prefectly split
                if total_loss == 0:
                    print(f"best_loss = 0")
                    return None, None
                # there are still some loss
                if total_loss < best_loss:
                    best_idx, best_thr = feature_idx, threshold
                    best_loss = total_loss
        # *** END YOUR CODE ***
        print(f"best_loss = {best_loss}")
        # Return the best split with the feature index and threshold.           
        return best_idx, best_thr
        
    def _grow_tree(self, X, y, depth=0):
        # Build a decision tree by recursively finding the best split.
        # param depth is the current depth of the tree.
        # Construct the root node.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        root_node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            # TODO: Find the best split using _best_split and grow the tree recursively.
            # *** START YOUR CODE ***
            print(f"================================================")
            best_idx, best_thr = self._best_split(X, y)
            print(f"depth = {depth}")
            print(f"best_idx = {best_idx}")
            print(f"best_thr = {best_thr}")
            # if best_idx is None, we already get 0 loss
            if best_idx is None:
                return root_node
            
            left_indices = [i for i, x in enumerate(X[:, best_idx]) if x <= best_thr]
            right_indices = [i for i, x in enumerate(X[:, best_idx]) if x > best_thr]
            
            # print(f"X[left_indices,:] {X[left_indices,:]}")
            # print(f"X[right_indices,:] {X[right_indices,:]}")       
            left_subtree = self._grow_tree(X[left_indices,:], y[left_indices], depth + 1)
            right_subtree = self._grow_tree(X[right_indices,:], y[right_indices], depth + 1)
            root_node.feature_index = best_idx
            root_node.threshold = best_thr
            root_node.left = left_subtree
            root_node.right = right_subtree
            # *** END YOUR CODE ***
        # Return the root node.
        return root_node
        
    def _predict(self, inputs):
        # Predict the class of ONE input based on the tree structure.
        node = self.tree_
        while not node.is_leaf_node():
            # TODO: Traverse the tree to find the corresponding node and predict the class of the input.
            # Hint: iteratively update the node to be its left or right child until a leaf node is reached.
            # *** START YOUR CODE ***
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
            # *** END YOUR CODE ***
        return node.predicted_class
    
    '''
    convert an array into an array of mid points
    '''
    def find_midpoints(self, arr):
        # Sort the input array
        sorted_arr = sorted(arr)
        
        # Initialize an empty list to store midpoints
        midpoints = []
        
        # Calculate midpoints between adjacent elements
        for i in range(len(sorted_arr) - 1):
            midpoint = (sorted_arr[i] + sorted_arr[i + 1]) / 2.0
            midpoints.append(midpoint)
        
        return midpoints
    
class Node:
    def __init__(self, *, predicted_class):
        self.predicted_class = predicted_class  # Class predicted by this node
        self.feature_index = None  # Index of the feature used for splitting
        self.threshold = None  # Threshold value for splitting
        self.left = None  # Left child
        self.right = None  # Right child

    def is_leaf_node(self):
        # Check if this node is a leaf node.
        return self.left is None and self.right is None


if __name__ == "__main__":
    data = pd.DataFrame({
        "Age": [24, 53, 23, 25, 32, 52, 22, 43, 52, 48], 
        "Salary": [40, 52, 25, 77, 48, 110, 38, 44, 27, 65], 
        "College": [1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    })
    # Create a simple dataset for testing the decision tree
    X = np.array(data[["Age", "Salary"]]) # (10, 2)
    y = np.array(data["College"])
    
    # Initialize and fit the decision tree
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)
    
    # Print the classification accuracy
    print(f"Accuracy for college degree dataset: {round(np.mean(clf.predict(X) == y)*100, 2)}%")
    
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # print("Multi-class labels", np.unique(y))
    
    # Split the data into training and testing sets
    # DO NOT MODIFY THE RANDOM STATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=229)

    # Train the decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = tree.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy for iris dataset: {round(accuracy*100, 2)}%")

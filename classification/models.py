import random

from scipy import stats
import numpy as np
import pandas as pd


class RandomForestClassifier:

    def __init__(self, n_estimators=10, subsample_size=None, max_depth=None, max_features=None, bootstrap=True,
                 random_state=None):
        self.num_trees = n_estimators
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_features = None
        # Will store individually trained decision trees
        self.decision_trees = []

    def sample(self, X, y, random_state):

        n_rows, n_cols = X.shape

        # Sample with replacement
        if self.subsample_size is None:
            sample_size = n_rows
        else:
            sample_size = int(n_rows * self.subsample_size)

        if random_state is not None:
            np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=sample_size, replace=self.bootstrap)

        return X[samples], y[samples]

    def init(self, X, y, max_features, max_depth):
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_features = int(X.shape[1])

        if max_features is None or max_features == 'None' or max_features == 'none':
            self.max_features = self.n_features
        elif max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.n_features))
        elif max_features < 1:
            self.max_features = int(self.n_features * max_features)
        else:
            self.max_features = int(max_features)

        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        # Build each tree of the forest
        num_built = 0
        random_state = self.random_state

        while num_built < self.num_trees:
            _X, _y = self.sample(X, y, random_state)
            random.seed(random_state)
            max_features = self.max_features
            n_features = self.n_features
            feat_indices = random.sample(range(n_features), max_features)
            node = Node(_X, _y, depth=0, feat_indices=feat_indices)
            n_classes = len(set(_y))
            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=random_state,
                X=_X,
                y=_y,
                tree=node,
                n_classes=n_classes
            )

            self.decision_trees.append(tree)

            num_built += 1

            if random_state is not None:
                random_state += 1

    def predict(self, X):

        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))

        # Reshape so we can find the most common value
        y = np.swapaxes(y, axis1=0, axis2=1)

        # Use majority voting for the final prediction (added keepdims=True)
        predicted_classes = stats.mode(y, axis=1, keepdims=True)[0].reshape(-1)

        return predicted_classes


class DecisionTree:

    def __init__(self, X=None, y=None, max_depth=None, max_features=None, random_state=None, tree=None, n_classes=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.X = X
        self.y = y
        self.tree = tree
        self.n_classes = n_classes

    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.values

        predicted_classes = np.array([self.predict_example(inputs) for inputs in X])

        return predicted_classes

    def predict_example(self, inputs):

        node = self.tree

        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class


class Node:

    def __init__(self, X, y, depth, feat_indices):
        self.feature_index = 0
        self.threshold = 0
        self.gini_index = None
        self.left = None
        self.right = None
        self.X = X
        self.y = y
        self.depth = depth
        self.predicted_class = None
        self.feat_indices = feat_indices

    def get_leaf_nodes(self):
        """Get the nodes which don't have any left of right sub nodes"""
        leaves = []

        # If no child nodes
        if not self.left and not self.right:
            return [self]

        # If no any left child
        if not self.left:
            leaves = self.right.get_leaf_nodes()

        # If no any right child
        if not self.right:
            leaves = self.left.get_leaf_nodes()

        # If has left as well left child
        if self.left and self.right:
            leaves = self.left.get_leaf_nodes() + self.right.get_leaf_nodes()

        return leaves

    def get_leaf_nodes_of_depth(self, depth: int):
        leaves = self.get_leaf_nodes()
        leaf_nodes = []
        for leaf in leaves:
            if leaf.depth == depth:
                leaf_nodes.append(leaf)

        return leaf_nodes

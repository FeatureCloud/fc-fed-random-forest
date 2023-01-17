from scipy import stats
import numpy as np
import pandas as pd


class RandomForestClassifier:

    def __init__(self, n_estimators=100, subsample_size=None, max_depth=None, max_features=None, bootstrap=True,
                 random_state=None):
        self.num_trees = n_estimators
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        # Will store individually trained decision trees
        self.decision_trees = []

    def sample(self, X, y, random_state):

        n_rows, n_cols = X.shape

        # Sample with replacement
        if self.subsample_size is None:
            sample_size = n_rows
        else:
            sample_size = int(n_rows * self.subsample_size)

        np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=sample_size, replace=self.bootstrap)

        return X[samples], y[samples]

    def init(self, X, y, n_features, max_features, max_depth):
        self.n_features = n_features
        self.max_features = max_features
        self.max_depth = max_depth

        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        # Build each tree of the forest
        num_built = 0

        random_state = self.random_state

        while num_built < self.num_trees:
            _X, _y = self.sample(X, y, random_state)
            node = Node(X, y, depth=0)
            n_classes = len(set(_y))
            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state,
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

    def __init__(self, X, y, depth):
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.X = X
        self.y = y
        self.depth = depth
        self.predicted_class = None

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
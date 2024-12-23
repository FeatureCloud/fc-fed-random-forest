from typing import Optional, Union, Dict, Any, List, Iterator

import numpy as np
from scipy import stats
# pylint: disable=too-many-instance-attributes, invalid-name

class RandomForest:
    """
    A random forest model for classification and regression to be used for federated histogram
    based random forest learning and prediction.
    """
    def __init__(self,
                 n_estimators: int,
                 max_samples: float,
                 feat_idcs: np.ndarray, # List of feature indices for each tree,
                                        # n_estimators x n_features
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 n_patients_local: int, # Number of patients in the local dataset
                 n_patients_global: int, # Number of patients in the global, aggregated dataset
                 bootstrap: bool,
                 random_state: int,
                 quantile: np.ndarray, # List of feature indices using quantile binning
                 global_mean: np.ndarray, # Global mean for each feature
                 global_stddev: np.ndarray, # global standard deviation for each feature
                 split_points: np.ndarray, #  n_features x n_bins - 1,  #TODO: fix them
                 prediction_mode: str, # 'classification' or 'regression'
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                        # same order for all clients
                 oob: bool,
                 class_weights: Optional[Dict[Any, float]],
                    # weights of the classes (dict[class]=weight)
                ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feat_idcs = feat_idcs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_patients_local = n_patients_local
        self.n_patients_global = n_patients_global
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.decision_trees: List[DecisionTree] = []
        np.random.seed(random_state)
        self.quantile = quantile
        self.global_mean = global_mean
        self.global_stddev = global_stddev
        self.split_points = split_points
        self.prediction_mode = prediction_mode
        self.global_classes = global_classes
        self.class_weights = class_weights
        self.oob = oob
        self.finished = False

        # init the trees
        for _ in range(self.n_estimators):
            sample_idcs = self._bootstrap_samples()
            tree = DecisionTree(samples_idcs=sample_idcs,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                feat_idcs=self.feat_idcs,
                                mode=self.prediction_mode,
                                global_classes=self.global_classes,
                                class_weights=self.class_weights)
            self.decision_trees.append(tree)

    def _bootstrap_samples(self):
        sample_size = max(round(self.n_patients_local * self.max_samples), 1)
        sample_idcs = np.random.choice(self.n_patients_local, sample_size, replace=self.bootstrap)
        return sample_idcs

    def predict(self, X):
        #TODO: double check this, this funcction was not verified!!!
        bucket_idcs = np.setdiff1d(np.arange(len(X[0])), self.quantile)

        if len(bucket_idcs) > 0:
            # Bucket Binning
            bucket_split_points = self.split_points[bucket_idcs, :]

            X_T_bucket = np.transpose(X[:, bucket_idcs])
            # Assign data points to bins
            X_hist_bucket = np.array([np.digitize(X_T_bucket[i], bucket_split_points[i]) \
                                        for i in range(X_T_bucket.shape[0])]) - 1

        if len(self.quantile) > 0:
            # Quantile Binning
            a = (X[:, self.quantile] - self.global_mean)
            b = self.global_stddev
            normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            normalized[normalized == np.inf] = 0
            normalized[normalized == -np.inf] = 0
            normalized[normalized == np.nan] = 0

            quantile_split_points = self.split_points[self.quantile, :]
            X_T_quantile = np.transpose(normalized)
            # Assign data points to bins
            X_hist_quantile = np.array([np.digitize(X_T_quantile[i], quantile_split_points[i]) \
                                        for i in range(X_T_quantile.shape[0])])

        if len(bucket_idcs) > 0 and len(self.quantile) > 0:
            X_hist = np.concatenate((X_hist_quantile, X_hist_bucket))
            # Place the values of array at specified indices
            X_hist[self.quantile] = X_hist_quantile
            X_hist[bucket_idcs] = X_hist_bucket
            X_hist = np.transpose(X_hist)

        elif len(bucket_idcs) > 0:
            X_hist = np.transpose(X_hist_bucket)

        else:
            X_hist = np.transpose(X_hist_quantile)

        # Make predictions with every tree in the forest
        y = np.array([tree.predict(X_hist) for tree in self.decision_trees])
        # Reshape so we can find the most common value
        y = np.swapaxes(y, axis1=0, axis2=1)

        if self.prediction_mode == 'classification':
            if not self.oob:
                # Use majority voting for the final prediction
                predicted_values = stats.mode(y, axis=1, keepdims=True)[0].reshape(-1)
            else:
                predicted_values = []
                classes = np.unique(y)
                for i in range(len(X_hist)):
                    counter = []
                    for c in classes:
                        indices = np.where(y[i] == c)[0]
                        counter.append(np.sum([self.decision_trees[j].weight for j in indices]))
                    predicted_values.append(classes[np.argmax(counter)])
        else:
            predicted_values = np.mean(y, axis=0)

        return predicted_values

    def iterate_trees(self) -> Iterator["DecisionTree"]:
        """
        Iterate over the decision trees in the random forest.
        """
        for tree in self.decision_trees:
            yield tree

    def get_split_scores(self, X_Hist: np.ndarray, y: np.ndarray, n_bins: int) -> List[List[List[List[float]]]]:
        """
        Calculate the score of a split for each feature and bin. Works on the cur_depth_nodes.
        If they already have an assigned feature and threshold, an error is thrown.

        Args:
            X_hist: input data, 2d array of shape (n_samples, n_features).
                The values are NOT the actual values but the bin indices the samples belong to
                for each feature.
            y: target data, 1d array of shape (n_samples), indicating the target value
                for each sample
            n_bins: number of bins to consider for each feature.

        Returns:
            scores: 4d array of shape (n_estimators, num_nodes_cur_level,
            len(self.feat_idcs), n_bins), containing the score of a split per feature and bin.
        """
        scores = []
        for tree in self.decision_trees:
            tree_scores = tree.get_split_scores(X_Hist, y, n_bins)
            scores.append(tree_scores)
        return scores


class DecisionTree:
    """
    A decision tree model for classification and regression to be used for federated histogram
    based random forest learning and prediction.
    """
    def __init__(self,
                 samples_idcs: np.ndarray, # 1d array of indices of the samples to be used
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 feat_idcs: np.ndarray, # List of feature indices for each tree,
                                        # n_estimators x n_features
                 mode: str, # classification or regression
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                        # same order for all clients
                 class_weights: Optional[Dict[Any, float]],
                    # weights of the classes (dict[class]=weight)
                 ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feat_idcs = feat_idcs
        self.finished = False
        self.leaves = []
        self.weight = 1
        self.mode = mode
        self.global_classes = global_classes
        self.class_weights = class_weights
        if self.mode == 'regression' and self.class_weights is not None:
            raise ValueError('Weights are not supported for regression.')

        # init the root node
        self.root = Node(depth=0,
                         sample_idcs=samples_idcs,
                         mode=mode,
                         global_classes=global_classes,
                         class_weights=class_weights,
                         feature_idcs=feat_idcs)

        self.cur_depth_nodes = [self.root]

    def predict(self,
                X):
        """
        Predict the target values for the input data X (2d array).
        Assumes the same features with the same indices as the training data.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def iterate_cur_depth_nodes(self) -> Iterator["Node"]:
        """
        Iterate over the nodes at the current depth.
        """
        for node in self.cur_depth_nodes:
            yield node

    def get_split_scores(self, X_hist: np.ndarray, y: np.ndarray, n_bins: int) -> List[List[List[float]]]:
        """
        Calculate the score of a split for each feature and bin. Works on the cur_depth_nodes.
        If they already have an assigned feature and threshold, an error is thrown.

        Args:
            X_hist: input data, 2d array of shape (n_samples, n_features).
                The values are NOT the actual values but the bin indices the samples belong to
                for each feature.
            y: target data, 1d array of shape (n_samples), indicating the target value
                for each sample
            n_bins: number of bins to consider for each feature.

        Returns:
            scores: 3d array of shape (num_nodes_cur_level, len(self.feat_idcs), n_bins),
            containing the score of a split per feature and bin.
        """
        scores = []
        for node in self.cur_depth_nodes:
            node_scores = node.get_split_scores(X_hist, y, n_bins)
            scores.append(node_scores)
        return scores

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to find the leaf node for the input data x, effectively making a prediction.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class Node:
    """
    A single decision tree Node.
    """
    def __init__(self,
                 depth: int,
                 sample_idcs: np.ndarray, # 1d array of indices of the samples to be used
                 mode: str, # classification or regression
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                             # same order for all clients
                 feature_idcs: np.ndarray, # List of feature indices for each tree,
                 class_weights: Optional[Dict[Any, float]],
                    # weights of the classes: (dict[class]=weight)
                 feature: Optional[int] = None, # the feature index used in this node
                 threshold: Optional[Union[float, int]] = None,
                    # the threshold whether to go left or right. The decision is x <= threshold
                    #TODO: right? Wait maybe we need the info if we're right or left
                 score: Optional[float] = None, # the score (e.g. gini impurity) of the split
                 parent: Optional["Node"] = None, # pointer to the parent node
                 left: Optional["Node"] = None, # pointer to the left child
                 right: Optional["Node"] = None, # pointer to the right child
                 global_leaf: bool = False, # whether the node is a leaf globally
                 local_leaf: bool = False, # whether the node is a leaf locally #TODO: how is the local/global leaf determined?
                 value=None #TODO: what is the value?
                 ) -> None:
        self.depth = depth
        self.sample_idcs = sample_idcs
        self.feature = feature
        self.threshold = threshold
        self.score = score
        self.parent = parent
        self.left = left
        self.right = right
        self.global_leaf = global_leaf
        self.local_leaf = local_leaf
        self.value = value
        self.global_classes = global_classes
        self.feature_idcs = feature_idcs
        if len(set(global_classes)) != len(global_classes):
            raise ValueError('Classes must be unique.')
        self.num_classes = len(global_classes)
        if mode not in ['classification', 'regression']:
            raise ValueError('Mode must be either classification or regression.')
        self.mode = mode
        for cl in global_classes:
            if cl not in class_weights:
                raise ValueError('Class weights must be given for all classes or not at all')
        self.class_weights = class_weights

    def is_leaf_node(self):
        """
        Whether the node is GLOBALLY considered a leaf node.
        """
        return self.global_leaf

    def get_split_scores(self, X_hist: np.ndarray, y: np.ndarray, n_bins: int) -> List[List[float]]:
        """
        Calculate the score of a split for each feature and bin.
        Throws an error if the node alreadt has a set feature and threshold.

        Args:
            X_hist: input data, 2d array of shape (n_samples, n_features).
                The values are NOT the actual values but the bin indices the samples belong to
                for each feature.
            y: target data, 1d array of shape (n_samples), indicating the target value
                for each sample
            n_bins: number of bins to consider for each feature.

        Returns:
            scores: 2d array of shape (len(self.feature_idcs), n_bins), containing the score of a split per
                feature and bin.
        """
        if self.feature is not None or self.threshold is not None:
            raise ValueError('Node already has a split.')
        # ensure input data formatting
        if len(X_hist.shape) != 2:
            raise ValueError('X_hist must be a 2d array.')
        if X_hist.shape[1] != len(y):
            raise ValueError('X_hist and y must have the same number of samples.')
        if len(y.shape) != 1:
            raise ValueError('y must be a 1d array.')
        node_data = X_hist[self.sample_idcs]
        scores = []
            # num_features x num_bins
        for feature_idx in self.feature_idcs:
            feature_data = node_data[:, feature_idx]
            scores_per_bin = []
                # array of length num_bins containing the scores per_bin
            for bin_idx in range(n_bins):
                # we split the data into <= threshold and > threshold
                left_idxs = np.where(feature_data <= bin_idx)[0]
                    # feature_data <= bin_idx returns a boolean array
                    # np.where then returns the indices of the True values
                    # we need to add [0] as np.where always returns a tuple
                right_idxs = np.where(feature_data > bin_idx)[0]
                left_y = y[left_idxs]
                right_y = y[right_idxs]
                if self.mode == 'classification':
                    # Each node is a binary split, we just check that the impurity is minimized
                    score = self._gini_split_score(left_y, right_y)
                else: # regression
                    score = self._mse_split_score(left_y, right_y)
                scores_per_bin.append(score)
            scores.append(scores_per_bin)
        return scores

    def _gini_split_score(self, left_y: np.ndarray, right_y: np.ndarray) \
            -> np.floating:
        """
        Calculate the gini impurity of one specific split. This is done by calculating the
        gini impurity of the left and right node and then weighting them by the number of samples
        that each child node would contain.
        Formula is:
            gini_split = #samples_left / #samples_total_split * gini_left +
                #samples_right / #samples_total_split * gini_right
            gini_left and right are calculated via the gini impurity formula:
            gini_impurity = 1 - sum(p_i^2), where p_i is the probability of class i
        practically, this is calculated as:
            gini_impurity = 1 - sum_class_i((#samples_of_class_i / #samples)^2)
            This is done for both the left and right node, also only using the samples that
            would end up in the respective node!!

        if weights are given, then we go from each sample being represented by a one
        to each sample being represented by it's weight.
        This changes especially the calculations that previously only took counts of samples into
        account. Instead, they now use the accumulated weights of samples.
        1. The gini impurity calculation changes:
            gini_impurity = 1 -
                sum_class_i((sum_weight_of_samples_of_class_i / sum_weight_of_all_samples)^2)
        2. The gini score formula changes:
            gini = sum_weight_of_samples_left / sum_weight_of_samples * gini_left +
                sum_weight_of_samples_right / sum_weight_of_samples * gini_right

        Args:
            left_y: target values of the samples that would end up in the left node
            right_y: target values of the samples that would end up in the right node

        Returns:
            gini: the gini impurity of the split
        """
        if len(left_y.shape) != 1 or len(right_y.shape) != 1:
            raise ValueError('y must be a 1d array.')
        left_weights = np.ones((len(left_y))) if self.class_weights is None else np.vectorize(self.class_weights.get)(left_y)
            # either just one for any sample or the weight of the sample by their class
            # np.vectorize(self.class_weights.get)(left_y) runs weights.get on each element
            # of left_y constructing an np.array. we therefore get an np.array of length samples
            # with the weight for each sample as values.
        right_weights = np.ones((len(right_y))) if self.class_weights is None else np.vectorize(self.class_weights.get)(right_y)
        total_left = np.sum(left_weights)
        total_right = np.sum(right_weights)
        total_y = total_left + total_right

        gini_left = 1.0 - np.sum((np.bincount(left_y.astype('int'), minlength=self.num_classes, weights=left_weights) / total_left) ** 2)
            # with np.bincount we get an array of length n_classes with the index being the specific
            # class. The values are the added weights of the samples of each class.
            # if self.class_weights is None, we just add one for each sample, so we have the
            # counts of samples of each class
            # we then divide by the total weight/total number of samples
            # to get the probability of each class
        gini_right = 1.0 - np.sum((np.bincount(right_y.astype('int'), minlength=self.num_classes, weights=right_weights) / total_right) ** 2)
        gini = (total_left / total_y) * gini_left + (total_right / total_y) * gini_right
            # we add the two gini impurities weighted by the number of samples/weights by total weight
        return gini

    def _mse_split_score(self, left_y: np.ndarray, right_y: np.ndarray) -> np.floating:
        """
        Calculates the mean squared error of a split. This is done similarly to the gini impurity
        split score. We calculate the mean squared error of the left and right node and then weight
        them by the number of samples that each child node would contain.
        Formula is:
            mse_split = #samples_left / #samples_total_split * mse_left +
                #samples_right / #samples_total_split * mse_right
            mse_left and right are calculated via the mse formula:
            mse = 1/n * sum((y_node - mean(y_node))^2), where y_node are the target values of the
                samples in the right/left node.
        Weights are not supported for regression.

        Args:
            left_y: target values of the samples that would end up in the left node
            right_y: target values of the samples that would end up in the right node

        Returns:
            mse: the mean squared error of the split
        """
        if len(left_y.shape) != 1 or len(right_y.shape) != 1:
            raise ValueError('y must be a 1d array.')
        if self.class_weights is not None:
            raise ValueError('Weights are not supported for regression.')

        total_left = len(left_y)
        total_right = len(right_y)
        total_y = total_left + total_right

        mse_left = np.mean((left_y - np.mean(left_y)) ** 2)
            # np.mean(x) is the same then np.sum(x) / total_left
        mse_right = np.mean((right_y - np.mean(right_y)) ** 2)
        mse = (total_left / total_y) * mse_left + (total_right / total_y) * mse_right
        return mse

from typing import Optional, Union, Dict, Any, List, Iterator, Tuple, Generator

import numpy as np
import joblib
# pylint: disable=too-many-instance-attributes, invalid-name

class RandomForest:
    """
    A random forest model for classification and regression to be used for federated histogram
    based random forest learning and prediction.
    """
    def __init__(self,
                 n_estimators: int,
                 max_samples: float,
                 feat_idcs: np.ndarray,
                 max_depth: int,
                 min_samples_split: int,
                 bootstrap: bool,
                 random_state: int,
                 quantile: np.ndarray,
                 global_mean: np.ndarray,
                 global_stddev: np.ndarray,
                 split_points: np.ndarray,
                 prediction_mode: str,
                 global_classes: np.ndarray,
                 oob: bool,
                 class_weights: Optional[Dict[Any, float]],
                 X_hist: np.ndarray,
                 y: np.ndarray,
                 num_bins: int,
                ) -> None:
        """
        The random forest model for federated learning. Sets the following parameters at initialization

        Args:
            n_estimators: number of trees in the forest
            max_samples: the maximum fraction of samples to use for each tree
            feat_idcs: List of feature indices for each tree, n_estimators x n_features
            max_depth: the maximum depth of the trees, e.g. 5 means at most 5 splits
            min_samples_split: the minimum number of samples required to split an internal node
            min_samples_leaf: the minimum number of samples required to be at a leaf node
            n_patients_local: number of patients in the local dataset
            n_patients_global: number of patients in the global, aggregated dataset
            bootstrap: whether to use bootstrap samples
            random_state: the random seed to use
            quantile: List of feature indices using quantile binning
            global_mean: global mean for each feature (n_estimators x n_quantile_features)
            global_stddev: global standard deviation for each feature (n_estimators x n_quantile_features)
            split_points: The threshold values for each feature.
                Dimensions are n_features x n_bins - 1 as the values min and max are excluded:
                    ]min, val1, ..., max[
                    A bin_idx of 0 would mean a threshold of <= split_points[0] -> left child,
                    A bin_idx of len(split_points-1) would mean a threshold of <= split_points[-1] -> left child
            prediction_mode: 'classification' or 'regression'
            global_classes: global classes. Should be the same in the same order for all clients
            oob: whether to use out-of-bag # TODO: finnish this description, should be used in the evaluation?
            class_weights: weights of the classes (dict[class]=weight)
                Set to None if no class weights are used and when the training finishes
            X_hist: input data, 2d array of shape (n_samples, n_features).
                The values are NOT the actual values but the bin indices the samples belong to
                This is used in the training process to calculate the split scores
                When the training is done, the data is removed
            y: target data, 1d array of shape (n_samples), indicating the target value
                This is used in the training process to calculate the split scores
                When the training is done, the data is removed
            n_bins: number of bins to consider for each feature.
        """
        # private vars
        self.__max_samples = max_samples
        self.__feat_idcs = feat_idcs
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__bootstrap = bootstrap
        self.__random_state = random_state
        self.__decision_trees: List[DecisionTree] = []
        np.random.seed(random_state)
        self.__quantile = quantile
        self.__global_mean = global_mean
        self.__global_stddev = global_stddev
        self.__split_points = split_points
        self.__global_classes = global_classes
        self.__class_weights = class_weights
        self.__oob = oob
        self.__X_hist = X_hist
        self.__y = y
        self.__num_bins = num_bins
        if num_bins-1 != split_points.shape[1]:
            raise ValueError('Number of bins must match number of split points.')
        self.finished = False

        # public vars
        self.prediction_mode = prediction_mode
        self.n_estimators = n_estimators

        # init the trees
        for tree_idx in range(self.n_estimators):
            sample_idcs = self.__bootstrap_samples(n_patients_local=len(y))
            tree = DecisionTree(samples_idcs=sample_idcs,
                                max_depth=self.__max_depth,
                                min_samples_split=self.__min_samples_split,
                                feat_idcs=self.__feat_idcs[tree_idx],
                                mode=self.prediction_mode,
                                global_classes=self.__global_classes,
                                class_weights=self.__class_weights)
            self.__decision_trees.append(tree)

    def get_max_depth(self) -> int:
        """
        Returns the maximum allowed depth of the trees in the forest.
        """
        return self.__max_depth

    def get_min_samples_split(self) -> int:
        """
        Returns the minimum number of samples required for a node to be considered for splitting.
        If a node has fewer samples than this, it will be a leaf node.
        """
        return self.__min_samples_split

    def __bootstrap_samples(self, n_patients_local: int) -> np.ndarray:
        sample_size = max(round(n_patients_local * self.__max_samples), 1)
        sample_idcs = np.random.choice(n_patients_local, sample_size, replace=self.__bootstrap)
        return sample_idcs

    def get_leaf_node_samples(self) -> List[List[List[int]]]:
        """
        Receives the local potential predicted classes per leaf. The order of the leaf nodes
        is given by performing DFS on the tree nodes until leaf nodes are reached.

        Returns:
            For each tree in the forest, for each leaf node, a list of the frequencies of the
            global_classes in the leaf node. The index of the returned list corresponds to the index
            of the global classes.
        """
        if self.__y is None:
            raise ValueError('No y data to calculate the leaf node samples.')
        tree_leaves = []
        for estimator in self.__decision_trees:
            tree_leaves.append(estimator.get_leaf_node_samples(y=self.__y))
        return tree_leaves

    def set_final_leaf_nodes(self, global_leaf_samples: List[np.ndarray]) -> None:
        """
        Given the global classes predicted by the leaf nodes, sets the leaf nodes

        Args:
            global_leaf_samples: List[np.ndarray] (n_estimators x num_leaf_nodes):
                The class_idx for each leaf node in each tree in each split
        """
        if len(global_leaf_samples) != len(self.__decision_trees):
            raise ValueError('Number of trees does not match the number of global leaf trees.')
        for tree_idx, tree_data in enumerate(global_leaf_samples):
            tree = self.__decision_trees[tree_idx]
            counter = 0
            for leaf_node, leaf_class_idx in zip(tree._traverse_tree_dfs_leaves(), tree_data):
                leaf_node.value = self.__global_classes[leaf_class_idx]
                counter += 1
            assert counter == len(tree_data), 'Got a different number of leaf nodes than expected: ' + \
                f'tree iteration {counter} vs got these globally {len(tree_data)}'


    def calc_oob(self) -> List[Tuple[int, int]]:
        """
        Calculate the out-of-bag error for the random forest.

        Returns:
            List[Tuple[int, int]]: List of tuples of the incorrect samples and the total oob samples
                for each tree in the forest.
        """
        if not self.__oob:
            raise ValueError('OOB error is not used but the oob calculation is called')
        if self.__X_hist is None or self.__y is None:
            raise ValueError('No data to calculate the oob error.')
        oob_errors = []
        for tree in self.__decision_trees:
            if not tree.finished:
                raise ValueError('Tree is not finished, cannot calculate oob error.')
            wrong_predictions, total_oob_samples = tree.calc_oob_error(
                X_hist=self.__X_hist,
                y=self.__y)
            oob_errors.append((wrong_predictions, total_oob_samples))
        return oob_errors


    def predict(self,
                X: np.ndarray) -> np.ndarray:
        """
        Given the input data X, predict the target values. The input data should have the same
        columns as the data used for training the model (in the same order).
        Automatically performs z-score normalization for the quantile binned features.
        Then uses that data to predict using the decision trees.

        Args:
            X: input data, 2d array of shape (n_samples, n_features).

        Returns:
        np.ndarray: the predicted target values. Length is n_samples.
        """
        # normalize the data
        X = self.normalize(X)

        # predict each tree
        predicted_values = np.array([tree.predict(X) for tree in self.__decision_trees])
            # n_estimators x n_samples
        # we switch the axes to have the samples as the first axis
        predicted_values = np.swapaxes(predicted_values, axis1=0, axis2=1)
            # n_samples x n_estimators
        weights = np.array([tree.get_weight() for tree in self.__decision_trees])

        # apply the weights and make the final prediction
        sum_predicted_classes = np.zeros((len(X), len(self.__global_classes)))
            # n_samples x n_classes, for each sample the sum of the predicted classes
        for class_idx, cl in enumerate(self.__global_classes):
            sum_predicted_classes[:, class_idx] = np.sum((predicted_values == cl) * weights, axis=1)
                # for each class, sum the weights of the trees * if the corresponding class is
                # predicted by the treex
                # this works because the True is seen as 1 and the False as 0
        predicted_values:np.ndarray = np.argmax(sum_predicted_classes, axis=1)
            # we go from n_samples x n_classes to n_samples by taking the index of the highest value
        # now we translate from class_idx to the actual class
        predicted_values = np.vectorize(lambda x: self.__global_classes[x])(predicted_values)
        return predicted_values


    def normalize(self, X: np.ndarray):
        """
        Normalizes all quantile binning features in X using the global mean and standard deviation.

        Args:
            X: input data, 2d array of shape (n_samples, n_features).

        Returns:
            np.ndarray: the normalized input data X.
            Formula is the z-score: (X - mean) / stddev
        """
        # get quantile specific data
        X_quantile = X[:, self.__quantile]
        a = (X_quantile - self.__global_mean)
        b = self.__global_stddev
        normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        normalized[normalized == np.inf] = 0
        normalized[normalized == -np.inf] = 0
        normalized[normalized == np.nan] = 0

        # replace the quantile data with the normalized data
        X[:, self.__quantile] = normalized
        return X

    def iterate_trees(self) -> Iterator["DecisionTree"]:
        """
        Iterate over the decision trees in the random forest.
        """
        for tree in self.__decision_trees:
            yield tree

    def get_split_scores(self) -> \
            Tuple[List[Optional[List[List[List[float]]]]],
                  List[Optional[List[List[int]]]],
                  List[Optional[List[Optional[Any]]]]]:
        """
        Calculate the score of a split for each feature and bin. Works on the cur_depth_nodes.
        Any tree that is finished will return None for scores and counts.
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
            counts: 3d array of shape (n_estimators, num_nodes_cur_level, len(self.feat_idcs)),
                containing the counts of samples for each feature and node per tree
            only_class: 2d array of shape (n_estimators, num_nodes_cur_level), containing the
                class if the node has only one class, None otherwise
                The whole array of nodes might be None if the tree is finished.
        """
        scores = []
        counts = []
        only_classes = []
            # per tree
        if self.__X_hist is None or self.__y is None:
            raise ValueError('No data to calculate the split scores. Is the model finished?')
        for tree in self.__decision_trees:
            if tree.finished:
                scores.append(None)
                counts.append(None)
                only_classes.append(None)
            else:
                tree_scores, tree_counts, tree_only_class = tree.get_split_scores(
                    X_hist=self.__X_hist,
                    y=self.__y,
                    n_bins=self.__num_bins)
                scores.append(tree_scores)
                counts.append(tree_counts)
                only_classes.append(tree_only_class)
        return scores, counts, only_classes

    def set_currently_unset_nodes(self,
                                  global_best_split: List[Optional[List[Tuple[int, int, float]]]],
                                  global_leaf_info: List[Optional[List[int]]]) \
                                -> None:
        """
        Based on the global_split_scores, set the nodes that have not been set yet (current_depth_nodes)
        Important: This assumes that the indexing of the global_split_scores is the same as the
        indexing of the trees and nodes in the current_depth_nodes list per tree.
        Then sets the next current_depth_nodes based on the leaf info. If that list is empty,
        the tree is set to be finished.

        Args:
            global_split_scores: 2d list of dimensions (n_estimators, num_nodes_cur_level)
                Contains for each estimator and node a tuple of (feature_idx, bin_idx, score)
                describing the globally best split. If any entry in the n_estimators dimension is None,
                the corresponding tree is considered finished and there is no update for this tree.
                Careful, the feature_idx is the index in the node's feature_idcs list!
            global_leaf_info: 2d list of dimensions (n_estimators, num_leaf_nodes),
                Contains for each estimator the indexes of the leaf nodes.
                Can be None if the tree is finished.
        """
        if self.__X_hist is None or self.__y is None or self.__split_points is None:
            raise ValueError('No data to set the nodes. Is the model finished?')
        for tree_idx, tree in enumerate(self.__decision_trees):
            if global_best_split[tree_idx] is None or global_leaf_info[tree_idx] is None:
                if not tree.finished:
                    raise ValueError('Global model assumes finished tree, local model does not.')
                continue

            relevant_leaf_idxs = set(global_leaf_info[tree_idx]) #type: ignore
                # type ignore as this is handled by the if above
            next_depth_nodes = []
            for node_idx, node in enumerate(tree.iterate_cur_depth_nodes()):
                if node.is_leaf_node():
                    raise ValueError('Leaf nodes should not be in the current depth nodes when setting this nodes.')
                # set node to leaf if necessary
                if node_idx in relevant_leaf_idxs:
                    # we need to set the node to a leaf node
                    # when the model is finished the actual predicted value by the leaf node
                    # is set using the set_final_leaf_nodes method
                    node.set_as_leaf_node()
                    continue
                # update the node and create the children correctly
                if node_idx >= len(global_best_split[tree_idx]): #type: ignore
                    raise ValueError('Global model assumes more nodes than local model.')
                feature_idx, bin_idx, score = global_best_split[tree_idx][node_idx] #type: ignore
                # CAREFUL, we need to use the feature index in the node's feature_idcs list!
                # we can simply use the trees feature_idcs as all nodes of a tree have the same
                # feature_idcs list
                left_child, right_child = node.set_node(
                    feature_idx=self.__feat_idcs[tree_idx][feature_idx],
                    bin_idx=bin_idx,
                    threshold=self.__split_points[self.__feat_idcs[tree_idx][feature_idx], bin_idx],
                    score=score,
                    X_hist=self.__X_hist
                )
                # add the children to the next depth nodes list
                next_depth_nodes.append(left_child)
                next_depth_nodes.append(right_child)
            # update the current depth nodes to the next depth nodes
            tree.update_cur_depth_nodes(next_depth_nodes)

    def set_weights(self, weights: List[float]) -> None:
        """
        Set the weights of the trees in the forest.
        """
        if len(weights) != self.n_estimators:
            raise ValueError('Number of weights does not match the number of trees.')
        for tree_idx, tree in enumerate(self.__decision_trees):
            tree.__weight = weights[tree_idx]


    def check_finished(self):
        """
        Check if all trees are finished.
        Performs cleanup and sets the finished flag if all trees are finished.
        Returns:
            True if all trees are finished, False otherwise.
        """
        for tree in self.__decision_trees:
            if not tree.finished:
                return False
        # cleanup
        self.finished = True
        return True

    def get_hyperparameters_used(self):
        """
        Returns the hyperparameters used in training this model. Check the function to see which
        hyperparameters are returned.
        """
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.__max_samples,
            'feat_idcs': self.__feat_idcs,
            'max_depth': self.__max_depth,
            'min_samples_split': self.__min_samples_split,
            'bootstrap': self.__bootstrap,
            'random_state': self.__random_state,
            'quantile': self.__quantile,
            'global_mean': self.__global_mean,
            'global_stddev': self.__global_stddev,
            'split_points': self.__split_points,
            'prediction_mode': self.prediction_mode,
            'global_classes': self.__global_classes,
            'oob': self.__oob,
            'class_weights': self.__class_weights,
            'num_bins': self.__num_bins
        }

    def cleanup_model(self):
        """
        Removes any traces of the training process. Should be called before sharing this model!
        """
        self.__split_points = None
        self.__class_weights = None
        self.__X_hist = None
        self.__y = None
        for tree in self.__decision_trees:
            tree._cleanup_tree()

    def save_model(self, path: str):
        """
        Cleanes the model and saves it to the given basepath as model.pkl.
        Uses joblib to save the model.
        """
        # cleanup
        self.cleanup_model()

        # save the model as well as this class
        joblib.dump(self, path)



class DecisionTree:
    """
    A decision tree model for classification and regression to be used for federated histogram
    based random forest learning and prediction.
    """
    def __init__(self,
                 samples_idcs: np.ndarray, # 1d array of indices of the samples to be used
                 max_depth: int,
                 min_samples_split: int,
                 feat_idcs: np.ndarray, # List of feature indices for this tree:
                                        # 1d array of length n_features
                 mode: str, # classification or regression
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                        # same order for all clients
                 class_weights: Optional[Dict[Any, float]],
                    # weights of the classes (dict[class]=weight)
                 ) -> None:
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__feat_idcs = feat_idcs
        self.finished = False
        self.__weight:float = 1.0
        self.__global_classes = global_classes
        self.__class_weights = class_weights

        self.mode = mode
        if self.mode == 'regression' and self.__class_weights is not None:
            raise ValueError('Weights are not supported for regression.')

        # init the root node
        self.root = Node(depth=0,
                         sample_idcs=samples_idcs,
                         mode=mode,
                         global_classes=global_classes,
                         class_weights=class_weights,
                         feature_idcs=feat_idcs,
                         parent=None)

        self.__cur_depth_nodes = [self.root]

    def get_weight(self) -> float:
        """
        Returns the weight of this tree.
        """
        return self.__weight

    def predict(self,
                X):
        """
        Predict the target values for the input data X (2d array).
        Assumes the same features with the same indices as the training data.
        """
        return np.array([self._traverse_tree_predict(x, self.root) for x in X])

    def predict_hist(self,
                     X_hist: np.ndarray):
        """
        Predict the target values for the input data X_hist (2d array).
        Assumes the same features with the same indices as the training data.
        """
        return np.array([self._traverse_tree_predict_hist(x, self.root) for x in X_hist])

    def iterate_cur_depth_nodes(self) -> Iterator["Node"]:
        """
        Iterate over the nodes at the current depth.
        """
        for node in self.__cur_depth_nodes:
            yield node

    def get_split_scores(self, X_hist: np.ndarray, y: np.ndarray, n_bins: int) -> \
            Tuple[List[List[List[float]]], List[List[int]], List[Optional[Any]]]:
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
            scores: 3d list of dimensions (num_nodes_cur_level, len(self.feat_idcs), n_bins),
            containing the score of a split per feature and bin.
            counts: 2d list of dimensions (num_nodes_cur_level, len(self.feat_idcs)), containing
                the counts of samples for each feature and node
            only_classes: 1d list of length num_nodes_cur_level, containing the class if the node
                has only one class, None otherwise
        """
        scores = []
        counts = []
        only_classes = []
            # per node
        if len(self.__cur_depth_nodes) == 0 or self.finished:
            # this method should only be called if there are nodes to calculate the split for
            raise ValueError('No nodes to calculate the split for, the tree is finished.')
        for node_idx, node in enumerate(self.__cur_depth_nodes):
            # IMPORTANT: we actively don't check if the node is finished, as the nodes in
            # the current depth are required to not be set yet!
            # this is why we throw an error if the node is already set
            # we later should only add the children of non leaf nodes to
            # current_depth_nodes, this is why if we here then find a set node, we should
            # throw an error
            node_scores, node_counts, node_only_class = node.get_split_scores(
                X_hist,
                y,
                n_bins)
            scores.append(node_scores)
            counts.append(node_counts)
            only_classes.append(node_only_class)
        return scores, counts, only_classes

    def get_cur_depth_nodes(self):
        """
        Returns all nodes at the current depth.
        """
        return self.__cur_depth_nodes

    def get_cur_depth_node(self, idx):
        """
        Returns the node at the current depth with the given index.
        """
        return self.__cur_depth_nodes[idx]

    def update_cur_depth_nodes(self, new_nodes):
        """
        Updates the current depth nodes to the new nodes.
        If this results in an empty list, the tree is considered finished.
        """
        self.__cur_depth_nodes = new_nodes
        if len(self.__cur_depth_nodes) == 0:
            self.__feat_idcs = None
            self.__class_weights = None
            self.finished = True

    def get_leaf_node_samples(self, y: np.ndarray) -> List[int]:
        """
        Receives the local potential predicted classes per leaf. The order of the leaf nodes
        is given by performing DFS on the tree nodes until leaf nodes are reached.

        Returns:
            List[int]: For each leaf node, contains the frequencies of the global_classes
            in the leaf node. The index of the returned list corresponds to the index of the
            global classes.
            The leaf interation is done via _traverse_tree_dfs
        """
        leaves = []
        for node in self._traverse_tree_dfs_leaves():
            y_node = y[node.get_sample_idcs()]
            class_frequencies = np.sum(y_node[:, np.newaxis] == self.__global_classes, axis=0)
            # go from y_node 1d array of len num_samples to 2d array of shape (num_samples, 1)
            # then compare each element with the global classes, creating a boolean array
            # of shape (num_samples, num_classes)
            # then sum over the samples to get the frequency of each class
            leaves.append(class_frequencies)
        return leaves


    def calc_oob_error(self,
                       X_hist: np.ndarray,
                       y: np.ndarray) -> Tuple[int, int]:
        """
        Calculates the oob error of this tree. Predicts the samples, then returns the amount of
        incorrect samples and the total samples.

        Args:
            X: input data, 2d array of shape (n_samples, n_features).
            y: target data, 1d array of shape (n_samples), indicating the target value

        Returns:
            Tuple[int, int]: the amount of incorrect samples and the total amount of oob samples
        """
        # 1. get the oob samples
        # 2. predict the oob samples
        # 3. return the amount of incorrect samples and the total oob samples
        relevant_samples = self.root.get_sample_idcs()
        if relevant_samples is None:
            raise ValueError('No sample indices for this trees root.')
        oob_samples = np.setdiff1d(np.arange(len(X_hist)), relevant_samples)
        predicted_values = self.predict_hist(X_hist[oob_samples])
        incorrect_samples = np.sum(predicted_values != y[oob_samples])
        return incorrect_samples, len(oob_samples)


    def _check_leaf_node_consistency(self, node: "Node"):
        """
        Recursive function that finds all nodes without children and checks that they are leaf nodes.

        Raises:
            ValueError: If a node without children is not a leaf node or a if a leaf node has children.
        """
        if node.global_leaf:
            if node.left or node.right:
                raise ValueError('Leaf node has children.')
            return # leaf node with no children, all good
        if node.left is None or node.right is None:
            raise ValueError('Non-leaf node without children.')
        self._check_leaf_node_consistency(node.left)
        self._check_leaf_node_consistency(node.right)

    def _traverse_tree_predict_hist(self, x_hist: np.ndarray, node: "Node"):
        """
        Traverse the tree to find the leaf node for the input data x_hist, effectively making a prediction.

        Args:
            x_hist: contains one single raw, the data is not the actual data but the bin_idxes
                each value belongs to.
        """
        if len(x_hist) != 1:
            raise ValueError("Function must be called with only a singular raw")
        if node.is_leaf_node():
            return node.value
        if node.feature_idx is None or node.threshold is None or node.left is None or node.right is None or not node.bin_idx:
            raise ValueError('Node is not set correctly')
        if x_hist[node.feature_idx] <= node.bin_idx:
            return self._traverse_tree_predict_hist(x_hist, node.left)
        return self._traverse_tree_predict_hist(x_hist, node.right)

    def _traverse_tree_predict(self, x: np.ndarray, node: "Node"):
        """
        Traverse the tree to find the leaf node for the input data x, effectively making a prediction.

        Args:
            x: contains one single raw
        """
        if len(x.shape) != 1:
            raise ValueError("Function must be called with only a singular raw")
        if node.is_leaf_node():
            return node.value
        if node.feature_idx is None or node.threshold is None or node.left is None or node.right is None:
            raise ValueError('Node is not set correctly')
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree_predict(x, node.left)
        return self._traverse_tree_predict(x, node.right)

    def _traverse_tree_dfs_leaves(self):
        """
        Traverse the tree in a depth-first search manner and return only the leaf nodes.
        """
        for node in self._traverse_tree_dfs_helper(self.root):
            if node.is_leaf_node():
                yield node

    def _traverse_tree_dfs_helper(self, node: "Node") -> Generator["Node", None, None]:
        """
        Helper function for the depth_first_search.
        Returns the given node, then the left child, then the right child.
        Recursive
        """
        yield node
        if not node.is_leaf_node():
            if node.left is None or node.right is None:
                raise ValueError("A leaf node has children")
            yield from self._traverse_tree_dfs_helper(node.left)
            yield from self._traverse_tree_dfs_helper(node.right)

    def _cleanup_tree(self):
        """
        When a tree is finished, this function cleans up the tree of any private variables that are
        not needed anymore.
        """
        if not self.finished:
            raise ValueError('Tree is not finished, but trying to clean it')
        self.__cur_depth_nodes = []
        self.__class_weights = None
        for node in self._traverse_tree_dfs_helper(self.root):
            node._cleanup_node()


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
                 feature_idcs: np.ndarray, # List of feature indices of this nodes tree::
                                           # 1d array of length n_features
                 class_weights: Optional[Dict[Any, float]],
                    # weights of the classes: (dict[class]=weight)
                 parent: Optional["Node"], # pointer to the parent node
                 ) -> None:
        """
        The node of a decision tree.
        Some parameters are not set at initialization.

        Args:
            depth: the depth of the node in the tree
            sample_idcs: 1d array of indices of the samples this node uses
                These are the samples that passed through the parent node in the decision tree
            mode: classification or regression
            global_classes: global classes. Should be the same in the same order for all clients
            feature_idcs: List of feature indices of this nodes tree
                1d array of length n_features
            class_weights: weights of the classes: (dict[class]=weight)
            parent: pointer to the parent node
        """
        # OPTIMIZATION: we save the feature_idcs in each node, maybe we could save them per tree
        # and only pass which features we cannot use per node
        self.depth = depth
        self.parent = parent
        self.__sample_idcs = sample_idcs
        self.__global_classes = global_classes
        self.__feature_idcs = feature_idcs

        self.feature_idx = None
        self.bin_idx = None
        self.threshold = None
        self.score = None
        self.left = None
        self.right = None
        self.global_leaf = None

        if len(set(global_classes)) != len(global_classes):
            raise ValueError('Classes must be unique.')
        self._num_classes = len(global_classes)
        if mode not in ['classification', 'regression']:
            raise ValueError('Mode must be either classification or regression.')
        self.mode = mode
        for cl in global_classes:
            if class_weights and cl not in class_weights:
                raise ValueError('Class weights must be given for all classes or not at all')
        self.class_weights = class_weights

    def set_node(self,
                 feature_idx: int,
                 bin_idx: int,
                 threshold: Union[float, int],
                 score: float,
                 X_hist: np.ndarray) -> Tuple["Node", "Node"]:
        """
        Set the feature, threshold and score of the node.
        Then creates the left and right child nodes and returns them.

        Args:
            feature_idx: the feature_idx to use in this node. This is the index of all data, not
                of this nodes subset of features!
            bin_idx: the bin index to use in this node
            threshold: the threshold value at which to split the data
                threshold is used with the actual data, bin_idx with the histogram data
                threshold is the value where to split, splitting is
                x <= threshold -> left child, x > threshold -> right child
            score: the score (e.g. gini impurity) of the split
            X_hist: input data, 2d array of shape (n_samples, n_features).
                The values are NOT the actual values but the bin indices the samples belong to
                for each feature. This is used to calculate the sampleset for the children together
                with the bin_idx. The threshold is not used here.

        Returns:
            A tuple left_child_node, right_child_node
            Contains the left and right child nodes of the current node.
        """
        if self.feature_idx or self.bin_idx or self.threshold or self.score:
            raise ValueError('Trying to set a node that is set already.')
        if self.global_leaf:
            raise ValueError('Trying to set a leaf node.')
        if self.__sample_idcs is None:
            raise ValueError('Node has no sample indices. Cannot set a split.')

        # set the node
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx
        self.threshold = threshold
        self.score = score

        relevant_data = X_hist[self.__sample_idcs, feature_idx]
        # create the left and right child nodes
        left_child_idcs, right_child_idcs = self.perform_hist_based_split(relevant_data, bin_idx)
        # Careful, these indices are the indices in the sample_idcs list of the parent node
        # we need to translate them back to the sample_idcs
        left_child_idcs = self.__sample_idcs[left_child_idcs]
        right_child_idcs = self.__sample_idcs[right_child_idcs]
        left_child = Node(depth=self.depth + 1,
                            sample_idcs=left_child_idcs,
                            mode=self.mode,
                            global_classes=self.__global_classes,
                            class_weights=self.class_weights,
                            feature_idcs=self.__feature_idcs,
                            parent=self)
        right_child = Node(depth=self.depth + 1,
                            sample_idcs=right_child_idcs,
                            mode=self.mode,
                            global_classes=self.__global_classes,
                            class_weights=self.class_weights,
                            feature_idcs=self.__feature_idcs,
                            parent=self)
        self.left = left_child
        self.right = right_child

        return left_child, right_child

    def set_as_leaf_node(self) -> None:
        """
        Set the node as a leaf node.
        The value can be set later in the set_finally_as_leaf_node function, as setting the value
        requires the local class frequencies
        """
        self.global_leaf = True
        if self.left or self.right:
            raise ValueError('Leaf node has children.')
        if self.feature_idx or self.bin_idx or self.threshold or self.score:
            raise ValueError('Leaf node should not have a split.')

    def get_sample_idcs(self) -> Optional[np.ndarray]:
        """
        Simple getter for the sample_idcs asssociated with this node.
        Basically the sample_idcs that split according to the parents.
        """
        return self.__sample_idcs

    def set_finally_as_leaf_node(self, value: Optional[Any]) -> None:
        """
        Set the node as a leaf node.
        The value can be set later, as setting the value requires the local class frequencies in
        that leaf node. This can be done in each iteration or in the end.
        """
        if not self.global_leaf:
            raise ValueError('Node is not a leaf node, should have been set already using set_as_leaf_node.')
        if self.left or self.right:
            raise ValueError('Leaf node has children.')
        if self.feature_idx or self.bin_idx or self.threshold or self.score:
            raise ValueError('Leaf node should not have a split.')
        self.value = value

    def _cleanup_node(self):
        """
        When a node is finished (either set as a leaf or as a split node), we remove the
        variables that are not needed anymore and potentially private.
        """
        self.__sample_idcs = None
        self.feature_idcs = None
        self.class_weights = None

    def is_leaf_node(self):
        """
        Whether the node is GLOBALLY considered a leaf node.
        """
        return self.global_leaf

    def get_split_scores(self, X_hist: np.ndarray, y: np.ndarray, n_bins: int) -> \
            Tuple[List[List[float]], List[int], Optional[Any]]:
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
            scores: 2d array of shape (len(self.feature_idcs), n_bins-1), containing the score of a split per
                feature and bin.
            counts: list of length len(self.feature_idcs), containing the counts of samples for each feature
            only_class: the class if the node sampleset has only one class, None otherwise
        """
        # TODO: possible optimization:
        # the splitscore calculation is wrong right now.
        # Consider the following for the gini impurity formula:
        # gini = #samples_left / #samples_total * (1 - sum_class_i(#samples_left_class_i/#samples_left)**2) + right...
        # To correctly calculate this in a federated setting, we could exchange #samples_left
        # and #samples_right per client with the coordinator to aggregate it, then broadcast it
        # back to the clients and use it to calculate the gini impurity. If weighting is used,
        # we could instead exchange the corresponding weightsums.
        if self.feature_idx is not None or self.threshold is not None:
            raise ValueError('Node already has a split.')
        # ensure input data formatting
        if len(X_hist.shape) != 2:
            raise ValueError('X_hist must be a 2d array.')
        if X_hist.shape[0] != len(y):
            raise ValueError('X_hist and y must have the same number of samples.')
        if len(y.shape) != 1:
            raise ValueError('y must be a 1d array.')
        node_data = X_hist[self.__sample_idcs, :]
        node_y = y[self.__sample_idcs]
        scores = []
            # num_features x num_bins
        counts = []
            # list of length num_features, containing the counts of samples for each feature
        for feature_idx in self.__feature_idcs:
            feature_data = node_data[:, feature_idx]
            scores_per_bin = []
                # array of length num_bins containing the scores per_bin
            for bin_idx in range(n_bins-1):
                # since we check for <= bin_idx -> left, for the last bin_idx, left would always
                # contain all data, so we don't need to check it
                left_idxs, right_idxs = self.perform_hist_based_split(feature_data, bin_idx)
                left_y = node_y[left_idxs]
                right_y = node_y[right_idxs]
                if self.mode == 'classification':
                    # Each node is a binary split, we just check that the impurity is minimized
                    score = self._gini_split_score(left_y, right_y)
                else: # regression
                    score = self._mse_split_score(left_y, right_y)
                scores_per_bin.append(score)
            scores.append(scores_per_bin)
            counts.append(len(feature_data))

        # potential stopping criteria:
        # node only contains one class
        # this is independant of the split_feature and bin, these influence the child nodes
        # only_class status and are considered when this function is called on the child nodes
        only_class = None
        if len(np.unique(y)) == 1:
            # if other clients have different classes, this is not a leaf!
            # just send this information along and decide on the coordinator
            only_class = y[0]
        return scores, counts, only_class

    def perform_hist_based_split(self, X_hist_col: np.ndarray, bin_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Based on specific samples of a specific feature in the X_hist data as well as the bin_idx
        where to split, return the indices of the samples that would end up in the left and right
        child node.
        Uses the following formula:
            left_idxs = np.where(X_hist_col <= bin_idx)[0]
            right_idxs = np.where(X_hist_col > bin_idx)[0]
        Therefore left_idxs includes the samples right at the threshold

        Args:
            X_hist_col: the column of the X_hist data that we are considering. A 1d array.
            bin_idx: the bin index where to split

        Returns:
            Tuple[left_idxs, right_idxs]: the indices of the samples that would end up in the left
                and right child node.
        """
        left_idxs = np.where(X_hist_col <= bin_idx)[0]
        right_idxs = np.where(X_hist_col > bin_idx)[0]
        return left_idxs, right_idxs

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

        In case of a perfect split, the gini impurity is 0, so the gini score is 0
        (1-1 for one class other classes are all 0). In the worst case, the gini impurity goes
        towards 1.

        Args:
            left_y: target values of the samples that would end up in the left node
            right_y: target values of the samples that would end up in the right node

        Returns:
            gini: the gini impurity of the split
        """
        if len(left_y.shape) != 1 or len(right_y.shape) != 1:
            raise ValueError('y must be a 1d array.')

        left_weights = np.ndarray((0))
        total_left = 0
        if len(left_y) > 0:
            # vectorize fails on empty arrays, so we need this if
            left_weights = np.ones((len(left_y))) if self.class_weights is None else np.vectorize(self.class_weights.get)(left_y)
            # either just one for any sample or the weight of the sample by their class
            # np.vectorize(self.class_weights.get)(left_y) runs weights.get on each element
            # of left_y constructing an np.array. we therefore get an np.array of length samples
            # with the weight for each sample as values.
            total_left = np.sum(left_weights)
        right_weights = np.ndarray((0))
        total_right = 0
        if len(right_y) > 0:
            right_weights = np.ones((len(right_y))) if self.class_weights is None else np.vectorize(self.class_weights.get)(right_y)
            total_right = np.sum(right_weights)

        total_y = total_left + total_right
        if total_left == 0 or total_right == 0:
            # this means that our 'split' does not split at all
            # numpy just creates a nan tho
            # we want to disencourage such splits, they don't have any information gain
            # we simply set the gini score to the worst possible score (1)
            return np.float64(1.0)

        gini_left = 1.0 - np.sum((np.bincount(left_y.astype('int'), minlength=self._num_classes, weights=left_weights) / total_left) ** 2)
            # with np.bincount we get an array of length n_classes with the index being the specific
            # class. The values are the added weights of the samples of each class.
            # if self.class_weights is None, we just add one for each sample, so we have the
            # counts of samples of each class
            # we then divide by the total weight/total number of samples
            # to get the probability of each class
        gini_right = 1.0 - np.sum((np.bincount(right_y.astype('int'), minlength=self._num_classes, weights=right_weights) / total_right) ** 2)

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

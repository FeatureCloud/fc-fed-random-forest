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

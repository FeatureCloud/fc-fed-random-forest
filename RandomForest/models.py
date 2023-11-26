import numpy as np
from scipy import stats

class RandomForest:

    def __init__(self, n_estimators, max_samples, feat_idcs, max_depth, min_samples_split, \
                  min_samples_leaf, n_patients, bootstrap, random_state, quantile, global_mean, \
                  global_stddev, split_points, prediction_mode, oob):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feat_idcs = feat_idcs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_patients = n_patients
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.decision_trees = []
        np.random.seed(random_state)
        self.quantile = quantile
        self.global_mean = global_mean
        self.global_stddev = global_stddev
        self.split_points = split_points
        self.prediction_mode = prediction_mode
        self.oob = oob
        self.finished = False
    
    def init_trees(self, y):
        self.estimators = []
        for i in range(self.n_estimators):
            samples = self._bootstrap_samples()
            root_node = Node(depth=0, samples=samples)
            n_classes = len(np.unique(y[samples]))
            tree = DecisionTree(
                samples=samples,
                feat_idcs=self.feat_idcs[i],
                n_classes=n_classes,
                root=root_node,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                cur_depth_nodes = [root_node],
                next_depth_nodes = []
            )
            self.decision_trees.append(tree)

    def _bootstrap_samples(self):
        if self.max_samples is None:
            sample_size = self.n_patients
        elif isinstance(self.max_samples, float):
            sample_size = max(round(self.n_patients * self.max_samples), 1)
        else:
            sample_size = self.max_samples
        samples = np.random.choice(self.n_patients, sample_size, replace=self.bootstrap)
        return samples

    def predict(self, X):
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


class DecisionTree:

    def __init__(self, samples, max_depth, min_samples_split, min_samples_leaf, feat_idcs, \
                 n_classes, root, cur_depth_nodes, next_depth_nodes):
        self.samples = samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feat_idcs = feat_idcs
        self.n_classes = n_classes
        self.root = root
        self.cur_depth_nodes = cur_depth_nodes
        self.next_depth_nodes = next_depth_nodes
        self.finished = False
        self.leaves = []
        self.weight = 1

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class Node:

    def __init__(self, depth, samples, feature=None, threshold=None, score=None, parent=None,
                  left=None, right=None, global_leaf=False, local_leaf=False, value=None):
        self.depth = depth
        self.samples = samples
        self.feature = feature
        self.threshold = threshold
        self.score = score
        self.parent = parent
        self.left = left
        self.right = right
        self.global_leaf = global_leaf
        self.local_leaf = local_leaf
        self.value = value
    
    def is_leaf_node(self):
        return self.global_leaf
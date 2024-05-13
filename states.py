import numpy as np
import pandas as pd
import os
import joblib
from scipy.stats import norm
from FeatureCloud.app.engine.app import AppState, app_state, Role
from helper.io import read_config, read_files
from RandomForest.models import RandomForest, Node
from RandomForest.splitting import split_score


@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Read config file and input data.
    """

    def register(self):
        self.register_transition('local_binning', Role.BOTH)

    def run(self) -> str or None:
        self.update(message=f'Read files', progress=0.05)
        self.log('Read config-file...')
        train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, \
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, \
            max_features, bootstrap, max_samples, random_state, prediction_mode, quantile, \
            n_bins, oob = read_config()

        self.log('Read data...')
        X, y, X_test, y_test = [], [], [], []  
        if len(quantile) > 0:
            if ',' in quantile:
                quantile = np.fromstring(quantile, dtype=int, sep=',')
            else:
                quantile = np.array([int(quantile)])
        else:
            quantile = np.empty(0, dtype=int)
        
        if split_mode == 'directory':
            for split_name in os.listdir('/mnt/input/' + split_dir):
                X_, y_, X_test_, y_test_ = read_files(os.path.join(split_dir, split_name, \
                     train), os.path.join(split_dir, split_name, test_input), sep, label_col)
                X.append(X_)
                y.append(y_)
                X_test.append(X_test_)
                y_test.append(y_test_)
        else:
            X_, y_, X_test_, y_test_ = read_files(train, test_input, sep, label_col)
            X.append(X_)
            y.append(y_)
            X_test.append(X_test_)
            y_test.append(y_test_)
        
        np.random.seed(random_state)

        # Store parameters from config file
        self.store('pred', pred)
        self.store('test_output', test_output)
        self.store('sep', sep)
        self.store('label_col', label_col)
        self.store('split_mode', split_mode)
        self.store('split_dir', split_dir)

        # Parameters RandomForest
        self.store('n_estimators', n_estimators)
        self.store('criterion', criterion)
        self.store('max_depth', max_depth)
        self.store('min_samples_split', min_samples_split)
        self.store('min_samples_leaf', min_samples_leaf)
        self.store('max_features', max_features)
        self.store('bootstrap', bootstrap)
        self.store('max_samples', max_samples)
        self.store('random_state', random_state)

        self.store('prediction_mode', prediction_mode)
        self.store('quantile', quantile)
        self.store('n_bins', n_bins)

        self.store('oob', oob)
        
        if self.load('prediction_mode') == 'regression':
            self.store('oob', False)
        
        # Store data
        self.store('X', X)
        self.store('y', y)
        self.store('X_test', X_test)
        self.store('y_test', y_test)
        self.store('classes', np.unique(y[0]))
        
        self.store('n_features', X[0].shape[1])
        self.store('depth', 0)

        return 'local_binning'


@app_state('local_binning', Role.BOTH)
class LocalBinningState(AppState):
    """
    Calculate values for z-score normalization and determine local minima and maxima 
    for bucket binning and send these values to the coordinator.
    """

    def register(self):
        self.register_transition('aggregate_binning', Role.BOTH)
        
    def run(self) -> str or None:
        
        # Normalize data for quantile binning
        X = self.load('X')
        quantile_idcs = self.load('quantile')
        local_matrix_list = []
        for split in range(len(X)):
            X_quantile = X[split][:, quantile_idcs]
            local_matrix = np.zeros((X_quantile.shape[1], 3))
            local_matrix[:, 0] = X_quantile.shape[0]  
            local_matrix[:, 1] = np.sum(np.square(X_quantile), axis=0)
            local_matrix[:, 2] = np.sum(X_quantile, axis=0)
            local_matrix_list.append(local_matrix)

        # Get minimum and maximum for bucket binning
        send_data_bucket = []
        bucket_idcs = np.setdiff1d(np.arange(len(X[0][0])), quantile_idcs)

        for split in range(len(X)):
            min_array = np.min(X[split][:, bucket_idcs], axis=0)
            max_array = np.max(X[split][:, bucket_idcs], axis=0)
            send_data_bucket.append(np.array([min_array, max_array]))
        
        self.send_data_to_coordinator([local_matrix_list, send_data_bucket])

        return 'aggregate_binning'



@app_state('aggregate_binning', Role.BOTH)
class AggregateBinningState(AppState):

    """
    Aggregate data for federated z-score normalization and calculate global minima and maxima
    for bucket binning.
    """

    def register(self):
        self.register_transition('global_binning', Role.BOTH)
        
    def run(self) -> str or None:
        if self.is_coordinator:
            gathered_data = self.gather_data()

            broadcast_data_quantile = []
            local_matrix_list = [gathered_data[i][0] for i in range(len(gathered_data))]
            for split in range(len(self.load('X'))):
                data = [d[split] for d in local_matrix_list]
                global_matrix = np.sum(data, axis=0)
                mean_square = global_matrix[:, 1] / global_matrix[:, 0]
                mean = global_matrix[:, 2] / global_matrix[:, 0]
                stddev = np.sqrt(mean_square - np.square(mean))
                broadcast_data_quantile.append(np.array([mean, stddev]))

            split_points_bucket = []
            n_bins = self.load('n_bins')
            data_bucket = [gathered_data[i][1] for i in range(len(gathered_data))]

            for split in range(len(self.load('X'))):
                data = [d[split] for d in data_bucket]
                min_max_values = np.array(data)
                min_values = np.minimum.reduce(min_max_values[:, 0])
                max_values = np.maximum.reduce(min_max_values[:, 1])
                bins = [np.linspace(float(min_values[i]), float(max_values[i]), \
                    n_bins + 1) for i in range(len(min_values))]


                split_points_bucket.append([b[:-1] for b in bins])

            data = [broadcast_data_quantile, split_points_bucket]
            self.broadcast_data(data, send_to_self=False)

        else:
            data = self.await_data()

        global_mean = [d[0] for d in data[0]]
        global_stddev = [d[1] for d in data[0]]
        self.store('global_mean', global_mean)
        self.store('global_stddev', global_stddev)

        X = self.load('X')
        bucket_idcs = np.setdiff1d(np.arange(len(X[0][0])), self.load('quantile'))
        tmp_split_points = []
        tmp_X_hist = []

        for split in range(len(X)):
            split_points = np.array(data[1][split])
            tmp_split_points.append(split_points)
            X_T = np.transpose(X[split][:, bucket_idcs])
            # Assign data points to bins
            X_hist = np.array([np.digitize(X_T[i], split_points[i]) \
                                    for i in range(X_T.shape[0])]) - 1
            tmp_X_hist.append(X_hist)
        
        self.store('split_points_bucket', tmp_split_points)
        self.store('X_hist_bucket', tmp_X_hist)

        return 'global_binning'


@app_state('global_binning', Role.BOTH)
class BinningGlobalState(AppState):
    """
    Normalize data.
    """

    def register(self):
        self.register_transition('global_quantile_binning', Role.BOTH)

    def run(self) -> str or None:
        X = self.load('X')
        quantile_idcs = self.load('quantile')
        global_mean = self.load('global_mean')
        global_stddev = self.load('global_stddev')
        X_normalized = []
        for split in range(len(X)):
            a = (X[split][:, quantile_idcs] - global_mean[split])
            b = global_stddev[split]
            normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            normalized[normalized == np.inf] = 0
            normalized[normalized == -np.inf] = 0
            normalized[normalized == np.nan] = 0
            X_normalized.append(normalized)
        self.store('X_normalized_quantile', X_normalized)

        return 'global_quantile_binning'


@app_state('global_quantile_binning', Role.BOTH)
class GlobalQuantileBinningState(AppState):
    """
    Quantile Binning.
    """

    def register(self):
        self.register_transition('combine_binning', Role.BOTH)

    def run(self) -> str or None:
        n_bins = self.load('n_bins')
        X = self.load('X_normalized_quantile')
        X_hist_list = []
        
        percentiles = np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1)
        split_points = [norm.ppf(p) for p in percentiles]
        split_points = np.concatenate((split_points, [np.inf]))

        split_points_quantile = []

        for split in range(len(X)):
            X_T = np.transpose(X[split])
            # Assign data points to bins
            X_hist = np.array([np.digitize(X_T[i], split_points) \
                                    for i in range(X_T.shape[0])])
            X_hist_list.append(X_hist)

            split_points_quantile.append(np.tile(split_points, (len(X[split][0]), 1)))
        
        self.store('split_points_quantile', split_points_quantile)
        self.store('X_hist_quantile', X_hist_list)

        return 'combine_binning'


@app_state('combine_binning', Role.BOTH)
class CombineBinningState(AppState):
    """
    Concatenate Quantile Binning Data and Bucket Binning Data.
    """
        
    def register(self):
        self.register_transition('feat_idcs', Role.BOTH)

    def run(self) -> str or None:
        X = self.load('X')
        quantile_idcs = self.load('quantile')
        bucket_idcs = np.setdiff1d(np.arange(len(X[0][0])), quantile_idcs)
        
        X_hist_quantile = self.load('X_hist_quantile')
        X_hist_bucket = self.load('X_hist_bucket')
        X_hist_list = []

        split_points_quantile = self.load('split_points_quantile')
        split_points_bucket = self.load('split_points_bucket')
        split_points_list = []

        for split in range(len(X)):
            if len(quantile_idcs) > 0 and len(bucket_idcs) > 0:
                X_hist = np.concatenate((X_hist_quantile[split], X_hist_bucket[split]))
                # Place the values of array at specified indices
                X_hist[quantile_idcs] = X_hist_quantile[split]
                X_hist[bucket_idcs] = X_hist_bucket[split]
                X_hist_list.append(np.transpose(X_hist))

                split_points = np.tile(split_points_quantile[split], (len(X[split][0]), 1))
                # Place the values of array at specified indices
                split_points[quantile_idcs] = split_points_quantile[split]
                split_points[bucket_idcs] = split_points_bucket[split]
                split_points_list.append(split_points)
            
            elif len(bucket_idcs) > 0: 
                X_hist_list.append(np.transpose(X_hist_bucket[split]))
                split_points_list.append(split_points_bucket[split])
            
            else:
                X_hist_list.append(np.transpose(X_hist_quantile[split]))
                split_points_list.append(split_points_quantile[split])

        self.store('X_hist', X_hist_list)
        self.store('split_points', split_points_list)

        return 'feat_idcs'


@app_state('feat_idcs', Role.BOTH)
class FeatureIndicesState(AppState):
    """
    Choose feature indices.
    """

    def register(self):
        self.register_transition('init_forest', Role.BOTH)

    def run(self) -> str or None:
        self.log('Choose feature indices...')

        if self.is_coordinator:
            n_features = self.load('n_features')
            max_features = self.load('max_features')
            
            if max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif max_features < 1:
                max_features = int(n_features * max_features)
            else:
                max_features = int(max_features)
            self.store('max_features', max_features)
            
            RF_feat_idcs = []
            for _ in range(self.load('n_estimators')):
                feat_idcs = np.random.choice(n_features, size= \
                                                max_features, replace=False)
                RF_feat_idcs.append(feat_idcs)
            RF_feat_idcs = np.array(RF_feat_idcs)
            self.broadcast_data(RF_feat_idcs, send_to_self=False)
        
        else:
            RF_feat_idcs = self.await_data()
        
        self.store('RF_feat_idcs', RF_feat_idcs)
        return 'init_forest'
 

@app_state('init_forest', Role.BOTH)
class InitForestState(AppState):
    """
    Initialize the RandomForest(s).
    """

    def register(self):
        self.register_transition('find_local_splits', Role.BOTH)

    def run(self) -> str or None:
        self.log('Initialize forest...')
        mode = self.load('prediction_mode')


        if mode == 'classification' or mode == 'regression':
            X_hist = self.load('X_hist')
            rf_models = []

            for split in range(len(X_hist)):
                rf_model: RandomForest = \
                        RandomForest(n_estimators=self.load('n_estimators'),\
                                            random_state=self.load('random_state'),\
                                            max_depth=self.load('max_depth'),\
                                            min_samples_split=self.load('min_samples_split'),\
                                            min_samples_leaf=self.load('min_samples_leaf'),\
                                            bootstrap=self.load('bootstrap'), \
                                            feat_idcs=self.load('RF_feat_idcs'),\
                                            n_patients=X_hist[split].shape[0],\
                                            max_samples=self.load('max_samples'),\
                                            quantile=self.load('quantile'),\
                                            global_mean=self.load('global_mean')[split],\
                                            global_stddev=self.load('global_stddev')[split],\
                                            split_points=self.load('split_points')[split],\
                                            prediction_mode=self.load('prediction_mode'),\
                                            oob=self.load('oob'))
                rf_model.init_trees(self.load('y')[split])
                rf_models.append(rf_model)

        else:
            raise Exception('Only classification and regression are valid modes.')

        self.store('rf_models', rf_models)
        self.store('depth', 0)

        return 'find_local_splits'


@app_state('find_local_splits', Role.BOTH)
class LocalSplitState(AppState):
    """
    Each participants calculates split score for each feature-threshold combination and send the
    split scores to the coordinator.
    """

    def register(self):
        self.register_transition('aggregate_splits', Role.BOTH)

    def run(self) -> str or None:
        rf_models = self.load('rf_models')
        X_hist = self.load('X_hist')
        y = self.load('y')
        n_bins = self.load('n_bins')
        local_splits = []

        for split in range(len(X_hist)):
            tmp_split = []
            rf_model = rf_models[split]
            if not rf_model.finished:
                for decision_tree in rf_model.decision_trees:
                    tmp_dt = []
                    if not decision_tree.finished:
                        depth_nodes = decision_tree.cur_depth_nodes
                        for node in depth_nodes:
                            if not node.local_leaf:
                                local_split_score = split_score(X_hist[split][node.samples], \
                                                y[split][node.samples], decision_tree.feat_idcs, \
                                                n_bins, self.load('prediction_mode'), \
                                                self.load('classes'))
                            else:
                                local_split_score = [[0] * n_bins for _ in \
                                                     range(len(decision_tree.feat_idcs))]
                            tmp_dt.append(local_split_score)
                    if len(tmp_dt) > 0:
                        tmp_split.append(tmp_dt)
            if len(tmp_split) > 0:
                local_splits.append(tmp_split)
        self.send_data_to_coordinator(local_splits)

        return 'aggregate_splits'


@app_state('aggregate_splits', Role.BOTH)
class AggregateSplitState(AppState):
    """
    The coordinator receives the local split scores from each client, aggreagtes them and 
    chooses the feature-threshold combination with the minimal score value for splitting.
    The participants receive the best feature-threshold combination for splitting the data
    and they split the data based on the received feature and threshold.
    """

    def register(self):
       self.register_transition('local_stopping_criteria', Role.BOTH)

    def run(self) -> str or None:
        if self.is_coordinator:
            rf_models = self.load('rf_models')
            data = self.gather_data()
            global_splits = []
            counter_split = 0
  
            for split in range(len(self.load('X_hist'))):
                rf_model = rf_models[split]
                tmp_split = []
                if not rf_model.finished:
                    counter_dt = 0
                    for decision_tree in rf_model.decision_trees:
                        tmp_dt = []
                        if not decision_tree.finished:
                            nodes = decision_tree.cur_depth_nodes
                            for node in range(len(nodes)):
                                split_scores = [np.array(data[i][counter_split][counter_dt][node]) \
                                        for i in range(len(data))]
                                sum_split_score = np.sum(split_scores, axis=0)
                                best_split = [np.unravel_index(np.argmin(sum_split_score), \
                                            sum_split_score.shape), np.min(sum_split_score)]
                                tmp_dt.append(best_split)
                            counter_dt = counter_dt + 1
                            tmp_split.append(tmp_dt)
                    counter_split = counter_split + 1
                    if len(tmp_split) > 0:
                        global_splits.append(tmp_split)
            self.broadcast_data(global_splits, send_to_self=False)
        else:
            global_splits = self.await_data()

        rf_models = self.load('rf_models')
        depth = self.load('depth')
        max_depth = self.load('max_depth')
        X_hist = self.load('X_hist')
        y = self.load('y')
        counter_split = 0
        
        for split in range(len(X_hist)):
            rf_model = rf_models[split]
            if not rf_model.finished:
                counter_dt = 0
                for decision_tree in rf_model.decision_trees:
                    if not decision_tree.finished:
                        depth_nodes = decision_tree.cur_depth_nodes
                        next_depth_nodes = []
                        for dn, node in enumerate(depth_nodes):
                            node.feature = decision_tree.feat_idcs[global_splits\
                                                            [counter_split][counter_dt][dn][0][0]]
                            node.threshold = global_splits[counter_split][counter_dt][dn][0][1]
                            node.score = global_splits[counter_split][counter_dt][dn][1]
                        
                            left_idcs = np.where(X_hist[split][node.samples, node.feature] <= \
                                                 node.threshold)[0]
                            right_idcs = np.where(X_hist[split][node.samples, node.feature] > \
                                                  node.threshold)[0]
                            left_child = Node(depth+1, node.samples[left_idcs])
                            right_child = Node(depth+1, node.samples[right_idcs])

                            if((len(left_idcs) == 0) or len(np.unique(y[split][node.samples\
                                                                            [left_idcs]])) == 1):
                                left_child.local_leaf = True
                            if((len(right_idcs) == 0) or len(np.unique(y[split][node.samples\
                                                                            [right_idcs]])) == 1):
                                right_child.local_leaf = True
                                
                            node.left = left_child
                            node.left.parent = node
                            node.right = right_child
                            node.right.parent = node
                            next_depth_nodes.append(left_child)
                            next_depth_nodes.append(right_child)
                        
                        decision_tree.next_depth_nodes = next_depth_nodes

                        counter_dt = counter_dt + 1
                
                counter_split = counter_split + 1

        self.store('rf_models', rf_models)
        self.store('depth', depth+1)

        self.update(message=f'Depth {depth+1} of {max_depth}', progress=float(depth / max_depth))

        return 'local_stopping_criteria'
    
   
@app_state('local_stopping_criteria', Role.BOTH)
class LocalStoppingCriteria(AppState):
    """
    Check if a node is already a leaf node.
    """

    def register(self):
        self.register_transition('stopping_criteria', Role.BOTH)

    def run(self) -> str or None:
        rf_models = self.load('rf_models')
        stopping_criteria = []

        for split in range(len(self.load('X_hist'))):
            tmp_split = []
            rf_model = rf_models[split]
            if not rf_model.finished:
                for decision_tree in rf_model.decision_trees:
                    if not decision_tree.finished:
                        local_leaves = list(map(lambda node: 1 if node.local_leaf else 0, \
                                                decision_tree.next_depth_nodes))
                        n_samples = [len(node.samples) for node in decision_tree.next_depth_nodes]
                        tmp_split.append([local_leaves, n_samples])
                
                if len(tmp_split) > 0:
                    stopping_criteria.append(tmp_split)
        self.send_data_to_coordinator(stopping_criteria)
    
        return 'stopping_criteria'


@app_state('stopping_criteria', Role.BOTH)
class StoppingCriteria(AppState):

    """
    The coordinator receives from each participant if a node is already a local leaf node
    and aggregates the information to check if a node is a global leaf node.
    The participants reveive information whether to stop or continue building the decision tree.
    """

    def register(self):
        self.register_transition('find_local_splits', Role.BOTH)
        self.register_transition('compute_global_leaves', Role.BOTH)

    def run(self) -> str or None:

        if self.is_coordinator:
            # Aggregate stopping criteria
            rf_models = self.load('rf_models')
            min_samples_split = self.load('min_samples_split')
            min_samples_leaf = self.load('min_samples_leaf')

            data = self.gather_data()
            cur_global_leaves = []
            next_global_leaves = []
            del_next = []
            counter_split = 0

            for split in range(len(self.load('X_hist'))):
                rf_model = rf_models[split]
                if not rf_model.finished: 
                    split_cur_global_leaves = []
                    split_next_global_leaves = []
                    split_del_next = []   
                    counter_dt = 0

                    for decision_tree in rf_model.decision_trees:
                        if not decision_tree.finished:
                            dt_cur_global_leaves = np.empty((0,))
                            dt_next_global_leaves = np.empty((0,)) 
                            dt_del_next = np.empty((0,)) 

                            n_samples = [np.array(data[i][counter_split][counter_dt][1]) for i in \
                                        range(len(data))]
                            aggr_n_samples = np.sum(n_samples, axis=0)

                            idcs_min_split = np.where(aggr_n_samples < min_samples_split)[0]
                            if len(idcs_min_split) > 0:
                                dt_next_global_leaves = np.union1d(dt_next_global_leaves, \
                                                               idcs_min_split)
                            
                            idcs_min_leaf = np.where(aggr_n_samples < min_samples_leaf)[0]
                            if len(idcs_min_leaf) > 0:
                                parent = np.floor(idcs_min_leaf / 2)
                                dt_cur_global_leaves = np.union1d(dt_cur_global_leaves, parent)
                                dt_del_next = np.union1d(dt_del_next, 2 * parent)
                                dt_del_next = np.union1d(dt_del_next, 2 * parent + 1)
                        
                            n_local_leaves = [np.array(data[i][counter_split][counter_dt][0]) for i \
                                            in range(len(data))]
                            aggr_n_local_leaves = np.sum(n_local_leaves, axis=0)
                            idcs_all_local_leaves = np.where(aggr_n_local_leaves == \
                                                         len(self.clients))[0]
                            if len(idcs_all_local_leaves) > 0:
                                dt_next_global_leaves = np.union1d(dt_next_global_leaves, \
                                                               idcs_all_local_leaves)
                            split_cur_global_leaves.append(dt_cur_global_leaves)
                            split_next_global_leaves.append(dt_next_global_leaves)
                            split_del_next.append(dt_del_next)
                            counter_dt = counter_dt + 1

                    cur_global_leaves.append(split_cur_global_leaves)
                    next_global_leaves.append(split_next_global_leaves)
                    del_next.append(split_del_next)
                
                    counter_split = counter_split + 1
            data = [cur_global_leaves, next_global_leaves, del_next]
            self.broadcast_data(data, send_to_self=False)

        else:
            data = self.await_data()

        rf_models = self.load('rf_models')
        depth = self.load('depth')
        max_depth = self.load('max_depth')

        counter_split = 0

        for split in range(len(self.load('X_hist'))):
            rf_model = rf_models[split]
            if not rf_model.finished:
                counter_dt = 0
                for decision_tree in rf_model.decision_trees:
                    if not decision_tree.finished:
                        cur_depth_nodes = decision_tree.cur_depth_nodes
                        next_depth_nodes = decision_tree.next_depth_nodes
                        global_cur_depth_nodes = data[0][counter_split][counter_dt]
                        global_next_depth_nodes = data[1][counter_split][counter_dt]
                        del_next = data[2][counter_split][counter_dt]

                        for node in global_cur_depth_nodes.astype(int):
                            cur_depth_nodes[node].global_leaf = True
                            decision_tree.leaves.append(cur_depth_nodes[node])
                        
                        for node in global_next_depth_nodes.astype(int):
                            next_depth_nodes[node].global_leaf = True
                            decision_tree.leaves.append(next_depth_nodes[node])
                        
                        for node in del_next.astype(int):
                            next_depth_nodes[node].parent.left = None
                            next_depth_nodes[node].parent.right = None
                            next_depth_nodes[node].parent = None

                        remove_from_next = np.concatenate((global_next_depth_nodes, del_next))

                        new_next_depth_nodes = [node for idx, node in enumerate(next_depth_nodes) \
                                                if idx not in remove_from_next]
                        
                        if max_depth is not None and depth == max_depth:
                            decision_tree.next_depth_nodes = []

                        elif len(new_next_depth_nodes) == 0:
                            decision_tree.finished = True
                            decision_tree.cur_depth_nodes = []
                            decision_tree.next_depth_nodes = []

                        else:
                            decision_tree.cur_depth_nodes = new_next_depth_nodes
                            decision_tree.next_depth_nodes = []

                        counter_dt = counter_dt + 1

                if all(decision_tree.finished for decision_tree in rf_model.decision_trees):
                    rf_model.finished = True
                
                counter_split = counter_split + 1

        all_finished = all(rf_model.finished for rf_model in rf_models)

        if all_finished or max_depth is not None and depth == max_depth:
            self.update(message='Get leaf nodes')
            return 'compute_global_leaves'
        
        return 'find_local_splits'


@app_state('compute_global_leaves', Role.BOTH)
class ComputeGlobalLeavesState(AppState):
    """
    Each participant calculates the leaf node values and sends them to the coordinator.
    """

    def register(self):
        self.register_transition('construct_global_rf', Role.BOTH)

    def run(self) -> str or None:
        rf_models = self.load('rf_models')
        y = self.load('y')
        classes = self.load('classes')
        leave_values = []

        if self.load('max_depth') is not None:
            for split in range(len(self.load('X_hist'))):
                rf_model = rf_models[split]
                for decision_tree in rf_model.decision_trees:
                    decision_tree.leaves.extend(decision_tree.cur_depth_nodes)
                    for node in decision_tree.cur_depth_nodes:
                        node.global_leaf = True
        
        for split in range(len(self.load('X_hist'))):
            rf_model = rf_models[split]
            tmp_split = []
            for decision_tree in rf_model.decision_trees:
                tmp_dt = []
                for leaf in decision_tree.leaves:
                    if self.load('prediction_mode') == 'classification':
                        labels = np.sum(y[split][leaf.samples][:, np.newaxis] == classes, axis=0)
                    else:
                        if len(y[split][leaf.samples]) > 0:
                            labels = [np.mean(y[split][leaf.samples]), 1]
                        else:
                            labels = [0, 0]
                    tmp_dt.append(labels)
                tmp_split.append(tmp_dt)
            leave_values.append(tmp_split)

        self.send_data_to_coordinator(leave_values)
        
        return 'construct_global_rf'


@app_state('construct_global_rf', Role.BOTH)
class ConstructGlobalLeavesState(AppState):
    """
    The coordinator aggregates the leaf node values and sends the global values to each 
    participant.
    Construct global RandomForest(s) and set samples to None for privacy.
    """

    def register(self):
        self.register_transition('calculate_local_oob', Role.BOTH)
        self.register_transition('write', Role.BOTH)
       
    def run(self) -> str or None:
        if self.is_coordinator:
            gathered_data = self.gather_data()
            leaf_values = []

            for split in range(len(self.load('X_hist'))):
                tmp_split = []
                for dt in range(self.load('n_estimators')):
                    local_values = [np.array(gathered_data[j][split][dt]) for j in \
                                    range(len(gathered_data))]
                    summed_values = np.sum(local_values, axis=0)
                
                    if self.load('prediction_mode') == 'classification':
                        global_values = [np.argmax(summed_values[i]) for i in range(len(summed_values))]
                    else:                
                        values = np.array([summed_values[i][0] for i in range(len(summed_values))])
                        n_clients = np.array([summed_values[i][1] for i in range(len(summed_values))])                    
                        global_values = values / n_clients
                
                    tmp_split.append(global_values)
                leaf_values.append(tmp_split)  

            self.broadcast_data(leaf_values, send_to_self=False)
        
        else:
            leaf_values = self.await_data()
        
        rf_models = self.load('rf_models')
        classes = self.load('classes')

        for split in range(len(self.load('X_hist'))):
            rf_model = rf_models[split]
            for dt, decision_tree in enumerate(rf_model.decision_trees):
                decision_tree.samples = None
                for l, leaf in enumerate(decision_tree.leaves):
                    if self.load('prediction_mode') == 'classification':
                        leaf.value = classes[leaf_values[split][dt][l]]
                    else:
                        leaf.value = leaf_values[split][dt][l]
        
        if self.load('oob'):
            return 'calculate_local_oob'
        
        return 'write'


@app_state('calculate_local_oob', Role.BOTH)
class CalculateLocalOOBState(AppState):

    def register(self):
        self.register_transition('get_global_oob', Role.BOTH)

    def run(self) -> str or None:
        rf_models = self.load('rf_models')
        X_hist = self.load('X_hist')
        y = self.load('y')
        local_oob_error = []
        for split in range(len(self.load('X_hist'))):
            rf_model = rf_models[split]
            tmp_split = []
            for decision_tree in rf_model.decision_trees:
                all = np.arange(len(y[split]))
                oob_samples = all[~np.isin(all, decision_tree.samples)]
                y_pred = decision_tree.predict(X_hist[split][oob_samples])
                y_true = y[split][oob_samples]
                oob_error = np.sum(y_pred != y_true)
                tmp_split.append([oob_error, len(y[split])])
            local_oob_error.append(tmp_split)

        self.send_data_to_coordinator(local_oob_error)
        
        return 'get_global_oob'


@app_state('get_global_oob', Role.BOTH)
class AggregateOOBState(AppState):

    def register(self):
        self.register_transition('write', Role.BOTH)

    def run(self) -> str or None:
        if is_coordinator:
            gathered_data = self.gather_data()
            weights = []
            for split in range(len(self.load('X_hist'))):
                tmp_dt = []
                for dt in range(self.load('n_estimators')):
                    local_values = [np.array(gathered_data[j][split][dt]) for j in \
                                    range(len(gathered_data))]
                    oob = np.sum(local_values, axis=0)                 
                    global_oob = oob[0] / oob[1]
                    global_acc = 1 - global_oob
                    tmp_dt.append(global_acc)
            
                normalized_oob_acc = tmp_dt / np.sum(tmp_dt)
                weights.append(normalized_oob_acc)
        
            self.broadcast_data(weights, send_to_self=False)
        
        else:
            weights = self.await_data()

        rf_models = self.load('rf_models')
        for split in range(len(self.load('X_hist'))):
            rf_model = rf_models[split]
            for dt, decision_tree in enumerate(rf_model.decision_trees):
                decision_tree.weight = weights[split][dt]
        return 'write'


@app_state('write', Role.BOTH)
class WriteState(AppState):
    """
    Save the trained RandomForest(s) to a file.
    """

    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self) -> str or None:
        self.update(message='Writing Output')
        rf_models = self.load('rf_models')
        X_test = self.load('X_test')
        y_true = self.load('y_test')

        def write_output(path, data):
            df = pd.DataFrame(data=data)
            df.to_csv(path, index=False, sep=self.load('sep'))

        base_dir_in = os.path.normpath(os.path.join(f'/mnt/input/', self.load('split_dir')))
        base_dir_out = os.path.normpath(os.path.join(f'/mnt/output/', self.load('split_dir')))

        if self.load('split_mode') == 'directory':
            for i, split_name in enumerate(os.listdir(base_dir_in)):
                rf_model = rf_models[i]
                y_pred = rf_model.predict(X_test[i])
                os.makedirs(os.path.join(base_dir_out, split_name), exist_ok=True)
                write_output(os.path.join(base_dir_out, split_name, self.load('pred')), \
                             {'pred': y_pred})
                write_output(os.path.join(base_dir_out, split_name, self.load('test_output')), \
                             {'y_true': y_true[i]})
                joblib.dump(rf_model, os.path.join(base_dir_out, split_name, 'rf_model.pkl')) 
        elif self.load('split_mode') == 'file':
            rf_model = rf_models[0]
            y_pred = rf_model.predict(X_test[0])
            write_output(os.path.join(base_dir_out, self.load('pred')), {'pred': y_pred})
            write_output(os.path.join(base_dir_out, self.load('test_output')), \
                         {'y_true': y_true[0]})
            joblib.dump(rf_model, os.path.join(base_dir_out, 'rf_model.pkl'))

        return 'terminal'
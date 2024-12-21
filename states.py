# pylint: disable=all
import numpy as np
import pandas as pd
import os
import joblib
from scipy.stats import norm
from FeatureCloud.app.engine.app import AppState, app_state, Role
from helper.io import read_config, read_files
from helper.util import validate_input_data
from RandomForest.models import RandomForest, Node
from RandomForest.splitting import split_score
from typing import Union, List
from copy import deepcopy
from src.client import FedHistRandomForestClient
from logging import getLogger

# if missing values want to be supported, check the MISSING_VALUES_SUPPORT comments
# here and in the called classes/functions
@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    TODO: description
    """

    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        ### Initializing the app
        self.update(message='Read files', progress=0.05)
        client = FedHistRandomForestClient()
        quantile_binning_aggregation = client.get_quantile_binning_aggregation()
        fixed_width_binning_bounds = client.get_fixed_width_binning_bounds()

        self.send_data_to_coordinator([quantile_binning_aggregation, fixed_width_binning_bounds,
                                       list(client.feature_names), list(client.quantile_idcs), list(client.fixed_width_idcs)])
        if self.is_coordinator:
            ### Binning: Get means, split points (fixed-width-binning) and sample counts
            gathered_initial_data = self.gather_data()
            # clients x [quantile_binning_aggregation, fixed_width_binning_bounds,
            #   feature_names, quantile_idcs, fixed_width_idcs]
            quantile_binning_aggregation = [gathered_initial_data[i][0] for i in range(len(gathered_initial_data))]
            fixed_width_binning_bounds = [gathered_initial_data[i][1] for i in range(len(gathered_initial_data))]
            feature_names = [gathered_initial_data[i][2] for i in range(len(gathered_initial_data))]
            quantile_idcs = [gathered_initial_data[i][3] for i in range(len(gathered_initial_data))]
            fixed_width_idcs = [gathered_initial_data[i][4] for i in range(len(gathered_initial_data))]
            client.coord_ensure_config_alignment(feature_names=feature_names,
                                                quantile_idcs=quantile_idcs,
                                                fixed_width_idcs=fixed_width_idcs)
            client.coord_ensure_same_num_splits(fixed_width_binning_bounds=fixed_width_binning_bounds,
                                                quantile_binning_aggregation=quantile_binning_aggregation)
            split_points_fixed_width = \
                client.coord_calculate_global_fixed_width_binning_splitpoints(fixed_width_binning_bounds)
            global_means, global_sample_counts = \
                client.coord_calculate_global_mean_count(
                    data=quantile_binning_aggregation)
            self.broadcast_data([global_means, split_points_fixed_width, global_sample_counts], send_to_self=True)

        ### Binning: calculate local stddevs
        means, global_fixed_width_splitpoints, sample_counts = tuple(self.await_data())
        client.set_fixed_witdh_bins(global_fixed_width_splitpoints)
        stddevs = client.calc_local_stddev(means, sample_counts)
        self.send_data_to_coordinator(stddevs)

        if self.is_coordinator:
            ### aggregate stddevs
            gathered_stddevs = self.gather_data()
            global_stddevs_coord = client.coord_aggregate_stddevs(gathered_stddevs)
            self.broadcast_data(global_stddevs_coord, send_to_self=True)

        ### Binning: perform the quantile binning (z-score normalization)
        global_stddevs: List[List[float]] = self.await_data()
            # splits x features_quantile
        client.set_quantilie_bins(global_stddevs=global_stddevs)

        ### Model initialization
        class_freqs = client.get_class_frequencies()
        self.send_data_to_coordinator(class_freqs, memo="classes")
        if self.is_coordinator:
            class_frequencies_clients = self.gather_data(memo="classes")
                # clients x splits x dict[class_i] = frequency
            client.coord_set_class_weights(class_frequencies_clients)
            RF_feat_idcs = client.coord_get_RF_feat_idcs()
                # n_estimators x max_features
                # for each tree in the random forest the feature indices
                # are randomly choosen
            self.broadcast_data([RF_feat_idcs, client.get_available_classes(), client.get_class_weights()], send_to_self=True)

        RF_feat_idcs, classes, weights = tuple(self.await_data())
        client.set_RF_feat_idcs(RF_feat_idcs)
        client.set_available_classes(classes)
        client.set_class_weights(weights)
        client.init_forest()
        #TODO: continue here
        # Missing bugs to fix:
        # The splitscore should be calculated correctly using the correct sampleset, not always
        # the full decision trees sampleset
        # The aggregation of the splitscores should take into account the number of samples correctly
        # (right now all clients are weighted equally)
        # Potentially bugs in the stopping criteria?










        return 'terminal'

@app_state('find_local_splits', Role.BOTH)
class LocalSplitState(AppState):
    """
    Each participants calculates split score for each feature-threshold combination and send the
    split scores to the coordinator.
    """

    def register(self):
        self.register_transition('aggregate_splits', Role.BOTH)

    def run(self):
        rf_models = self.load('rf_models')
        X_hist = self.load('X_hist')
        y = self.load('y')
        n_bins = self.load('n_bins')
        local_splits = []
            # split x tree x nodes_current_depth x feature x n_bins

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
                                local_split_score = split_score(X_hist[split][decision_tree.samples],
                                                y[split][decision_tree.samples],
                                                decision_tree.feat_idcs,
                                                n_bins,
                                                self.load('prediction_mode'),
                                                classes=self.load('classes'),
                                                weights=self.load('weights')[split])
                            else:
                                # leaf node
                                local_split_score = [[0] * n_bins for _ in \
                                                     range(len(decision_tree.feat_idcs))]
                                # we set the score of 0 for leaf nodes
                                # dimensionality needs to fit, we have
                                # feat_idcs x n_bins
                            tmp_dt.append(local_split_score)
                    if len(tmp_dt) > 0:
                        tmp_split.append(tmp_dt)
            if len(tmp_split) > 0:
                local_splits.append(tmp_split)
        self.send_data_to_coordinator(local_splits)
            # split x tree x nodes_current_depth x feature x n_bins

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

    def run(self):
        if self.is_coordinator:
            rf_models = self.load('rf_models')
            data = self.gather_data()
                # split x tree x nodes_current_depth x feature x n_bins
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

    def run(self):
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

    def run(self):

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

    def run(self):
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

    def run(self):
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

    def run(self):
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

    def run(self):
        if self.is_coordinator:
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

    def run(self):
        self.update(message='Writing Output')
        rf_models = self.load('rf_models')
        X_test = self.load('X_test')
        y_true = self.load('y_test')
        output_mode = self.load('output_mode')

        def write_output(path, data):
            df = pd.DataFrame(data=data)
            df.to_csv(path, index=False, sep=self.load('sep'))

        base_dir_in = os.path.normpath(os.path.join('/mnt/input/', self.load('split_dir')))
        base_dir_out = os.path.normpath(os.path.join('/mnt/output/', self.load('split_dir')))

        if self.load('split_mode') == 'directory':
            for i, split_name in enumerate(os.listdir(base_dir_in)):
                rf_model = rf_models[i]
                if output_mode in ['pred', 'model+pred']:
                    y_pred = rf_model.predict(X_test[i])
                    os.makedirs(os.path.join(base_dir_out, split_name), exist_ok=True)
                    write_output(os.path.join(base_dir_out, split_name, self.load('pred')), \
                                {'pred': y_pred})
                    write_output(os.path.join(base_dir_out, split_name, self.load('test_output')), \
                                {'y_true': y_true[i]})
                if output_mode in ['model', 'model+pred']:
                    joblib.dump(rf_model, os.path.join(base_dir_out, split_name, 'rf_model.pkl'))
        elif self.load('split_mode') == 'file':
            rf_model = rf_models[0]
            if output_mode in ['pred', 'model+pred']:
                y_pred = rf_model.predict(X_test[0])
                write_output(os.path.join(base_dir_out, self.load('pred')), {'pred': y_pred})
                write_output(os.path.join(base_dir_out, self.load('test_output')), \
                            {'y_true': y_true[0]})
            if output_mode in ['model', 'model+pred']:
                joblib.dump(rf_model, os.path.join(base_dir_out, 'rf_model.joblib'))
            # as a user potentially has no access to the rf_model class
            # they cannot use the model yet
            # Therefore, we also save the source code of the model class to the
            # output directory
            # this is a bit hacky tbh, but probably the easiest to be able
            # to really actually use the model
            import RandomForest.models as model_definition
            import inspect

            if output_mode in ['model', 'model+pred']:
                with open(os.path.join(base_dir_out, 'rf_model.py'), 'w') as f:
                    f.write(inspect.getsource(model_definition))

        return 'terminal'

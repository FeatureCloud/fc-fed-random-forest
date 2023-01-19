import random
from enum import Enum

import pandas as pd
from FeatureCloud.app.engine.app import AppState, app_state, Role, SMPCOperation
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score

from classification.models import RandomForestClassifier, Node
from classification.splitting import gini_split
from helper.io import read_config, read_files, write_config, write_test, write_model, read_test, write_pred

USE_SMPC = True


@app_state('initial', Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition('find_local_splits', Role.BOTH)

    def run(self) -> str or None:
        self.update(message=f'Initialize model', progress=0.05)
        self.log('Read config-file...')
        train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, n_estimators, max_depth, \
            max_features, max_samples, bootstrap, prediction_mode, random_state = read_config()

        self.log('Read data...')
        X, y, X_test, y_test = read_files(train, test_input, label_col)

        self.log('Initialize forest...')
        if prediction_mode == 'classification':
            rf_model: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators,
                                                                      random_state=random_state, max_depth=max_depth,
                                                                      bootstrap=bootstrap, subsample_size=max_samples)
            rf_model.init(X, y, max_features, max_depth)

        else:
            raise Exception('Only classification is supported for now')

        self.store('test_input', test_input)
        self.store('sep', sep)
        self.store('pred', pred)
        self.store('test_output', test_output)
        self.store('label_col', label_col)
        self.store('rf_model', rf_model)
        self.store('depth', 0)

        return 'find_local_splits'


@app_state('find_local_splits', Role.BOTH)
class LocalSplitState(AppState):

    def register(self):
        self.register_transition('aggregate_splits', Role.COORDINATOR)
        self.register_transition('obtain_splits', Role.PARTICIPANT)

    def run(self) -> str or None:
        rf_model: RandomForestClassifier = self.load('rf_model')
        depth = self.load('depth')
        best_local_splits = []
        for idx, decision_tree in enumerate(rf_model.decision_trees):
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes_of_depth(depth=depth)
            best_tree_splits = []
            for leaf_node in leaf_nodes:
                best_tree_splits.append(gini_split(X=leaf_node.X, y=leaf_node.y, n_classes=decision_tree.n_classes,
                                                   feat_indices=leaf_node.feat_indices))
            best_local_splits.append(best_tree_splits)

        self.configure_smpc(exponent=6, operation=SMPCOperation.ADD)
        self.send_data_to_coordinator(best_local_splits, use_smpc=USE_SMPC)
        if self.is_coordinator:
            return 'aggregate_splits'
        else:
            return 'obtain_splits'


@app_state('aggregate_splits', Role.COORDINATOR)
class AggregateSplitState(AppState):

    def register(self):
        self.register_transition('obtain_splits', Role.COORDINATOR)

    def run(self) -> str or None:
        s = self.aggregate_data(SMPCOperation.ADD, use_smpc=USE_SMPC)
        best_global_splits = []
        for idx, tree_results in enumerate(s):
            best_tree_splits = []
            for leaf_result in tree_results:
                df = pd.DataFrame(leaf_result, columns=['threshold', 'gini_index']) / len(self.clients)
                df = df.reset_index(drop=False)
                try:
                    best_tree_splits.append(df[df.gini_index == df.gini_index.min()].iloc[0, :3].to_list())
                except IndexError:
                    best_tree_splits.append(None)
            best_global_splits.append(best_tree_splits)

        self.broadcast_data(best_global_splits)

        return 'obtain_splits'


@app_state('obtain_splits', Role.BOTH)
class ObtainSplitState(AppState):

    def register(self):
        self.register_transition('find_local_splits')
        self.register_transition('calculate_local_leafs')

    def run(self) -> str or None:
        rf_model: RandomForestClassifier = self.load('rf_model')
        depth: int = self.load('depth')
        best_global_splits = self.await_data()
        updated_trees = []
        for tree_idx, decision_tree in enumerate(rf_model.decision_trees):
            best_split_per_tree = best_global_splits[tree_idx]
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes_of_depth(depth=depth)
            for leaf_idx, leaf_node in enumerate(leaf_nodes):
                if best_split_per_tree[leaf_idx] is not None:
                    feature_id = int(best_split_per_tree[leaf_idx][0])
                    thr = best_split_per_tree[leaf_idx][1]
                    gini = best_split_per_tree[leaf_idx][2]

                    indices_left = leaf_node.X[:, feature_id] < thr
                    X_left, y_left = leaf_node.X[indices_left], leaf_node.y[indices_left]
                    X_right, y_right = leaf_node.X[~indices_left], leaf_node.y[~indices_left]

                    leaf_node.feature_index = feature_id
                    leaf_node.threshold = thr
                    leaf_node.gini_index = gini

                    if gini > 0:
                        if decision_tree.random_state is not None:
                            decision_tree.random_state += 1
                        random.seed(decision_tree.random_state)
                        feat_indices = random.sample(range(rf_model.n_features), rf_model.max_features)
                        leaf_node.left = Node(X=X_left, y=y_left, depth=depth+1, feat_indices=feat_indices)

                        if decision_tree.random_state is not None:
                            decision_tree.random_state += 1
                        random.seed(decision_tree.random_state)
                        feat_indices = random.sample(range(rf_model.n_features), rf_model.max_features)
                        leaf_node.right = Node(X=X_right, y=y_right, depth=depth+1, feat_indices=feat_indices)
                    else:
                        leaf_node.X = None
                        leaf_node.y = None
                else:
                    leaf_node.X = None
                    leaf_node.y = None

            updated_trees.append(decision_tree)
        rf_model.decision_trees = updated_trees

        self.store('rf_model', rf_model)
        self.store('depth', depth+1)

        self.update(message=f'Depth {depth} of {rf_model.max_depth}', progress=float(depth / rf_model.max_depth))

        if rf_model.max_depth is not None and depth+1 == rf_model.max_depth:
            self.update(message='Calculate leaf nodes')
            return 'calculate_local_leafs'
        else:
            return 'find_local_splits'


@app_state('calculate_local_leafs', Role.BOTH)
class LocalLeafState(AppState):

    def register(self):
        self.register_transition('aggregate_leafs', Role.COORDINATOR)
        self.register_transition('obtain_leafs', Role.PARTICIPANT)

    def run(self) -> str or None:
        rf_model: RandomForestClassifier = self.load('rf_model')

        local_leafs = []
        for idx, decision_tree in enumerate(rf_model.decision_trees):
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes()
            local_tree_leafs = []
            for leaf_node in leaf_nodes:
                local_tree_leafs.append([int(np.sum(leaf_node.y == i)) for i in range(decision_tree.n_classes)])
            local_leafs.append(local_tree_leafs)
        self.configure_smpc(exponent=1, operation=SMPCOperation.ADD)
        self.send_data_to_coordinator(local_leafs, use_smpc=USE_SMPC)
        if self.is_coordinator:
            return 'aggregate_leafs'
        else:
            return 'obtain_leafs'


@app_state('aggregate_leafs', Role.COORDINATOR)
class AggregateLeafState(AppState):

    def register(self):
        self.register_transition('obtain_leafs', Role.COORDINATOR)

    def run(self) -> str or None:
        s = self.aggregate_data(SMPCOperation.ADD, use_smpc=USE_SMPC)
        self.log(s)
        local_leafs = []
        for tree_results in s:
            local_tree_leafs = []
            for leaf_result in tree_results:
                local_tree_leafs.append(np.argmax(leaf_result))
            local_leafs.append(local_tree_leafs)

        self.broadcast_data(local_leafs)

        return 'obtain_leafs'


@app_state('obtain_leafs', Role.BOTH)
class ObtainLeafState(AppState):

    def register(self):
        self.register_transition('write')

    def run(self) -> str or None:
        self.log('Obtain leafs')
        rf_model: RandomForestClassifier = self.load('rf_model')

        global_leafs = self.await_data()
        updated_trees = []
        for tree_idx, decision_tree in enumerate(rf_model.decision_trees):
            tree_leafs = global_leafs[tree_idx]
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes()
            for leaf_idx, leaf_node in enumerate(leaf_nodes):
                leaf_node.X = None
                leaf_node.y = None
                leaf_node.predicted_class = tree_leafs[leaf_idx]

            updated_trees.append(decision_tree)
        rf_model.decision_trees = updated_trees

        self.store('rf_model', rf_model)

        return 'write'


@app_state('write', Role.BOTH)
class WriteState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self) -> str or None:
        self.update(message='Writing output')
        rf_model = self.load('rf_model')

        test_input = self.load('test_input')
        sep = self.load('sep')
        pred = self.load('pred')
        label_col = self.load('label_col')
        test_output = self.load('test_output')

        X_test = read_test(test_input, sep)
        y_test = X_test.loc[:, label_col]
        X_test = X_test.drop(label_col, axis=1)
        y_pred = rf_model.predict(X_test)
        mcc = round(matthews_corrcoef(y_test, y_pred), 3)
        acc = round(accuracy_score(y_test, y_pred), 3)

        #write_config()
        #write_test(test_input, sep, test_output, label_col)
        write_pred(y_pred, y_test, pred, sep)
        #write_model(rf_model)

        self.update(message=f'MCC={mcc}; ACC={acc}', progress=1.0)

        return 'terminal'

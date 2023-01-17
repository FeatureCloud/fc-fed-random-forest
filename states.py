import random

import pandas as pd
from FeatureCloud.app.engine.app import AppState, app_state, Role, SMPCOperation
import numpy as np
from sklearn.metrics import matthews_corrcoef

from classification.models import RandomForestClassifier, Node
from classification.splitting import gini_split
from helper.io import read_config, read_files, write_config, write_test, write_model, read_test, write_pred

USE_SMPC = True


@app_state('initial', Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition('find_local_splits', Role.BOTH)

    def run(self) -> str or None:
        self.log('Read config-file...')
        train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, n_estimators, max_depth, \
            max_features, bootstrap, prediction_mode, random_state = read_config()

        self.store('test_input', test_input)
        self.store('sep', sep)
        self.store('pred', pred)
        self.store('test_output', test_output)
        self.store('label_col', label_col)

        self.log('Read data...')
        X, y, X_test, y_test = read_files(train, test_input, label_col)

        self.log('Set parameters...')
        n_features = int(X.shape[1])

        #if max_features is None or max_features == 'None' or max_features == 'none':
        #    max_features = n_features
        #elif max_features == 'sqrt':
        #    max_features = int(np.sqrt(n_features))
        #elif max_features < 1:
        #    max_features = int(n_features*max_features)

        #self.store('n_features', n_features)
        #self.store('max_features', max_features)

        self.log('Initialize forest...')
        if prediction_mode == 'classification':
            rf_model: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators,
                                                                      random_state=random_state, max_depth=max_depth,
                                                                      bootstrap=bootstrap)
            rf_model.init(X, y, n_features, max_features, max_depth)

        else:
            raise Exception('Only classification is supported for now')

        self.store('rf_model', rf_model)
        self.store('depth', 0)

        self.update(progress=0.1)

        return 'find_local_splits'


@app_state('find_local_splits', Role.BOTH)
class LocalSplitState(AppState):

    def register(self):
        self.register_transition('aggregate_splits', Role.COORDINATOR)
        self.register_transition('obtain_splits', Role.PARTICIPANT)

    def run(self) -> str or None:
        self.log('Find local split values')
        rf_model: RandomForestClassifier = self.load('rf_model')
        depth: int = self.load('depth')
        best_local_splits = []
        for idx, decision_tree in enumerate(rf_model.decision_trees):
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes()
            best_tree_splits = []
            for leaf_node in leaf_nodes:
                best_tree_splits.append(
                    gini_split(X=leaf_node.X, y=leaf_node.y, n_classes=decision_tree.n_classes,
                               n_features=rf_model.n_features, random_state=decision_tree.random_state + depth))
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
        rf_model = self.load('rf_model')
        for idx, tree_results in enumerate(s):
            best_tree_splits = []
            #random.seed(rf_model.decision_trees[idx].random_state)
            #max_features = self.load('max_features')
            #n_features = self.load('n_features')

            #feat_indices = random.sample(range(n_features), max_features)
            #print(feat_indices)
            for leaf_result in tree_results:
                df = pd.DataFrame(leaf_result, columns=['threshold', 'gini_index']) / len(self.clients)
                df = df.reset_index(drop=False)
                # df = df[df['index'].isin(feat_indices)]
                try:
                    best_tree_splits.append(df[df.gini_index == df.gini_index.min()].iloc[0, :2].to_list())
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
        self.log('Obtain best split and split nodes')
        self.update(message=f'Global aggregation')
        rf_model: RandomForestClassifier = self.load('rf_model')
        depth: int = self.load('depth')
        depth += 1
        best_global_splits = self.await_data()
        updated_trees = []
        for tree_idx, decision_tree in enumerate(rf_model.decision_trees):
            print(f'Tree {tree_idx}')
            best_split_per_tree = best_global_splits[tree_idx]
            print(best_split_per_tree)
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes()
            for leaf_idx, leaf_node in enumerate(leaf_nodes):
                if best_split_per_tree[leaf_idx] is not None:
                    m = len(leaf_node.y)
                    if m > 1:
                        num_class_parent = [np.sum(leaf_node.y == c) for c in range(decision_tree.n_classes)]
                        best_gini = 1.0 - sum((n / m) ** 2 for n in num_class_parent)

                        if m > 1 and best_gini != 0:
                            feature_id = int(best_split_per_tree[leaf_idx][0])
                            thr = best_split_per_tree[leaf_idx][1]

                            indices_left = leaf_node.X[:, feature_id] < thr
                            X_left, y_left = leaf_node.X[indices_left], leaf_node.y[indices_left]
                            X_right, y_right = leaf_node.X[~indices_left], leaf_node.y[~indices_left]

                            leaf_node.feature_index = feature_id
                            leaf_node.threshold = thr

                            leaf_node.left = Node(X=X_left, y=y_left, depth=depth)
                            leaf_node.right = Node(X=X_right, y=y_right, depth=depth)

            updated_trees.append(decision_tree)
        rf_model.decision_trees = updated_trees

        self.store('rf_model', rf_model)
        self.store('depth', depth)

        if depth == rf_model.max_depth:
            self.update(progress=0.9)
            return 'calculate_local_leafs'
        else:
            return 'find_local_splits'


@app_state('calculate_local_leafs', Role.BOTH)
class LocalLeafState(AppState):

    def register(self):
        self.register_transition('aggregate_leafs', Role.COORDINATOR)
        self.register_transition('obtain_leafs', Role.PARTICIPANT)

    def run(self) -> str or None:
        self.log('Obtain leafs... ')
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
        self.log('Aggregate leafs...')

        s = self.aggregate_data(SMPCOperation.ADD, use_smpc=USE_SMPC)

        s = np.array(s)

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
        self.update(message=f'Global aggregation of leaf classes')
        rf_model: RandomForestClassifier = self.load('rf_model')

        global_leafs = self.await_data()
        updated_trees = []
        for tree_idx, decision_tree in enumerate(rf_model.decision_trees):
            tree_leafs = global_leafs[tree_idx]
            leaf_nodes: [Node] = decision_tree.tree.get_leaf_nodes()
            for leaf_idx, leaf_node in enumerate(leaf_nodes):
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
        self.update(message='Writing output...')
        rf_model = self.load('rf_model')

        for idx, decision_tree in enumerate(rf_model.decision_trees):
            print(f'Tree {idx}')
            print(decision_tree.tree.feature_index, decision_tree.tree.threshold)
            print(decision_tree.tree.left.feature_index, decision_tree.tree.left.threshold)
            print(decision_tree.tree.right.feature_index, decision_tree.tree.right.threshold)


        test_input = self.load('test_input')
        sep = self.load('sep')
        pred = self.load('pred')
        label_col = self.load('label_col')
        test_output = self.load('test_output')
        write_model(rf_model)

        X_test = read_test(test_input, sep)
        y_test = X_test.loc[:, label_col]
        X_test = X_test.drop(label_col, axis=1)
        y_pred = rf_model.predict(X_test)

        self.log(f'MCC: {matthews_corrcoef(y_test, y_pred)}')

        write_config()
        write_test(test_input, sep, test_output, label_col)
        write_pred(y_pred, y_test, pred, sep)

        self.update(message='Finishing...')

        return 'terminal'

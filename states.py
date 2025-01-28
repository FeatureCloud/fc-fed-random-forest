# pylint: disable=all
from FeatureCloud.app.engine.app import AppState, app_state, Role
from typing import List
from src.client import FedHistRandomForestClient
from logic import main

# if missing values want to be supported, check the MISSING_VALUES_SUPPORT comments
# here and in the called classes/functions
@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    The only state used
    Trains a random forest in a federated manner.
    All clients train the same trees together. Binning is used.
    """

    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        main(self)
        return 'terminal'

    # TODO: remove this if everything is working using the function from logic.py
    # def run(self):
    #     ### Initializing the app
    #     client = FedHistRandomForestClient()
    #     oob = client.oob
    #     quantile_binning_aggregation = client.get_quantile_binning_aggregation()
    #     fixed_width_binning_bounds = client.get_fixed_width_binning_bounds()

    #     self.send_data_to_coordinator([quantile_binning_aggregation, fixed_width_binning_bounds,
    #                                    list(client.feature_names), list(client.quantile_idcs), list(client.fixed_width_idcs),
    #                                    oob], send_to_self=True)
    #     if self.is_coordinator:
    #         ### Binning: Get means, split points (fixed-width-binning) and sample counts
    #         gathered_initial_data = self.gather_data()
    #         # clients x [quantile_binning_aggregation, fixed_width_binning_bounds,
    #         #   feature_names, quantile_idcs, fixed_width_idcs]
    #         quantile_binning_aggregation = [gathered_initial_data[i][0] for i in range(len(gathered_initial_data))]
    #         fixed_width_binning_bounds = [gathered_initial_data[i][1] for i in range(len(gathered_initial_data))]
    #         feature_names = [gathered_initial_data[i][2] for i in range(len(gathered_initial_data))]
    #         quantile_idcs = [gathered_initial_data[i][3] for i in range(len(gathered_initial_data))]
    #         fixed_width_idcs = [gathered_initial_data[i][4] for i in range(len(gathered_initial_data))]
    #         oobs = [gathered_initial_data[i][5] for i in range(len(gathered_initial_data))]
    #         client.coord_ensure_config_alignment(feature_names=feature_names,
    #                                             quantile_idcs=quantile_idcs,
    #                                             fixed_width_idcs=fixed_width_idcs,
    #                                             oobs=oobs)
    #         client.coord_ensure_same_num_splits(fixed_width_binning_bounds=fixed_width_binning_bounds,
    #                                             quantile_binning_aggregation=quantile_binning_aggregation)
    #         split_points_fixed_width = \
    #             client.coord_calculate_global_fixed_width_binning_splitpoints(fixed_width_binning_bounds)
    #         global_means, global_sample_counts = \
    #             client.coord_calculate_global_mean_count(
    #                 data=quantile_binning_aggregation)
    #         self.broadcast_data([global_means, split_points_fixed_width, global_sample_counts], send_to_self=True)

    #     ### Binning: calculate local stddevs
    #     means, global_fixed_width_splitpoints, sample_counts = tuple(self.await_data())
    #     client.set_fixed_witdh_bins(global_fixed_width_splitpoints)
    #     stddevs = client.calc_local_stddev(means, sample_counts)
    #     self.send_data_to_coordinator(stddevs)

    #     if self.is_coordinator:
    #         ### aggregate stddevs
    #         gathered_stddevs = self.gather_data()
    #         global_stddevs_coord = client.coord_aggregate_stddevs(gathered_stddevs)
    #         self.broadcast_data(global_stddevs_coord, send_to_self=True)

    #     ### Binning: perform the quantile binning (z-score normalization)
    #     global_stddevs: List[List[float]] = self.await_data()
    #         # splits x features_quantile
    #     client.set_quantilie_bins(global_stddevs=global_stddevs)

    #     ### Model initialization
    #     class_freqs = client.get_class_frequencies()
    #     self.send_data_to_coordinator(class_freqs, memo="classes")
    #     if self.is_coordinator:
    #         class_frequencies_clients = self.gather_data(memo="classes")
    #             # clients x splits x dict[class_i] = frequency
    #         client.coord_set_class_weights(class_frequencies_clients)
    #         RF_feat_idcs = client.coord_get_RF_feat_idcs()
    #             # n_estimators x max_features
    #             # for each tree in the random forest the feature indices
    #             # are randomly choosen
    #         self.broadcast_data([RF_feat_idcs, client.get_available_classes(), client.get_class_weights()], send_to_self=True)

    #     RF_feat_idcs, classes, weights = tuple(self.await_data())
    #     client.set_RF_feat_idcs(RF_feat_idcs)
    #     client.set_available_classes(classes)
    #     client.set_class_weights(weights)
    #     client.init_forest()

    #     # Build the trees iteratively
    #     # two step loop:
    #     # calculate global split scores
    #     # set global leaf nodes and set current depth nodes to the next uncalculated nodes
    #     counter = 0
    #     while True:
    #         counter += 1

    #         # local scores
    #         local_scores, local_counts, only_class = client.get_current_level_splitscores()
    #             # TODO: explain this a bit
    #             # TODO: ensure the global_split_data really contains None for finished trees
    #         self.send_data_to_coordinator([local_scores, local_counts, only_class], memo=f"local_scores_{counter}")

    #         # local scores -> global scores
    #         if self.is_coordinator:
    #             result = self.gather_data(memo=f"local_scores_{counter}")
    #             global_split_scores, global_leaf_info = client.coord_aggregate_split_scores(client_split_scores=\
    #                                                     [result[i][0] for i in range(len(result))],
    #                                                 sample_count_per_client=\
    #                                                     [result[i][1] for i in range(len(result))],
    #                                                 only_class_per_client=[result[i][2] for i in range(len(result))])
    #             self.broadcast_data((global_split_scores, global_leaf_info), send_to_self=True)

    #         global_split_scores, global_leaf_info = self.await_data()
    #         # set nodes/leafs and create new nodes
    #         client.update_current_depth_nodes(
    #             global_best_split=global_split_scores,
    #             global_leaf_info=global_leaf_info
    #         )
    #         # check if we are done too escape the loop
    #         # this is done locally, but all clients should finish at the same time
    #         # as they just used the global data to set the globally synced models
    #         if client.check_finished():
    #             # finish the models by defining the leaves
    #             leaf_samples = client.get_leaf_node_samples()
    #                 # splits x n_estimators x n_leaf_nodes x Dict[class_i] = frequency
    #             self.send_data_to_coordinator(leaf_samples, send_to_self=True)
    #             if self.is_coordinator:
    #                 gathered_leaf_samples: List[List[List[List[List[int]]]]] = self.gather_data()
    #                     # clients x splits x n_estimators x n_leaf_nodes x
    #                     # List[idx: global_class_idx, val: frequency]
    #                 global_leaf_samples = client.coord_aggregate_leaf_samples(gathered_leaf_samples)
    #                 self.broadcast_data(global_leaf_samples, send_to_self=True)
    #             # update the models with the leaf nodes
    #             global_leaf_samples: List[List[List[int]]] = self.await_data()
    #             client.set_final_leaf_nodes(global_leaf_samples)
    #             break

    #     # in case of oob, calculation the local oob error (+ counts)
    #     if client.oob:
    #         # calculate the local oob error (incorrectly classified samples, total samples)
    #         oob_errors = client.calc_oob()
    #         self.send_data_to_coordinator(oob_errors, send_to_self=True)
    #         if self.is_coordinator:
    #             # aggregate them into weights (1-(sum of wrong predictions/total predictions))
    #             gathered_oob_errors = self.gather_data()
    #             weights = client.coord_aggregate_oob(gathered_oob_errors)
    #             self.broadcast_data(weights, send_to_self=True)
    #         weights = self.await_data()
    #         # update the weights
    #         client.update_weights(weights)

    #     # evaluate locally
    #     # predict and save the results
    #     client.evaluate_local()
    #     client.write_rf_model_class()
    #     return 'terminal'

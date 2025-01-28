from typing import List, Optional

from src.client import FedHistRandomForestClient
from src.helper.protocolfedlearningclass import ProtocolFedLearning

def main(protocol_fed_learning: ProtocolFedLearning,
         inputfolder: Optional[str] = None,
         outputfolder: Optional[str] = None):
    """
    This function, given the federated learning class, trains the random forest.
    This contains all the logic of the project.
    """
    ### Initializing the app
    if (inputfolder is None and outputfolder is not None) or (inputfolder is not None and outputfolder is None):
        raise ValueError("Both input and output folder must be provided or together or not provided at all")
    if inputfolder is not None and outputfolder is not None:
        client = FedHistRandomForestClient(inputfolder=inputfolder, outputfolder=outputfolder)
    else:
        client = FedHistRandomForestClient()
    oob = client.oob
    quantile_binning_aggregation = client.get_quantile_binning_aggregation()
    fixed_width_binning_bounds = client.get_fixed_width_binning_bounds()

    protocol_fed_learning.send_data_to_coordinator([quantile_binning_aggregation, fixed_width_binning_bounds,
                                    list(client.feature_names), list(client.quantile_idcs), list(client.fixed_width_idcs),
                                    oob])
    if protocol_fed_learning.is_coordinator:
        ### Binning: Get means, split points (fixed-width-binning) and sample counts
        gathered_initial_data = protocol_fed_learning.gather_data()
        # clients x [quantile_binning_aggregation, fixed_width_binning_bounds,
        #   feature_names, quantile_idcs, fixed_width_idcs]
        quantile_binning_aggregation = [gathered_initial_data[i][0] for i in range(len(gathered_initial_data))]
        fixed_width_binning_bounds = [gathered_initial_data[i][1] for i in range(len(gathered_initial_data))]
        feature_names = [gathered_initial_data[i][2] for i in range(len(gathered_initial_data))]
        quantile_idcs = [gathered_initial_data[i][3] for i in range(len(gathered_initial_data))]
        fixed_width_idcs = [gathered_initial_data[i][4] for i in range(len(gathered_initial_data))]
        oobs = [gathered_initial_data[i][5] for i in range(len(gathered_initial_data))]
        client.coord_ensure_config_alignment(feature_names=feature_names,
                                            quantile_idcs=quantile_idcs,
                                            fixed_width_idcs=fixed_width_idcs,
                                            oobs=oobs)
        client.coord_ensure_same_num_splits(fixed_width_binning_bounds=fixed_width_binning_bounds,
                                            quantile_binning_aggregation=quantile_binning_aggregation)
        split_points_fixed_width = \
            client.coord_calculate_global_fixed_width_binning_splitpoints(fixed_width_binning_bounds)
        global_means, global_sample_counts = \
            client.coord_calculate_global_mean_count(
                data=quantile_binning_aggregation)
        protocol_fed_learning.broadcast_data([global_means, split_points_fixed_width, global_sample_counts])

    ### Binning: calculate local stddevs
    means, global_fixed_width_splitpoints, sample_counts = tuple(protocol_fed_learning.await_data())
    client.set_fixed_witdh_bins(global_fixed_width_splitpoints)
    stddevs = client.calc_local_stddev(means, sample_counts)
    protocol_fed_learning.send_data_to_coordinator(stddevs)

    if protocol_fed_learning.is_coordinator:
        ### aggregate stddevs
        gathered_stddevs = protocol_fed_learning.gather_data()
        global_stddevs_coord = client.coord_aggregate_stddevs(gathered_stddevs)
        protocol_fed_learning.broadcast_data(global_stddevs_coord)

    ### Binning: perform the quantile binning (z-score normalization)
    global_stddevs: List[List[float]] = protocol_fed_learning.await_data()
        # splits x features_quantile
    client.set_quantilie_bins(global_stddevs=global_stddevs)

    ### Model initialization
    class_freqs = client.get_class_frequencies()
    protocol_fed_learning.send_data_to_coordinator(class_freqs, memo="classes")
    if protocol_fed_learning.is_coordinator:
        class_frequencies_clients = protocol_fed_learning.gather_data(memo="classes")
            # clients x splits x dict[class_i] = frequency
        client.coord_set_class_weights(class_frequencies_clients)
        RF_feat_idcs = client.coord_get_RF_feat_idcs()
            # n_estimators x max_features
            # for each tree in the random forest the feature indices
            # are randomly choosen
        protocol_fed_learning.broadcast_data([RF_feat_idcs, client.get_available_classes(), client.get_class_weights()])

    RF_feat_idcs, classes, weights = tuple(protocol_fed_learning.await_data())
    client.set_RF_feat_idcs(RF_feat_idcs)
    client.set_available_classes(classes)
    client.set_class_weights(weights)
    client.init_forest()

    # Build the trees iteratively
    # two step loop:
    # calculate global split scores
    # set global leaf nodes and set current depth nodes to the next uncalculated nodes
    counter = 0
    while True:
        counter += 1

        # local scores
        local_scores, local_counts, only_class = client.get_current_level_splitscores()
            # TODO: explain this a bit
            # TODO: ensure the global_split_data really contains None for finished trees
        protocol_fed_learning.send_data_to_coordinator([local_scores, local_counts, only_class], memo=f"local_scores_{counter}")

        # local scores -> global scores
        if protocol_fed_learning.is_coordinator:
            result = protocol_fed_learning.gather_data(memo=f"local_scores_{counter}")
            global_split_scores, global_leaf_info = client.coord_aggregate_split_scores(client_split_scores=\
                                                    [result[i][0] for i in range(len(result))],
                                                sample_count_per_client=\
                                                    [result[i][1] for i in range(len(result))],
                                                only_class_per_client=[result[i][2] for i in range(len(result))])
            protocol_fed_learning.broadcast_data((global_split_scores, global_leaf_info))

        global_split_scores, global_leaf_info = protocol_fed_learning.await_data()
        # set nodes/leafs and create new nodes
        client.update_current_depth_nodes(
            global_best_split=global_split_scores,
            global_leaf_info=global_leaf_info
        )
        # check if we are done too escape the loop
        # this is done locally, but all clients should finish at the same time
        # as they just used the global data to set the globally synced models
        if client.check_finished():
            # finish the models by defining the leaves
            leaf_samples = client.get_leaf_node_samples()
                # splits x n_estimators x n_leaf_nodes x Dict[class_i] = frequency
            protocol_fed_learning.send_data_to_coordinator(leaf_samples)
            if protocol_fed_learning.is_coordinator:
                gathered_leaf_samples: List[List[List[List[List[int]]]]] = protocol_fed_learning.gather_data()
                    # clients x splits x n_estimators x n_leaf_nodes x
                    # List[idx: global_class_idx, val: frequency]
                global_leaf_samples = client.coord_aggregate_leaf_samples(gathered_leaf_samples)
                protocol_fed_learning.broadcast_data(global_leaf_samples)
            # update the models with the leaf nodes
            global_leaf_samples: List[List[List[int]]] = protocol_fed_learning.await_data()
            client.set_final_leaf_nodes(global_leaf_samples)
            break

    # in case of oob, calculation the local oob error (+ counts)
    if client.oob:
        # calculate the local oob error (incorrectly classified samples, total samples)
        oob_errors = client.calc_oob()
        protocol_fed_learning.send_data_to_coordinator(oob_errors)
        if protocol_fed_learning.is_coordinator:
            # aggregate them into weights (1-(sum of wrong predictions/total predictions))
            gathered_oob_errors = protocol_fed_learning.gather_data()
            weights = client.coord_aggregate_oob(gathered_oob_errors)
            protocol_fed_learning.broadcast_data(weights)
        weights = protocol_fed_learning.await_data()
        # update the weights
        client.update_weights(weights)

    # evaluate locally
    # predict and save the results
    client.evaluate_local()
    client.write_rf_model_class()
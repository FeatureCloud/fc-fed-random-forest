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
from typing import Union
from copy import deepcopy

# if missing values want to be supported, check the MISSING_VALUES_SUPPORT comments
# here and in the called classes/functions
@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Read config file and input data, also validating the input

    ### Receives:
        nothing

    ### Sends:
        nothing, saves relevant input in self.store
    """

    def register(self):
        self.register_transition('local_get_binning_params1', Role.BOTH)

    def run(self):
        self.update(message='Read files', progress=0.05)
        self.log('Read config-file...')
        train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, \
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, \
            max_features, bootstrap, max_samples, random_state, prediction_mode, quantile, \
            n_bins, oob, weight_classes_bool, output_mode = read_config()
        self.log("The following parameters were read from the config file:")
        self.log(f"Train: {train}; Test: {test_input}; Prediction: {pred}; Test output: {test_output}; " +\
                 f"Separator: {sep}; Label column: {label_col}; Split mode: {split_mode}; Split directory: {split_dir}; " +\
                 f"Number of estimators: {n_estimators}; Criterion: {criterion}; Max depth: {max_depth}; " +\
                 f"Min samples split: {min_samples_split}; Min samples leaf: {min_samples_leaf}; " +\
                 f"Max features: {max_features}; Bootstrap: {bootstrap}; Max samples: {max_samples}; " +\
                 f"Random state: {random_state}; Prediction mode: {prediction_mode}; Quantile: {quantile}; " +\
                 f"Number of bins: {n_bins}; Out of bag: {oob}; Weight classes: {weight_classes_bool}; " +\
                 f"Output mode: {output_mode}")
        self.log('Read data...')
        X, y, X_test, y_test = [], [], [], []

        if split_mode == 'directory':
            num_features = None
            feature_names = None
            # multiple datasets as input from crossvalidation
            for split_name in os.listdir('/mnt/input/' + split_dir):
                # Take each folder in the split_dir as it's own dataset
                X_, y_, X_test_, y_test_, feature_names_split = read_files(os.path.join(split_dir, split_name, \
                     train), os.path.join(split_dir, split_name, test_input), sep, label_col)
                if num_features is None:
                    num_features = X_.shape[1]
                if feature_names is None:
                    feature_names = deepcopy(feature_names_split)
                        # we need to deepcpy or else featurenames is just
                        # a pointer to a pointer that get's changed every loop....
                elif feature_names != feature_names_split:
                    raise ValueError('Feature names do not match between datasets')
                validate_input_data(X_, y_, X_test_, y_test_, num_features)
                X.append(X_)
                y.append(y_)
                X_test.append(X_test_)
                y_test.append(y_test_)
            self.store('feature_names', feature_names)
        else:
            # otherwise just a single dataset
            X_, y_, X_test_, y_test_, feature_names_ = read_files(train, test_input, sep, label_col)
            num_features = X_.shape[1]
            validate_input_data(X_, y_, X_test_, y_test_, num_features)
            X.append(X_)
            y.append(y_)
            X_test.append(X_test_)
            y_test.append(y_test_)
            self.store('feature_names', feature_names_)

        if quantile and len(quantile) > 0:
            try:
                quantile = np.array(quantile, dtype=int)
            except ValueError as e:
                raise ValueError('Quantile indices must be integers') from e
        else:
            assert num_features is not None # to satisfy the pesky linter
            # in this case we consider all features for quantile binning
            quantile = np.arange(num_features)

        # calculate the num_features and max_features
        n_features = X[0].shape[1]
        if max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif max_features < 1:
            max_features = int(n_features * max_features)
        else:
            try:
                max_features = int(max_features)
            except ValueError as e:
                raise ValueError('Max features must be a float between 0 and 1, an integer or "sqrt"') from e
        self.store('max_features', max_features)
        self.store('n_features', n_features)

        np.random.seed(random_state)
        try:
            n_bins = int(n_bins)
        except ValueError as e:
            raise ValueError('Number of bins must be an integer') from e
        if n_bins < 2:
            raise ValueError('Number of bins must be at least 2')

        # Store parameters from config file
        self.store('pred', pred)
        self.store('test_output', test_output)
        self.store('sep', sep)
        self.store('label_col', label_col)
        self.store('split_mode', split_mode)
        self.store('split_dir', split_dir)
        self.store('weight_classes_bool', weight_classes_bool)
        self.store('output_mode', output_mode)

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
            if weight_classes_bool:
                raise ValueError('Weights are not supported for regression, there are no classes to weight in regression')
            self.store('oob', False)

        # Store data
        self.store('X', X)
        self.store('y', y)
        self.store('X_test', X_test)
        self.store('y_test', y_test)
        self.store('classes', np.unique(y[0]))
        self.store('depth', 0)

        return 'local_get_binning_params1'


@app_state('local_get_binning_params1', Role.BOTH)
class LocalBinningState1(AppState):
    """
    We perform two ways two find bins for which we calculate split points later:
    1. quantile binning: we z-normalize the data and can then based on
        the z-normalized data calculate the bins in a way that the bins
        contain the same number of samples
    2. fixed-width binning: we calculate the minimum and maximum values for each feature
        and bin with fixed-width bins between min and max
        For privacy reasons, we don't use the actual min/max but the mean of the
        top/bottom 5% of the values
    For the two ways to bin, quantile binning and fixed-width binning, we need
    to calculate:
    - fixed-width binning: the minimum and maximum values for each feature
    - quantile binning: the mean and standard deviation for each feature, which
        is used to normalize the data via z-score normalization
        The mean is easy and can be calculated by summing up all values and dividing
        by the sum of the number of samples.
        However, consider the formula for the standard deviation:
        stddev = sqrt(sum(x_i - mean)^2 / num_samples)
        To calculate this, we need to have the mean calculated first.

    Therefore, we first send around only the sum of values to calculate the mean
    and get the z-score normalized data and the quantile bins in a second step (get_binning_params2)

    ### Receives:
        nothing

    ### Sends:
        A Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        1. Information for quantile binning: List of matrices with two columns,
            each row_idx is a feature_idx, first column is the sum of values,
            second the number of values
        2. Information for fixed-width binning: List of matrices with two columns,
            each row_idx is a feature_idx, first column is the minimum value,
            second the maximum value
        3. List of feature names to ensure that all clients have the same features
            in the same order, else the model would not work at all
    """

    def register(self):
        self.register_transition('global_get_binning_params1', Role.COORDINATOR)
        self.register_transition('local_get_binning_params2', Role.PARTICIPANT)

    def run(self) -> Union[str, None]:

        # Normalize data for quantile binning
        X = self.load('X')
        quantile_idcs = self.load('quantile')
        local_matrix_list = []
            # List containing for each split a matrix
            # with num_features_quantile rows and with twp columns:
            # 1. Number of samples
            # 2. Sum of values per feature in the quantile_features
            # This is needed to calculate the mean per feature
        for split_idx, _ in enumerate(X):
            X_quantile = X[split_idx][:, quantile_idcs]
            num_features_quantile = X_quantile.shape[1]
            local_matrix = np.zeros((num_features_quantile, 2))
            # if num_features_quantile = 0, these will have no effect
            # as there is no row to fill
            local_matrix[:, 0] = X_quantile.shape[0] # num_rows = num_samples
                # MISSING_VALUES_SUPPORT: don't use shape, get count without missing values
            local_matrix[:, 1] = np.sum(X_quantile, axis=0) # column-wise sum -> per feature sum as one vector
            local_matrix_list.append(local_matrix)

        # Get minimum and maximum for bucket binning
        send_data_bucket = []
            # List containing for each split a matrix with two columns:
            # 1. Minimum values per feature in the bucket_features
            # 2. Maximum values per feature in the bucket_features
        num_features = len(X[0][0])
        bucket_idcs = np.setdiff1d(np.arange(num_features), quantile_idcs)
            # For bucket binning we consider all features that are not
            # used in quantile binning

        for split_idx, _ in enumerate(X):
            # get split specific data
            split_data = X[split_idx][:, bucket_idcs]
                # samples x features
            num_samples_total = split_data.shape[0]
            quantile5_end = int(num_samples_total * 0.05)
                # the last value to consider in a sorted array for the 5% quantile
            quantile95_start = int(num_samples_total * 0.95)
                # the first value to consider in a sorted array for the 95% quantile
            # MISSING_VALUES_SUPPORT: use the correct num_samples here as this changes per feature

            # get the per column mean of the top/bottom 5% of the values per feature
            split_data_sorted = np.sort(split_data, axis=0)
                # sort per column (per feature)
                # sorts ascending
                # fun fact: the documentation of np.sort does not contain the word ascending nor descending
            min_array = np.mean(split_data_sorted[:quantile5_end], axis=0)
            max_array = np.mean(split_data_sorted[quantile95_start:], axis=0)
                # column-wise mean -> per feature mean as one vector
            send_data_bucket.append(np.array([min_array, max_array]))

        # Lastly, we also send the feature names around to ensure that
        # all clients have the same features in the same order
        self.send_data_to_coordinator([local_matrix_list, send_data_bucket, self.load('feature_names')])
        if self.is_coordinator:
            return 'global_get_binning_params1'
        else:
            return 'local_get_binning_params2'



@app_state('global_get_binning_params1', Role.COORDINATOR)
class GlobalBinningState1(AppState):
    """
    Calculates the global mean for quantile binning and the fixed-width bins
    per feature. Means are needed for the standard deviation calculation, which
    is then handled by global_get_binning_params2.
    Mean and stddev are needed for z-score normalization, which is needed for
    quantile binning.

    ## Receives:
        What was sent in local_get_binning_params1

    ## Sends:
        A Tuple[List[List[float]], List[np.ndarray], List[np.ndarray]]:
        1. Per split the global mean values for quantile binning per feature (feature=index)
        2. Per split the global split points for fixed-width binning per feature (feature=index)
        3. Per split the sample count for the global stddev calculation (feature=index)
    """

    def register(self):
        self.register_transition('local_get_binning_params2', Role.COORDINATOR)

    def run(self):
        gathered_data = self.gather_data()
        # First we use the feature names and ensure that all clients have the same features
        # in the same order
        feature_names = pd.Index(gathered_data[0][2])
        for feature_names_other in gathered_data[1:]:
            feature_names_other = pd.Index(feature_names_other[2])
            if feature_names.equals(feature_names_other):
                print("Feature names don't match between clients.")
                print(f"Features client0,otherclient:\n{feature_names}\n{feature_names_other}")
                print(f"Features just in client0: {set(feature_names) - set(feature_names_other)}")
                print(f"Features just in other client: {set(feature_names_other) - set(feature_names)}")
                raise ValueError('Feature names do not match between clients')

        # Second we calculate the global mean for quantile binning
        broadcast_means = []
            # splits x features, each entry being the mean for the corresponding feature
            # globally
        sample_counts = []
            # splits x features, each entry being the number of samples for the corresponding feature
            # globally
        local_matrix_list = [gathered_data[client_idx][0] for client_idx in range(len(gathered_data))]
        # Ensure the num_splits are the same over all clients
        # since send to self is true we don;t need to look at the coordinator
        # seperately
        splits = [len(d) for d in local_matrix_list]
        if len(set(splits)) != 1:
            raise ValueError('The number of splits differ between clients')
        # calculate the global mean
        for split_idx, _ in enumerate(self.load('X')):
            try:
                data = np.array([d[split_idx] for d in local_matrix_list])
                # format is clients x num_features x 2, we removed the split
                # axis due to the split loop
            except IndexError:
                raise ValueError('The number of splits differ between clients')
            global_matrix = np.sum(data, axis=0)
                # we sum over the clients axis, new format is num_features x 2
            accumulated_sample_count = global_matrix[:, 0] # vector of shape num_features
            accumulated_sum = global_matrix[:, 1] # vector of shape num_features
            mean = accumulated_sum / accumulated_sample_count # vector of shape num_features
            broadcast_means.append(mean)
            sample_counts.append(accumulated_sample_count)

        split_points_bucket = []
            # List containing for each split, for each feature the split points
            # for fixed-width binning
            # dimensions is therefore splits x features x n_bins
            # List[List[np.ndarray(1d vector)]]
        n_bins = self.load('n_bins')
        data_bucket = [gathered_data[i][1] for i in range(len(gathered_data))]
            # format of data_bucket is clients x splits x 2 x num_features
            # for the two columns: first is min, second is max

        for split_idx, _ in enumerate(self.load('X')):
            data = [d[split_idx] for d in data_bucket]
                # we only consider this specific split
                # format of data is clients x 2 x num_features
            min_max_values = np.array(data)
                # np.array of shape clients x 2 x num_features
            min_values_all = min_max_values[:, 0, :]
            max_values_all = min_max_values[:, 1, :]
                # shape clients x num_features
            min_values = np.min(min_values_all, axis=0)
            max_values = np.max(max_values_all, axis=0)
                # reduce via min/max from clients x num_features to num_features, finding the min over all clients
            split_points_per_feature = [np.linspace(float(min_values[feature_idx]), float(max_values[feature_idx]), \
                n_bins + 1) for feature_idx in range(len(min_values))]
                # n_bins + 1 as for n_bins we need n_bins + 1 split points
                # 2 bins -> min, 1, max needed as split points
                # np.linspace includes the start and stop value
            split_points_bucket.append([split_points[1:-1] for split_points in split_points_per_feature])
                # We neither need min nor max due to how np.digitize works

        # save the sample count for the global stddev calculation
        data = [broadcast_means, split_points_bucket, sample_counts]
        self.broadcast_data(data, send_to_self=True)
        return 'local_get_binning_params2'

@app_state('local_get_binning_params2', Role.BOTH)
class LocalBinningState2(AppState):
    """
    Receives the global mean and standard deviation for quantile binning
    and the split points for fixed-width binning.

    ### Receives:
        What global_get_binning_params1 sends

    ### Sends:
        A list of dimensions splits x features_quantile with each entry being the
        local standard deviation for the corresponding feature.
    """

    def register(self):
        self.register_transition('local_calc_bins_normalize', Role.PARTICIPANT)
        self.register_transition('aggregate_stddev', Role.COORDINATOR)

    def run(self):
        means, splitpoints, sample_counts = tuple(self.await_data())
            # means is splits x features_quantile
            # splitpoints is splits x features_non_quantile x (n_bins-1)
            # sample_counts is splits x features_quantile
        means = np.array(means)
        splitpoints = np.array(splitpoints)
        sample_counts = np.array(sample_counts)
        self.store('split_points_bucket', splitpoints)

        # now that we have the global means we can calculate the local
        # std deviation for quantile binning and sent it to the coordinator for
        # the global stddev calculation
        X = self.load('X')
        quantile_idcs = self.load('quantile')
        stddevs = []
            # format is splits x num_features_quantile
            # each entry is the standard deviation for the corresponding feature
            # and split
            # Caveat: we don't send exactly the standard deviation but the sum of
            # (x_i - mean)^2
            # The global client still has the sample count from before
        for split_idx, _ in enumerate(X):
            X_quantile = X[split_idx][:, quantile_idcs]
                # samples x features_quantile
                # Extract only the features that are used for quantile binning
                # from the raw data
            # formula stddev is sqrt(sum(x_i - mean)^2 / num_samples)
            local_stddev = np.sum(((X_quantile - means[split_idx]) ** 2) / sample_counts[split_idx], axis=0)
                # X_quantile is samples x features_quantile, means[split_idx]
                # and sample_counts[split_idx] are vectors of shape features_quantile
                # broadcasting applies means and sample_counts row-wise (sample axis)
                # np.sum is used to collapse the samples axis
                # final shape becomes features_quantile vector
            stddevs.append(local_stddev)
                # stddevs gets shape splits x features_quantile with each
                # entry being the local stddev for the corresponding feature

        # save the means already so we don't need to broadcast them again later
        self.store('global_mean', means)
        self.store('sample_count', sample_counts)

        # send the local stddev to the coordinator, which can then finish
        # the global stddev calculation
        # and therefore finish the prepataion z-score normalization
        self.send_data_to_coordinator(stddevs)
        if self.is_coordinator:
            return 'aggregate_stddev'
        else:
            return 'local_calc_bins_normalize'

@app_state('aggregate_stddev', Role.COORDINATOR)
class AggregateStddevState(AppState):
    """
    Aggregates the local standard deviations and calculates the global
    standard deviation per feature for quantile binning.
    Collapses the received clients x splits x features_quantile
    to splits x features_quantile, using the sample counts from the
    previous step (global_get_binning_params1).

    ### Receives:
        What local_get_binning_params2 sends
    ### Sends:
        A list of dimensions splits x features_quantile with each entry being the
        global standard deviation for the corresponding feature.
    """

    def register(self):
        self.register_transition('local_calc_bins_normalize', Role.COORDINATOR)

    def run(self):
        stddevs = self.gather_data()
            # stddevs is clients x splits x features_quantile
        # we need to collapse the clients axis to get splits x features_quantile
        global_stddevs = []
        for split_idx, _ in enumerate(self.load('X')):
            try:
                data = np.array([d[split_idx] for d in stddevs])
                # format is clients x features_quantile
            except IndexError:
                raise ValueError('The number of splits differ between clients')
            global_stddev_split = np.sum(data, axis=0)
                # collapse the clients axis, clients x features_quantile -> features_quantile
            global_stddevs.append(global_stddev_split)
                # global_stddevs is splits x features_quantile

        self.broadcast_data(global_stddevs, send_to_self=True)
        return 'local_calc_bins_normalize'

@app_state('local_calc_bins_normalize', Role.BOTH)
class BinningGlobalState(AppState):
    """
    Z-score normalizaes data to be able to do quantile binning.
    Also creates the fixed-width bins.

    ### Receives:
        What aggregate_stddev sends

    ### Sends:
        Nothing, saves the z-score normalized data so that
        global_quantile_binning can continue with the quantile binning.
    """

    def register(self):
        self.register_transition('global_quantile_binning', Role.BOTH)

    def run(self):
        global_stddev = self.await_data()
            # global_stddevs is splits x features_quantile
        self.store('global_stddev', global_stddev)
        global_mean = self.load('global_mean')
        split_points_bucket = self.load('split_points_bucket')
            # split_points_bucket is splits x features_non_quantile x (n_bins-1)

        X = self.load('X')
        bucket_idcs = np.setdiff1d(np.arange(len(X[0][0])), self.load('quantile'))
        tmp_X_hist = []

        # Create the fixed-width bins
        for split in range(len(X)):
            split_points = np.array(split_points_bucket[split])
                # split_points is features_non_quantile x (n_bins-1)
            X_T = np.transpose(X[split][:, bucket_idcs])
                # X_T is features_non_quantile x samples
            # Assign data points to bins
            X_hist = np.array([np.digitize(X_T[feature_idx], split_points[feature_idx]) \
                                    for feature_idx in range(X_T.shape[0])])
                # Reminder: the split points are the interval ]min, 1, ..., max[
                # we do not need to supply the min and max value, as according
                # to the documentation of np.digitize:
                # If values in x are beyond the bounds of split_points,
                # 0 or len(split_points) is returned as appropriate.
                # The interval ]min, 1, ..., max[ has n_bins - 1 split points
                # therefore we end up with bin indexes 0, ..., n_bins - 1
                # which is perfect for our purposes
                # format is features_non_quantile x samples
            tmp_X_hist.append(X_hist)

        self.store('X_hist_bucket', tmp_X_hist)

        X = self.load('X')
        quantile_idcs = self.load('quantile')
        X_normalized = []

        for split in range(len(X)):
            # we z-score normalize the data
            # formula: (x_i - mean) / stddev
            dividend = (X[split][:, quantile_idcs] - global_mean[split])
            # format is samples x features_quantile
            divisor = global_stddev[split]
            normalized = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor != 0)
                # set data to 0 if stddev is 0
                # A a value of 0 means that the sum of differences between
                # the values and the mean is 0, which is the case if all
                # values are the same. All values are the same -> stddev of 0
            normalized[normalized == np.inf] = 0
            normalized[normalized == -np.inf] = 0
            normalized[normalized == np.nan] = 0
                # away with pesky missing values, they shall all be 0
                # format is samples x features_quantile
            X_normalized.append(normalized)
                # format is splits x samples x features_quantile
        self.store('X_normalized_quantile', X_normalized)

        return 'global_quantile_binning'


@app_state('global_quantile_binning', Role.BOTH)
class GlobalQuantileBinningState(AppState):
    """
    Finally perform the quantile binning, the z-score normalized data
    is available now.

    ## Receives:
        nothing

    ## Sends:
        nothing, saves the quantile binned data
    """

    def register(self):
        self.register_transition('combine_binning', Role.BOTH)

    def run(self):
        n_bins = self.load('n_bins')
        X = self.load('X_normalized_quantile')
        X_hist_list = []

        percentiles = np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1)
            # Finds the relevant percentiles for quantile binning
            # e.g. for n_bins = 2, we would want the 50% percentile
            # the min and max value of the split points are always 0 and 1
            # we can't use the min (0) and max (1) value as they are -inf and inf
            # this is why we use n_bins -1 and start at 1/n_bins, end at
            # 1 - 1/n_bins
        split_points = [norm.ppf(p) for p in percentiles]
            # reminder: the data for quantile binning is z-score normalized
            # -> we assume normally distributed data for the quantile binning
            # We now go from the percentile, e.g. from 0 to 25% of all values,
            # to the value x at which all values <= x together make up 25% of all values
            # norm.ppf does this for us
            # 1d array of length n_bins - 1

        split_points_quantile = []
        for split in range(len(X)):
            X_T = np.transpose(X[split])
                # format is features_quantile x samples
            # Assign data points to bins
            X_hist = np.array([np.digitize(X_T[i], split_points) \
                                    for i in range(X_T.shape[0])])
                # Reminder: the split points are the interval ]min, 1, ..., max[
                # we do not need to supply the min and max value, as according
                # to the documentation of np.digitize:
                # If values in x are beyond the bounds of split_points,
                # 0 or len(split_points) is returned as appropriate.
                # The interval ]min, 1, ..., max[ has n_bins - 1 split points
                # therefore we end up with bin indexes 0, ..., n_bins - 1
                # the value 0 is therefore membership of that sample for that
                # feature of the bin 0
                # which is perfect for our purposes
                # format is features_quantile x samples

            X_hist_list.append(X_hist)
            split_points_quantile.append(np.tile(split_points, (len(X[split][0]), 1)))
                # format is splits x features x n_bins - 1
                # for each feature we have the split points

        self.store('split_points_quantile', split_points_quantile)
        self.store('X_hist_quantile', X_hist_list)

        return 'combine_binning'


@app_state('combine_binning', Role.BOTH)
class CombineBinningState(AppState):
    """
    Concatenate Quantile Binning Data and Bucket Binning Data.
    Both use the same indexings, so we can just concatenate them.

    Receives:
        nothing, loads results previously calculated
    Sends:
        class_frequencies (List[Dict[int, int]]): List of dictionaries, each dictionary
            contains the class frequencies of each split. The keys are the class indices
            from the classes array. The values are the frequencies of the corresponding class.
    """

    def register(self):
        self.register_transition('feat_idcs', Role.BOTH)

    def run(self):
        X = self.load('X')
        quantile_idcs = self.load('quantile')
        bucket_idcs = np.setdiff1d(np.arange(len(X[0][0])), quantile_idcs)

        X_hist_quantile = self.load('X_hist_quantile')
        X_hist_bucket = self.load('X_hist_bucket')
        X_hist_list = []
            # format is splits x samples x features

        split_points_quantile = self.load('split_points_quantile')
        split_points_bucket = self.load('split_points_bucket')
            # split_points_bucket is splits x features_non_quantile x (n_bins-1)
        split_points_list = []

        for split in range(len(X)):
            if len(quantile_idcs) > 0 and len(bucket_idcs) > 0:
                # fixed witdh AND quantile binning
                X_hist = np.concatenate((X_hist_quantile[split], X_hist_bucket[split]))
                    # format is features x samples
                # Place the values of array at specified indices
                X_hist[quantile_idcs] = X_hist_quantile[split]
                X_hist[bucket_idcs] = X_hist_bucket[split]
                X_hist_list.append(np.transpose(X_hist))
                    # transpose to go back to the normal samples x features format

                split_points = np.tile(split_points_quantile[split], (len(X[split][0]), 1))
                    # TODO: this looks weird
                    # already before we save exactly the same thing for each
                    # feature
                    # now we save this again for each feature?
                    # so feature x feature x n_bins - 1
                    # WHY????
                    # also why do we overwrite it thhen?
                    # AMERICA EXPLAIN
                    # ALSO: we just use percentiles plus norm.ppf,
                    # SO THEY ARE ALL THE SAME FOR ANY FEATURE
                    # we save the same thing feature x feature times when
                    # we would only need it once!
                    # TODO: fix this
                # Place the values of array at specified indices
                split_points[quantile_idcs] = split_points_quantile[split]
                split_points[bucket_idcs] = split_points_bucket[split]
                split_points_list.append(split_points)

            elif len(bucket_idcs) > 0:
                # only fixed width binning
                X_hist_list.append(np.transpose(X_hist_bucket[split]))
                split_points_list.append(split_points_bucket[split])

            else:
                # only quantile binning
                X_hist_list.append(np.transpose(X_hist_quantile[split]))
                split_points_list.append(split_points_quantile[split])

        self.store('X_hist', X_hist_list)
            # format is splits x samples x features
        self.store('split_points', split_points_list)

        # As it can happen that one client does not have all classes
        # (of the predicted variable), we need to communicate the classes to all clients
        # if we weight the samples by class occurence, we also need to communicate
        # the class frequencies
        class_frequencies = list()
        y = self.load('y')
        classes = self.load('classes')
        use_weights = self.load('weight_classes_bool')
        for _y in y:
            frequency_dict = dict()
            for class_i in classes:
                if use_weights:
                    frequency_dict[class_i] = np.sum(_y == class_i)
                else:
                    frequency_dict[class_i] = 0
            class_frequencies.append(frequency_dict)

        self.send_data_to_coordinator(class_frequencies, memo="classes")

        return 'feat_idcs'


@app_state('feat_idcs', Role.BOTH)
class FeatureIndicesState(AppState):
    """
    Choose feature indices based on the max_features parameter.
    Chooses for each tree in the random forest a random subset of features.
    The same num_estimators x max_features indices are used in all splits.
    """

    def register(self):
        self.register_transition('init_forest', Role.BOTH)

    def run(self):
        self.log('Choose feature indices...')

        if self.is_coordinator:
            # not we gather which classes exist in the data
            class_frequencies = self.gather_data(memo="classes")
            # class_frequencies is a list of dictionaries, each dictionary
            # contains the class frequencies of each split
            if len(class_frequencies) <= 1:
                raise RuntimeError("Only one client exists, cannot run the app")

            # update classes to have all global classes
            classes = set()
            for class_frequency_list in class_frequencies:
                for class_frequency in class_frequency_list:
                    classes.update(class_frequency.keys())
            classes = np.array(list(classes))
            n_classes = len(classes)
            self.store('classes', classes)

            # set weights if necessary
            self.store('weights', None)
            if self.load('weight_classes_bool'):
                weights = list()
                    # list index is the split index, contains
                    # dictionaries with class_i as key and weight as value
                for split_idx, _ in enumerate(self.load("X")):
                    split_weights = dict()
                    split_total_samples = 0
                    # get the pure frequency counts per split per class
                    for client_class_frequencies in class_frequencies:
                        for class_i, frequency in client_class_frequencies[split_idx].items():
                            if class_i not in split_weights:
                                split_weights[class_i] = 0
                            split_weights[class_i] += frequency
                            split_total_samples += frequency
                    # calculate the weight from the frequency plus total samples
                    for class_i, frequency in split_weights.items():
                        if frequency != 0:
                            split_weights[class_i] = split_total_samples / (n_classes * frequency)
                            # we implement balanced class weights from the sklearn RandomForestClassifier
                            # According to their documentation:
                            # The “balanced” mode uses the values of y to automatically adjust weights
                            # inversely proportional to class frequencies in the input data as
                            # n_samples / (n_classes * np.bincount(y))
                        else:
                            split_weights[class_i] = 0
                    weights.append(split_weights)

                self.store('weights', weights)

            n_features = self.load('n_features')
            max_features = self.load('max_features')

            RF_feat_idcs = []
                # n_estimators x max_features
                # contains the feature indices randomly choosen for each
                # estimator (tree) in the random forest
            for _ in range(self.load('n_estimators')):
                feat_idcs = np.random.choice(n_features, size= \
                                                max_features, replace=False)
                RF_feat_idcs.append(feat_idcs)
            RF_feat_idcs = np.array(RF_feat_idcs)
            self.broadcast_data([RF_feat_idcs,
                                 self.load('classes'),
                                 self.load('weights')], send_to_self=False)

        else:
            RF_feat_idcs, classes, weights = tuple(self.await_data())
            self.store('classes', classes)
            self.store('weights', weights)

        self.store('RF_feat_idcs', RF_feat_idcs)
        return 'init_forest'


@app_state('init_forest', Role.BOTH)
class InitForestState(AppState):
    """
    Initialize the RandomForest(s).
    """

    def register(self):
        self.register_transition('find_local_splits', Role.BOTH)

    def run(self):
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
            raise AttributeError('Only classification and regression are valid modes.')

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

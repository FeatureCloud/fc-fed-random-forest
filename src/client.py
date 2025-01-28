"""
Contains the client class for the federated learning of a Randomforest using histograms.
Usage:
    TODO: write in which order the methods should be called by whom
"""
# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
import os
from typing import Optional, Union, List, Tuple, Dict, Any
import logging
from copy import deepcopy
import inspect

import bios
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.helper.util import validate_input_data, convert_to_np
from src.RandomForest.models import RandomForest

class FedHistRandomForestClient():
    """
    This class is used for all clientside computations in the learning of a
    federated Randomforest using histograms.
    It is meant to be used in conjuction with the coordinator class.
    #TODO: Add more information on how to use this class
    """
    def __init__(self,
                 config: Optional[dict] = None,
                 inputfolder: str = "mnt/input",
                 outputfolder: str = "mnt/output",
                 logging_class: Optional[logging.Logger] = None) -> None:
        """
        #TODO: Add docstring
        """
        # Read in all configuration parameters
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        if not logging_class:
            logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            logging_class = logging.getLogger(name='FedHistRandomForestClient')
        self.logging_class = logging_class
        self.logging_class.info('Read config-file...')
        self._read_config(config if config else {})
        self.logging_class.info("The following parameters were read from the config file:")
        self.logging_class.info(f"Train: {self.train_filename}; " +\
             f"Test: {self.test_input_filename}; " +\
             f"Prediction: {self.pred_filename}; Test output: {self.test_output_filename}; " +\
             f"Separator: {self.sep}; Label column: {self.label_col}; " +\
             f"Split mode: {self.split_mode}; "+\
             f"Split directory: {self.split_dir}; Number of estimators: {self.n_estimators}; "+\
             f"Criterion: {self.criterion}; Max depth: {self.max_depth}; " +\
             f"Min samples split: {self.min_samples_split}; " +\
             f"Max features: {self.__max_features_raw}; Bootstrap: {self.bootstrap}; " +\
             f"Max samples: {self.max_samples_raw}; Random state: {self.random_state}; " +\
             f"Prediction mode: {self.prediction_mode}; Quantile: {self.__quantile_idcs_raw}; " +\
             f"Number of bins: {self.n_bins}; Out of bag: {self.oob}; " +\
             f"Weight classes: {self.weight_classes_bool}; Output mode: {self.output_mode}")

        # Start reading in the data
        self.logging_class.info('Read data...')
        X, y, X_test, y_test = [], [], [], []

        self.num_features = 0
        if self.split_mode == 'directory':
            self.feature_names: pd.Index = pd.Index([])
            # multiple datasets as input from crossvalidation
            for split_name in os.listdir(self.inputfolder + self.split_dir):
                # Take each folder in the split_dir as it's own dataset
                train_file = os.path.join(self.inputfolder,
                                          self.split_dir,
                                          split_name,
                                          self.train_filename)
                test_input_file = os.path.join(self.inputfolder,
                                               self.split_dir,
                                               split_name,
                                               self.test_input_filename)
                X_, y_, X_test_, y_test_, feature_names_split = \
                    self._read_files(train_file, test_input_file)
                if self.num_features == 0:
                    self.num_features = X_.shape[1]
                if not self.feature_names:
                    self.feature_names = deepcopy(feature_names_split)
                        # we need to deepcpy or else featurenames is just
                        # a pointer to a pointer that get's changed every loop....
                elif self.feature_names != feature_names_split:
                    raise ValueError('Feature names do not match between datasets')
                validate_input_data(X_, y_, X_test_, y_test_, self.num_features)
                X.append(X_)
                y.append(y_)
                X_test.append(X_test_)
                y_test.append(y_test_)
        else:
            # otherwise just a single dataset
            train_file = os.path.join(self.inputfolder, self.train_filename)
            test_input_file = os.path.join(self.inputfolder, self.test_input_filename)
            X_, y_, X_test_, y_test_, self.feature_names_ = \
                self._read_files(train_file, test_input_file)
            self.num_features = X_.shape[1]
            validate_input_data(X_, y_, X_test_, y_test_, self.num_features)
            X.append(X_)
            y.append(y_)
            X_test.append(X_test_)
            y_test.append(y_test_)

        try:
            self.quantile_idcs = np.array(self.__quantile_idcs_raw, dtype=int)
        except ValueError as e:
            raise ValueError('Quantile indices must be integers') from e
        # ensure that the quantile indices are within the number of features
        if np.any(self.quantile_idcs >= self.num_features):
            raise ValueError('Quantile indices must be smaller than the number of features')
        self.fixed_width_idcs = np.setdiff1d(np.arange(self.num_features), self.quantile_idcs)

        # calculate the num_features and max_features
        self.n_features = int(X[0].shape[1]) # int function for the type checker
        if isinstance(self.__max_features_raw, str):
            if self.__max_features_raw == 'sqrt':
                self.max_features = int(np.sqrt(self.n_features))
            else:
                raise ValueError('Max features must be a float between 0 and 1, " +\
                                 "an integer or "sqrt"')
        elif self.__max_features_raw < 1:
            self.max_features = int(self.n_features * self.__max_features_raw)
        else:
            try:
                self.max_features = int(self.__max_features_raw)
            except ValueError as e:
                raise ValueError('Max features must be a float between 0 and 1, " +\
                                 "an integer or "sqrt"') from e

        np.random.seed(self.random_state)

        # set max_samples to the float format
        self.max_samples: float = 1.0
        if self.max_samples_raw is not None:
            if self.max_samples_raw <= 1 and self.max_samples_raw > 0:
                self.max_samples = float(self.max_samples_raw)
            else:
                raise ValueError('Max samples must be a float between 0 and 1, excluding 0')

        # calculate the class frequencies
        self.class_frequencies: List[Dict[Any, int]] = []
            # splits x Dict[class]=class_frequency/0
        self.class_weights: Optional[List[Dict[Any, Union[int, float]]]] = None
            # splits x Dict[class]=weight to use in the RandomForest
            # for any sample of that class
            # initialized later on by a coord method
        self.classes = np.unique(y[0])
        for _y in y:
            # ensure each split has the same class frequencies
            classes_y = np.unique(_y)
            if not np.array_equal(classes_y, self.classes):
                self.logging_class.warning('Classes differ between splits')
            frequency_dict = dict()
            for class_i in self.classes:
                if self.weight_classes_bool:
                    frequency_dict[class_i] = np.sum(_y == class_i)
                else:
                    frequency_dict[class_i] = 0
            self.class_frequencies.append(frequency_dict)

        # Store data
        self.X: List[np.ndarray] = X # (split x num_samples x num_features)
        self.y: List[np.ndarray] = y # (split x num_samples)
        self.X_test: List[np.ndarray] = X_test # (split x num_samples x num_features)
        self.y_test: List[np.ndarray] = y_test # (split x num_samples)
        self.classes: np.ndarray = np.unique(y[0]) # (num_classes)
        self.depth: int = 0

        # variables that are set by methods later on
        self.__split_points_fixed_width: Optional[np.ndarray] = None
            # splits x feature x n_bins - 1
        self.split_points: Optional[np.ndarray] = None
            # splits x n_features x n_bins - 1
        self.global_means: Optional[List[np.ndarray]] = None
            # splits x num_features (means per feature)
        self.global_counts: Optional[List[np.ndarray]] = None
            # splits x num_features (sample counts globally)
        self.global_stddevs: Optional[List[np.ndarray]] = None
            # splits x num_features (stddevs per feature)
        self.__X_hist_transposed: Optional[List[np.ndarray]] = None
            # splits x num_features x num_samples (histogram bin indexes)
            # values are the bin indexes the sample belongs to for each feature
        self.X_hist: Optional[List[np.ndarray]] = None
            # splits x num_samples x num_features (histogram bin indexes)
            # values are the bin indexes the sample belongs to for each feature
        self.global_classes: Optional[np.ndarray] = None
            # (num_classes) the classes that are available globally
        self.RF_feat_idcs: Optional[np.ndarray] = None
            # n_estimators x max_features (feature indices)
            # values are the feature indices for each estimator (decision tree)
            # in the random forest to use
        self.rf_models: Optional[List[RandomForest]] = None
            # n_estimators RandomForest models

    def get_fixed_width_binning_bounds(self) -> List[np.ndarray]:
        """
        Calculates the binning bounds for the fixed width binning.
        To preserve privacy, the boundaries are the mean of the top/bottom 5%
        of the data.

        Returns:
            List[np.ndarray] (splits x 2 x n_features): The binning bounds.
                First entry in the second dimension is the array of lower bounds,
                second entry is the array of upper bounds.
        """
        result = []
        for split_data in self.X:
            split_data = split_data[:, self.fixed_width_idcs]
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
                # fun fact: the documentation of np.sort does not contain the
                # word ascending nor descending
            min_array = np.mean(split_data_sorted[:quantile5_end], axis=0)
            max_array = np.mean(split_data_sorted[quantile95_start:], axis=0)
                # column-wise mean -> per feature mean as one vector
            result.append(np.array([min_array, max_array]))
        return result

    def set_fixed_witdh_bins(self, global_split_points: List[np.ndarray]) -> None:
        """
        Based on the given split points for the fixed width binning, assigns the
        data points to the bins and stores the bin indexes in self.__X_hist_transposed.
        Quantile binning then finalizes and creates self.X_hist.

        Args: global_split_points: List[List[List[float]]] (splits x n_features x n_bins - 1):
            The global split points for the fixed width binning. Open interval
            without a min/max value.

        Returns:
            None, just sets self.X_hist of the self.fixed_width_idcs
        """
        if self.__X_hist_transposed is None:
            self.__X_hist_transposed = []
        for split_idx, X in enumerate(self.X):
            split_points = global_split_points[split_idx]
            split_points = np.array(split_points)
            X_T = np.transpose(X[:, self.fixed_width_idcs])
            X_hist_fixed_witdh = \
                np.array([np.digitize(X_T[feature_idx], split_points[feature_idx]) \
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
            if len(self.__X_hist_transposed) <= split_idx:
                # for this split there is no entry yet
                self.__X_hist_transposed.append(np.zeros((self.num_features, X.shape[0])))
                    # we first create the full X_hist matrix
                    # and now only overwrite the fixed_width_idcs
                self.__X_hist_transposed[split_idx][self.fixed_width_idcs, :] = X_hist_fixed_witdh
            else:
                # for this split there is already an entry
                # we want to overwrite only the fixed_width_idcs
                self.__X_hist_transposed[split_idx][self.fixed_width_idcs, :] = X_hist_fixed_witdh

    def set_quantilie_bins(self, global_stddevs:List[List[float]]) -> None:
        """
        Based on the given global standard deviations for the quantile features,
        and previously calculated global_means and global_counts, assigns the
        bins to the data points desginated for quantile binning.
        Furthermore, merges the fixed width and quantile binning and created
        X_hist.
        Quantile binning is done by z-score normalizing the data and then
        assigning the data points to the bins based on the z-score and their
        corresponding percentiles.

        Args:
            global_stddevs: List[List[float]] (splits x num_features):
                The global standard deviation for each feature.
        """
        if not self.__X_hist_transposed:
            raise ValueError('Fixed width binning must be set before quantile binning')
        if not self.global_means:
            raise ValueError('Global means must be set before quantile binning')
        if not self.global_counts:
            raise ValueError('Global counts must be set before quantile binning')
        if not self.__split_points_fixed_width:
            raise ValueError('Fixed width binning must be set before quantile binning')
        X_normalized = []
        for split_idx, split_data in enumerate(self.X):
            # we z-score normalize the data
            # formula: (x_i - mean) / stddev
            dividend = (split_data[:, self.quantile_idcs] - self.global_means[split_idx])
            # format is samples x features_quantile
            divisor = np.array(global_stddevs[split_idx])
            normalized = np.divide(dividend, divisor,
                                   out=np.zeros_like(dividend),
                                   where=divisor != 0)
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

        percentiles = np.linspace(1 / self.n_bins, 1 - 1 / self.n_bins, self.n_bins - 1)
            # Finds the relevant percentiles for quantile binning
            # e.g. for n_bins = 2, we would want the 50% percentile
            # the min and max value of the split points are always 0 and 1
            # we can't use the min (0) and max (1) value as they are -inf and inf
            # this is why we use n_bins -1 and start at 1/n_bins, end at
            # 1 - 1/n_bins
        split_points_quantile = [norm.ppf(p) for p in percentiles]
            # reminder: the data for quantile binning is z-score normalized
            # -> we assume normally distributed data for the quantile binning
            # We now go from the percentile, e.g. from 0 to 25% of all values,
            # to the value x at which all values <= x together make up 25% of all values
            # norm.ppf does this for us
            # 1d array of length n_bins - 1

        for split_idx, split_data in enumerate(X_normalized):
            X_T = np.transpose(split_data[split_idx, :])
                # format is features_quantile x samples
                # X_normalized is already just the quantile features
            # Assign data points to bins
            X_hist_transposed = np.array([np.digitize(X_T[i], split_points_quantile) \
                                    for i in range(X_T.shape[0])])
                # Reminder: the split points are the interval ]min, 1, ..., max[
                # we do not need to supply the min and max value, as according
                # to the documentation of np.digitize:
                # If values in x are beyond the bounds of split_points,
                # 0 or len(split_points) is returned as appropriate.
                # The interval ]min, 1, ..., max[ has n_bins - 1 split points
                # therefore we end up with bin indexes 0, ..., n_bins - 1
                # so with exactly n_bins bins
                # the value 0 is therefore membership of that sample for that
                # feature of the bin 0
                # which is perfect for our purposes
                # format is features_quantile x samples
            if len(self.__X_hist_transposed) <= split_idx or \
                self.__X_hist_transposed[split_idx].shape[0] != self.num_features or \
                self.__X_hist_transposed[split_idx].shape[1] != self.X[split_idx].shape[0]:
                raise ValueError('Fixed width binning must be set before quantile binning')

            self.__X_hist_transposed[split_idx][self.quantile_idcs, :] = X_hist_transposed
                # we overwrite the quantile features in the X_hist matrix

        # create X_hist
        self.X_hist = []
        for split_idx, split_data in enumerate(self.__X_hist_transposed):
            self.X_hist.append(np.transpose(split_data))
                # just transpose from features x samples to samples x features

        # Also save merged split points
        self.split_points = np.zeros((len(self.X), self.num_features, self.n_bins - 1))
            # splits x features x n_bins - 1
        for split_idx, _ in enumerate(self.X_hist):
            self.split_points[split_idx, self.fixed_width_idcs, :] = \
                self.__split_points_fixed_width[split_idx]
            self.split_points[split_idx, self.quantile_idcs, :] = \
                np.tile(split_points_quantile, len(self.quantile_idcs))

    def get_quantile_binning_aggregation(self) -> List[np.ndarray]:
        """
        Calculates the sum of all quantile_features as well as the sample count.

        Returns:
            List[np.ndarray] (split x n_quantile_features x 2):
            The first entry in the last dimension is the array of sample counts,
            the second entry is the array of column-wise sums.
        """
        result = []
        for split_data in self.X:
            split_data = split_data[:, self.quantile_idcs]
            num_features_quantile = split_data.shape[1]
            local_matrix = np.zeros((num_features_quantile, 2))
            # if num_features_quantile = 0, these will have no effect
            # as there is no row to fill
            local_matrix[:, 0] = split_data.shape[0] # num_rows = num_samples
                # MISSING_VALUES_SUPPORT: don't use shape, get count without missing values
            local_matrix[:, 1] = np.sum(split_data, axis=0)
                # column-wise sum -> per feature sum as one vector
            result.append(local_matrix)
        return result

    def calc_local_stddev(self,
                          global_means: List[List[float]],
                          global_counts: List[List[int]]) -> List[np.ndarray]:
        """
        Given the global_means and global_counts, calculates the local standard deviation
        per feature for the quantile features and returns them.

        Args:
            global_means: List[List[float]] (splits x n_quantile_features):
                The global mean for each feature.
            global_counts: List[List[int]] (splits x n_quantile_features):
                The global sample count for each feature.
        """
        self.global_means = [np.array(d) for d in global_means]
        self.global_counts = [np.array(d) for d in global_counts]
            # MISSING_VALUES_SUPPORT: in this case the sample counts might differ and the following
            # check is not valid
        # ensure all global_counts are the same value per split
        for split_global_counts in self.global_counts:
            if not np.all(split_global_counts == split_global_counts[0]):
                raise ValueError('Global counts differ between features')
        stddevs = []
            # format is splits x num_features_quantile
            # each entry is the standard deviation for the corresponding feature
            # and split
            # Caveat: we don't send exactly the standard deviation but the sum of
            # (x_i - mean)^2
            # The global client can then devide by the number of samples
            # we could divide in each client, but we can simply only do the division
            # once in the end -> less calculations
        for split_idx, split_data in enumerate(self.X):
            X_quantile = split_data[:, self.quantile_idcs]
                # samples x features_quantile
                # Extract only the features that are used for quantile binning
                # from the raw data
            # formula stddev is sqrt(sum(x_i - mean)^2 / num_samples)
            local_stddev = np.sum(((X_quantile - self.global_means[split_idx]) ** 2) /
                                  self.global_counts[split_idx], axis=0)
                # X_quantile is samples x features_quantile, means[split_idx]
                # and sample_counts[split_idx] are vectors of shape features_quantile
                # broadcasting applies means and sample_counts row-wise (sample axis)
                # np.sum is used to collapse the samples axis
                # final shape becomes features_quantile vector
            stddevs.append(local_stddev)
                # stddevs gets shape splits x features_quantile with each
                # entry being the local stddev for the corresponding feature

        # we don't save the stddevs as we don't need them later on, we only
        # need the global stddevs later on
        return stddevs

    def get_class_frequencies(self) -> List[Dict[Any, int]]:
        """
        Receives a dictionary specifying a value per class in y.

        In case of weighted classes, the value is the frequency of the class.
        Otherwise, the value is 0.

        Returns:
            Dict[int, int]: The class frequencies.
        """
        if len(self.class_frequencies) == 0:
            raise ValueError('Class frequencies have not been calculated yet')
        return self.class_frequencies

    def get_class_weights(self) -> Optional[List[Dict[Any, Union[int, float]]]]:
        """
        Returns the class weights for each split.
        If class weights should be set but are not, raises a ValueError.
        """
        if self.weight_classes_bool and self.class_weights is None:
            raise ValueError('Class weights have not been set yet but should be set')
        return self.class_weights

    def get_available_classes(self) -> np.ndarray:
        """
        Returns the available classes.
        """
        return self.classes

    def set_RF_feat_idcs(self, RF_feat_idcs: List[List[int]]) -> None:
        """
        Sets the feature indices for the random forest.

        Args:
            RF_feat_idcs: List[List[int]] (n_estimators x max_features):
                The feature indices for the random forest.
        """
        self.RF_feat_idcs = np.array(RF_feat_idcs)

    def set_available_classes(self, classes: List[Any]) -> None:
        """
        Sets the available classes. Throws a warning if the current classes
        differ from the new classes.
        """
        if set(classes) != set(self.classes):
            self.logging_class.warning('This client has different classes " +\
                                       "than the union of classes')
        self.global_classes = np.array(classes)
        self.global_classes.sort()

    def set_class_weights(self, weights: Optional[List[Dict[Any, Union[int, float]]]]) -> None:
        """
        Sets the class weights for each split.

        Args:
            weights: List[Dict[int, int]] (splits x dict[class]=weight):
                The class weights for each split.
        """
        if self.weight_classes_bool and not weights:
            raise ValueError('Trying to set empty class weights also class weighting is enabled')
        self.class_weights = weights

    def init_forest(self) -> None:
        """
        Initializes the random forest model.
        Needs to be called after all other initialization methods are called.
        """
        self.logging_class.info('Initialize forest...')

        if self.prediction_mode not in ['classification', 'regression']:
            raise AttributeError('Only classification and regression are valid modes.')
        if not self.RF_feat_idcs:
            raise AttributeError('Feature indices must be set before initializing the forest.')
        if not self.X_hist or not self.split_points:
            raise AttributeError('Binning information must be set before initializing the forest.')
        if not self.global_means or not self.global_stddevs or not self.global_counts:
            raise AttributeError('Global statistics must be set before initializing the forest.')
        if not self.global_classes:
            raise AttributeError('Global classes must be set before initializing the forest.')


        self.rf_models = []
            # per split a RandomForest model
        for split_idx, _ in enumerate(self.X_hist):
            rf_model: RandomForest = \
                RandomForest(n_estimators=self.n_estimators,
                             global_classes=self.global_classes,
                             random_state=self.random_state,
                             max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             bootstrap=self.bootstrap,
                             feat_idcs=self.RF_feat_idcs,
                             max_samples=self.max_samples,
                             quantile=self.quantile_idcs,
                             global_mean=self.global_means[split_idx],
                             global_stddev=self.global_stddevs[split_idx],
                             split_points=self.split_points[split_idx],
                             prediction_mode=self.prediction_mode,
                             oob=self.oob,
                             class_weights=self.class_weights[split_idx] \
                                if self.class_weights else None,
                            X_hist=self.X_hist[split_idx],
                            y=self.y[split_idx],
                            num_bins=self.n_bins)
            self.rf_models.append(rf_model)

    def get_current_level_splitscores(self) -> Tuple[List[List[List[List[List[float]]]]],
                                                    List[List[List[List[int]]]],
                                                    List[List[Optional[List[Optional[Any]]]]]]:
        """
        Returns the split scores of all RandomForest models. Only calculates the one of the current
        level of each tree. Throws an error if the current level is already set, except if the
        current level is already just leave nodes.

        Returns:
            scores: List[List[List[List[List[float]]]]] (splits x n_estimators x n_nodes x
                n_features x n_bins):
                The scores of the trees for the current level of all possible splits by bins
            counts: List[List[List[List[int]]]] (splits x n_estimators x n_nodes x n_features):
                The amount of samples in the corresponding node
            only_class: List[Optional[List[List[Optional[Any]]]]] (splits x n_estimators x n_nodes):
                If the corresponding node only has samples of one target class, then this
                contains this target class. Otherwise, it is None.
                The whole array of nodes might be None if the tree is finished.
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet')
        if not self.X_hist:
            raise ValueError('Binning information must be set before calculating split scores')
        scores = []
        counts = []
        only_classes = []
        for split_idx, rf_model in enumerate(self.rf_models):
            score, count, only_class = rf_model.get_split_scores()
            scores.append(score)
            counts.append(count)
            only_classes.append(only_class)
        return scores, counts, only_classes

    def update_current_depth_nodes(self,
                                global_best_split: List[List[Optional[List[Tuple[int, int, float]]]]],
                                global_leaf_info: List[List[Optional[List[int]]]]) -> None:
        """
        Based on the globally calculated best splits, sets the current nodes of the trees.
        Sets their threshold, score and feature index.
        Also checks if the node is locally a leaf node.
        Returns the leaf information.
        The leaf information needs to be aggregated by the coordinator to finally set the
        current depth nodes.

        Args:
            global_best_split: List[List[Optional[List[Tuple[int, int, float]]]]]
                (splits x n_estimators x n_nodes):
                Per split, tree and node contains the best split
                The final tuple is (feature_idx, bin_idx, score)
            leaf_info: List[List[Optional[List[int]]]] (splits x n_estimators x num_leaf_nodes):
                The leaf information of the nodes. Per split, per tree, contains the indexes in
                current_depth_nodes that are leaf nodes. If the tree is finished, None is returned
                for this tree.

        Returns:
            None, just sets the leaf status of the current_depth_nodes, creates children and
            sets them as the new current_depth_nodes
            Also checks if the tree is finished and sets the tree.finished attribute
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet')
        if not self.X_hist or not self.split_points:
            raise ValueError('Binning information must be set before setting nodes')
        if not self.global_classes:
            raise ValueError('Global classes must be set before setting nodes')
        if not self.global_means or not self.global_stddevs or not self.global_counts:
            raise ValueError('Global statistics must be set before setting nodes')
        for split_idx, rf_model in enumerate(self.rf_models):
            rf_model.set_currently_unset_nodes(global_best_split[split_idx], global_leaf_info[split_idx])


    def check_finished(self) -> bool:
        """
        Checks whether all models of all splits are finished

        Returns:
            bool: True if all models are finished, False otherwise
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet. Cannot check if its done')
        for rf_model in self.rf_models:
            finished = rf_model.check_finished()
            if not finished:
                return False
        return True

    def get_leaf_node_samples(self) -> List[List[List[List[int]]]]:
        """
        Receives the local potential predicted classes per leaf. The order of the leaf nodes
        is given by performing DFS on the tree nodes until leaf nodes are reached.

        Returns:
            List[List[int]]: Per split, per tree, per leaf node contains the amount of
                samples per class in the leaf node. Per node, the list indexes are the same
                then the class indexes.
        """
        if not self.check_finished() or not self.rf_models:
            raise ValueError("Trying to set leaf node values before finishing the tree")
        leaf_samples: List[List[List[List[int]]]] = []
        for model in self.rf_models:
            leaf_samples.append(model.get_leaf_node_samples())
        return leaf_samples

    def set_final_leaf_nodes(self, global_leaf_samples: List[List[List[int]]]) -> None:
        """
        Given the global classes predicted by the leaf nodes, sets the leaf nodes

        Args:
            global_leaf_samples: List[List[List[int]]] (splits x n_estimators x num_leaf_nodes):
                The class_idx for each leaf node in each tree in each split
        """
        if not self.rf_models:
            raise ValueError('The forest has not been trained yet')
        for split_idx, rf_model in enumerate(self.rf_models):
            if not rf_model.check_finished():
                raise ValueError('The model is not finished yet')
            rf_model.set_final_leaf_nodes(global_leaf_samples[split_idx])


    def calc_oob(self) -> List[List[Tuple[int, int]]]:
        """
        Calculates the oob error rate for the random forest.

        Returns:
        (split x n_estimators): The oob error rate for each estimator in each split.
        Per tree, the tuple contains the number of incorrect predictions and the number of oob samples.
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet')
        if not self.check_finished():
            raise ValueError('The forest is not finished yet, cannot calculate oob')
        if not self.oob:
            raise ValueError('OOB settings must be set to calculate oob')
        oob = []
        for rf_model in self.rf_models:
            oob.append(rf_model.calc_oob())
        return oob

    def write_rf_model_class(self) -> None:
        """
        Writes the RandomForest.py class to the output folder.
        """
        with open(f'{self.outputfolder}/RandomForest.py', 'w') as f:
            f.write(inspect.getsource(RandomForest))

    def evaluate_local(self) -> None:
        """
        Evaluates the random forest model on the test data. Saves the wanted output, either
        just the model, just the predictions or both.

        Args:
            X_test: np.ndarray (num_splis x num_samples x num_features): The test data.
            y_test: np.ndarray (num_splits x num_samples): The test labels.

        Returns:
        Tuple[List[np.number], List[np.number], List[int]]
            Each tuple contains a list where the index is the split index. The lists are:
                acc: The accuracy of the model.
                mcc: The Matthews correlation coefficient of the model.
                counts: List[int]: The sample counts per split.
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet')
        if not self.check_finished():
            raise ValueError('The forest is not finished yet, cannot predict yet')
        if not self.global_classes:
            raise ValueError('Global classes must be set before evaluating')
        if not self.global_means or not self.global_stddevs or not self.global_counts:
            raise ValueError('Global statistics must be set before evaluating')
        X_test = self.X_test
        y_test = self.y_test
        if len(X_test) != len(self.rf_models) != len(y_test):
            raise ValueError('The number of splits differs between test data and models')

        for split_idx, rf_model in enumerate(self.rf_models):
            X_test_split = X_test[split_idx]
            y_test_split = y_test[split_idx]

            # predict the test data
            predictions = rf_model.predict(X=X_test_split)

            # save the information
            self.write_output(split_idx=split_idx,
                              y_test=y_test_split,
                              predictions=predictions)


    def write_output(self,
                     split_idx: int,
                     y_test: np.ndarray,
                     predictions: np.ndarray) -> None:
        """
        Writes the output of the predictions to a files. Also saves the model if wanted.
        Output depends on self.output_format.

        Args:
            split_idx: int: The index of the split.
            y_test: np.ndarray: The true labels.
            predictions: np.ndarray: The predicted labels.
        """
        # save the model if wanted
        basepath = self.outputfolder
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet, but trying to save the model')
        if 'model' in self.output_mode:
            modelpath = f'{basepath}/model.pkl'
            if self.split_mode == 'directory':
                modelpath = f'{basepath}/model_split_{split_idx}.pkl'
            self.rf_models[split_idx].save_model(modelpath)

        # save the predictions
        if 'pred' in self.output_mode:
            y_test_series = pd.Series(y_test)
            predictions_series = pd.Series(predictions)
            if self.split_mode == 'directory':
                y_test_series.to_csv(f'{basepath}/y_test_split_{split_idx}.csv', index=False)
                predictions_series.to_csv(f'{basepath}/predictions_split_{split_idx}.csv', index=False)
            else:
                y_test_series.to_csv(f'{basepath}/y_test.csv', index=False)
                predictions_series.to_csv(f'{basepath}/predictions.csv', index=False)


    def coord_ensure_config_alignment(self,
                                      feature_names: List[List[str]],
                                      quantile_idcs: List[List[int]],
                                      fixed_width_idcs: List[List[int]],
                                      oobs: List[bool]) -> None:
        """
        Receives the feature names, quantile indices and fixed width indices
        of all different clients and ensures they are the same.
        Should only be called by the aggregator.
        """
        if not all([feature_names[0] == feature_name for feature_name in feature_names]):
            print("ERROR: Feature names do not match between clients")
            # print union vs intersection of feature names
            print(f"INTERSECTION of features: {set.intersection(*[set(f) for f in feature_names])}")
            print(f"UNION of features: {set.union(*[set(f) for f in feature_names])}")
            raise ValueError('Feature names do not match between clients')
        if not all([quantile_idcs[0] == quantile_idx for quantile_idx in quantile_idcs]):
            print("ERROR: Quantile indices do not match between clients")
            print("INTERSECTION of quantile indices: " +\
                  f"{set.intersection(*[set(q) for q in quantile_idcs])}")
            print(f"UNION of quantile indices: {set.union(*[set(q) for q in quantile_idcs])}")
            raise ValueError('Quantile indices do not match between clients')
        if not all([fixed_width_idcs[0] == fixed_width_idx \
                    for fixed_width_idx in fixed_width_idcs]):
            print("ERROR: Fixed width indices do not match between clients")
            print("INTERSECTION of fixed width indices: " +\
                  f"{set.intersection(*[set(f) for f in fixed_width_idcs])}")
            print(f"UNION of fixed width indices: {set.union(*[set(f) for f in fixed_width_idcs])}")
            raise ValueError('Fixed width indices do not match between clients')
        if not all([oobs[0] == oob for oob in oobs]):
            raise ValueError('OOB settings do not match between clients')

    def coord_ensure_same_num_splits(self,
                                     fixed_width_binning_bounds: List[List[Any]],
                                     quantile_binning_aggregation: List[List[Any]]) -> None:
        """
        Ensures that the number of splits is the same for all clients.

        Args:
            fixed_width_binning_bounds: List[np.ndarray] (clients x splits x 2 x num_features):
                The binning bounds for fixed width binning.
            quantile_binning_aggregation: List[np.ndarray] (clients x splits x num_features x 2):
                The sums and sample counts for the quantile features.

        Raises:
            ValueError: If the number of splits differ between clients or if the number of clients
                differ between fixed width and quantile binning.
        """
        if len(fixed_width_binning_bounds) != len(quantile_binning_aggregation):
            raise ValueError('Number of clients differ between fixed width and quantile binning')
        splits = {len(d) for d in fixed_width_binning_bounds}
        if len(splits) != 1:
            raise ValueError('The number of splits differ between clients')

    def coord_calculate_global_fixed_width_binning_splitpoints(self,
                                                        bounds: List[List[List[List[Any]]]]) \
                                                        -> np.ndarray:
        """
        Calculates the splitpoints for the fixed width binning.
        Should only be called by the aggregator.
        Stores the splitpoints in self.__split_points_fixed_width as np.ndarray
        and returns them as list.

        Args:
            bounds: List[np.ndarray] (clients x splits x 2 x num_features): The binning bounds.
                First entry in the last dimension is the array of lower bounds,
                second entry is the array of upper bounds.

        Returns:
            List[np.ndarray] (splits x n_features x n_bins - 1): The splitpoints.
            for n_bins n_bins + 1 split points exist, we only need n_bins - 1
            as we don't need the min and max split points (basically open intervals)
        """
        splitpoints_fixed_witdh = []
            # split x feature x n_bins - 1
            # for n_bins n_bins + 1 split points exist, we only need n_bins - 1
            # as we don't need the min and max split points (basically open intervals)
        for split_idx, split_data in enumerate(self.X):
            split_data = split_data[:, self.fixed_width_idcs]
            min_max_values = [bounds[i][split_idx] for i in range(len(bounds))]
                # extract the min/max values for this specific split
                # dimension is therefore clients x 2 x num_features
            min_max_values = np.array(min_max_values)
                # np.array of shape clients x 2 x num_features
            min_values_all = min_max_values[:, 0, :]
            max_values_all = min_max_values[:, 1, :]
                # shape clients x num_features
            min_values = np.min(min_values_all, axis=0)
            max_values = np.max(max_values_all, axis=0)
                # reduce via min/max from clients x num_features to num_features,
                # finding the min over all clients
            splitpoints_per_split = [np.linspace(float(min_values[feature_idx]), \
                                                 float(max_values[feature_idx]), \
                self.n_bins + 1) for feature_idx in range(len(min_values))]
                # n_bins + 1 as for n_bins we need n_bins + 1 split points
                # 2 bins -> min, 1, max needed as split points
                # np.linspace includes the start and stop value
            splitpoints_fixed_witdh.append([split_points[1:-1] \
                                            for split_points in splitpoints_per_split])
                # We neither need min nor max due to how np.digitize works

        self.__split_points_fixed_width = np.array(splitpoints_fixed_witdh)
        return self.__split_points_fixed_width

    def coord_calculate_global_mean_count(self,
                                          data: List[List[List[List[Any]]]]) \
            -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Given the sums and sample counts for the quantile features per client,
        calculates the global mean and the global sample count.

        Args:
            data: List[np.ndarray] (clients x splits x num_features x 2):
                The first entry in the last dimension is the array of column wise sample counts,
                the second entry is the array of column-wise sums.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]:
                broadcast_means: List[np.ndarray] (splits x n_quantile_features):
                    The global mean for each feature.
                sample_counts: List[np.ndarray] (splits x n_quantile_features):
                    The global sample count for each feature.
        """
        means = []
            # splits x features, each entry being the mean for the corresponding feature
            # globally
        sample_counts = []
            # splits x features, each entry being the number of samples
            # for the corresponding feature globally

        # calculate the global mean
        for split_idx, _ in enumerate(self.X):
            try:
                mean_count_arr = np.array([d[split_idx] for d in data])
                # format is clients x n_quantile_features x 2, we removed the split
                # axis due to the split loop
            except IndexError as e:
                raise ValueError('The number of splits differ between clients') from e
            global_matrix = np.sum(mean_count_arr, axis=0)
                # we sum over the clients axis, new format is num_features x 2
            accumulated_sample_count = global_matrix[:, 0] # vector of shape n_quantile_features
            accumulated_sum = global_matrix[:, 1] # vector of shape n_quantile_features
            mean = accumulated_sum / accumulated_sample_count # vector of shape n_quantile_features
            means.append(mean)
            sample_counts.append(accumulated_sample_count)
        self.global_means = means
        self.global_counts = sample_counts
        return means, sample_counts

    def coord_aggregate_stddevs(self, stddevs: List[List[List[float]]]) -> List[np.ndarray]:
        """
        Aggregates the standard deviations from all clients to a global standard deviation.

        Args:
            stddevs: List[np.ndarray] (clients x splits x n_quantile_features):
                The standard deviation for each feature per client.

        Returns:
            List[np.ndarray] (splits x n_quantile_features): The global standard deviation.
        """
        global_stddevs = []
        for split_idx, _ in enumerate(self.X):
            try:
                data = np.array([d[split_idx] for d in stddevs])
                # format is clients x features_quantile
            except IndexError as e:
                raise ValueError('The number of splits differ between clients') from e
            global_stddev_split = np.sum(data, axis=0)
                # collapse the clients axis, clients x features_quantile -> features_quantile
            global_stddevs.append(global_stddev_split)
                # global_stddevs is splits x features_quantile
        self.global_stddevs = global_stddevs
        return global_stddevs

    def coord_set_class_weights(self,
                                class_frequencies_clients: List[List[Dict[int, int]]]) -> None:
        """
        Calculates class weights for each split based on the frequencies of the classes
        in the data.
        The weighting is heavily inspired (aka taken) from the sklearn RandomForestClassifier
        class_weight = "balanced" mode.
        According to their documentation:
        The “balanced” mode uses the values of y to automatically adjust weights inversely
        proportional to class frequencies in the input data as
        n_samples / (n_classes * np.bincount(y))

        Args:
            class_frequencies_clients: List[List[Dict[int, int]]]
            (clients x splits x dict[class]=frequency):
                The class frequencies from all clients as a list.

        Returns:
            None, sets the weights in self.class_weights
        """

        # update classes to have all global classes
        classes = set()
        for class_frequency_list in class_frequencies_clients:
            for class_frequency in class_frequency_list:
                prev_len = len(classes)
                classes.update(class_frequency.keys())
                if len(classes) != prev_len and prev_len != 0:
                    self.logging_class.warning('Classes differ between clients')

        classes = np.array(list(classes))
        n_classes = len(classes)
        if set(classes) != set(self.classes):
            self.logging_class.warning('This client has different classes " +\
                                       "than the union of classes')
        self.classes = classes

        # set weights if necessary
        if self.weight_classes_bool:
            weights: List[Dict[Any, Union[int, float]]] = []
                # list index is the split index, contains
                # dictionaries with class_i as key and weight as value
            for split_idx, _ in enumerate(self.X):
                split_weights: Dict[Any, float] = {}
                split_total_samples = 0
                # get the pure frequency counts per split per class
                for client_class_frequencies in class_frequencies_clients:
                    for class_i, frequency in client_class_frequencies[split_idx].items():
                        if class_i not in split_weights:
                            split_weights[class_i] = 0
                        split_weights[class_i] += frequency
                        split_total_samples += frequency
                # calculate the weight from the frequency plus total samples
                for class_i, frequency in split_weights.items():
                    if frequency != 0:
                        split_weights[class_i] = split_total_samples / (n_classes * frequency)
                        # we implement balanced class weights from the
                        # sklearn RandomForestClassifier
                        # According to their documentation:
                        # The “balanced” mode uses the values of y to automatically adjust weights
                        # inversely proportional to class frequencies in the input data as
                        # n_samples / (n_classes * np.bincount(y))
                    else:
                        split_weights[class_i] = 0
                weights.append(split_weights)

            self.class_weights = weights

    def coord_get_RF_feat_idcs(self) -> np.ndarray:
        """
        Based on self.max_features, self.n_features and self.n_estimators,
        chooses for each estimator (decision tree) in the random forest
        a random set of feature indices. The set size is determined by
        self.max_features.

        Returns:
            np.ndarray (n_estimators x max_features): The feature indices.
        """
        RF_feat_idcs = np.random.choice(self.n_features,
                                        size=(self.n_estimators, self.max_features),
                                        replace=False)
        self.RF_feat_idcs = RF_feat_idcs
        return RF_feat_idcs

    def coord_aggregate_split_scores(self,
                                     client_split_scores: List[List[List[Optional[List[List[float]]]]]],
                                     sample_count_per_client: List[List[List[Optional[List[List[int]]]]]],
                                     only_class_per_client: List[List[List[Optional[List[Optional[Any]]]]]]) \
            -> Tuple[List[List[Optional[List[Tuple[int, int, float]]]]],
                     List[List[int]]]:
        """
        Aggregates the split scores from all clients to a global split score.
        The split score per client was calculated by self.get_current_level_splitscores.

        Args:
            client_split_scores: List[List[List[List[List[float]]]]]
            (clients x splits x n_estimators x n_nodes x
                n_features x n_bins):
                The scores of the trees for the current level of all possible splits by bins
            sample_count_per_client: List[List[List[List[List[int]]]]]
                (clients x splits x n_estimators x n_nodes x n_features):
                The sample counts per client. Same order of clients as client_split_scores.
            only_class_per_client: List[List[Optional[List[List[Optional[Any]]]]]]
                (clients x splits x n_estimators x n_nodes):
                If the corresponding node only has samples of one target class, then this
                contains this target class. Otherwise, it is None. Might be None for a whole tree
                if the tree is finished.

        Returns:
            Tuple[global_split_scores, leaf_status]:
            global_split_scores: List[List[List[Tuple[int, int, float]]]]
                (splits x n_estimators x n_nodes x (feature_idx, bin_idx, score)):
                For each node, the feature index, bin index and score of the best split.
            leaf_status: List[List[int]]
                (splits x n_estimators x n_leaf_nodes):
                Contains the indexes in current_depth_nodes that have been determined to be
                leaf nodes.
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet')
        if not self.X_hist:
            raise ValueError('Binning information must be set before calculating split scores')

        global_split_scores = []
            # split x n_estimators x n_nodes x (feature_idx, bin_idx, score)
        leaf_status_per_split = []
        for split_idx, _ in enumerate(self.X_hist):
            tree_scores = []
            leaf_status_per_tree = []
            for tree_idx, tree in enumerate(self.rf_models[split_idx].iterate_trees()):
                leaf_status_per_node = []
                node_scores = []
                if tree.finished:
                    # ensure that no client sent any data for this tree
                    if any([specific_client_split_score[split_idx][tree_idx] is not None \
                            for specific_client_split_score in client_split_scores]):
                        raise ValueError(f'Tree {tree_idx} is already finished but clients sent data')
                    tree_scores.append(None)
                    continue
                # ensure that all clients sent data for this tree
                # pylance doesn't understand taht we do this so later we use type: ignore
                if not all([specific_client_split_score[split_idx][tree_idx] is not None \
                            for specific_client_split_score in client_split_scores]):
                    raise ValueError(f'Not all clients sent data for the tree {tree_idx}')
                if not all([specific_client_sample_count[split_idx][tree_idx] is not None \
                            for specific_client_sample_count in sample_count_per_client]):
                    raise ValueError(f'Not all clients sent sample counts for the tree {tree_idx}')
                if not all([specific_client_only_class[split_idx][tree_idx] is not None \
                            for specific_client_only_class in only_class_per_client]):
                    raise ValueError(f'Not all clients sent only class info for the tree {tree_idx}')
                for node_idx, _ in enumerate(tree.iterate_cur_depth_nodes()):
                    split_scores = [d[split_idx][tree_idx][node_idx] for d in client_split_scores] #type: ignore
                        # clients x features x n_bins
                    sample_counts = [c[split_idx][tree_idx][node_idx] for c  in sample_count_per_client] #type: ignore
                        # clients x features
                    total_counts = np.sum(sample_counts, axis=0)
                        # vector of length features of the total number of samples over all clients
                        # per feature of this specific node
                    ratios = np.divide(sample_counts, total_counts)
                        # this is the weight to use for the relevant client
                        # dividing clients x features by features -> ratio for each client and feature
                        # (clients x features dimensions)
                    # sum up the split scores considering the weights
                    sum_split_score = np.sum([split_scores[client_idx] * ratios[client_idx] for client_idx in range(len(split_scores))], axis=0)
                        # we multiply the split scores with the ratios of the relevant client
                        # then we can sum over the clients axis
                        # this results in the end in a features x n_bins matrix
                    assert sum_split_score.shape == (self.n_features, self.n_bins)
                    feature_idx, bin_idx = np.unravel_index(np.argmin(sum_split_score), sum_split_score.shape)
                        # we find the feature and bin index with the lowest score over all features and bins
                        # np.argmin returns the index of the flattened array, we need to unravel it
                        # back to the original (sum_split_score.shape) shape
                    min_score = np.min(sum_split_score)
                    node_scores.append((feature_idx, bin_idx, min_score))
                    # now we need to decide whether this node is a leaf node
                    # only one class
                    only_classes = [d[split_idx][tree_idx][node_idx] for d in only_class_per_client] #type: ignore
                        # list of length clients, containing either None or the only class
                    # get all unique classes
                    only_classes = {class_i for class_i in only_classes if class_i is not None}
                    if len(only_classes) == 1:
                        leaf_status_per_node.append(node_idx)
                        continue
                    # max_depth reached
                    # get the node
                    node = tree.get_cur_depth_node(node_idx)
                    if node.depth >= self.rf_models[split_idx].__max_depth - 1:
                        # max_depth is 1 indexed, depth is 0 indexed
                        leaf_status_per_node.append(node_idx)
                        continue
                    # min_samples_split reached
                    if total_counts[feature_idx] < self.rf_models[split_idx].__min_samples_split:
                        leaf_status_per_node.append(node_idx)
                        continue
                    # min_samples_leaf reached
                    # TODO: implement this at some point. Check this node as if it were a leaf
                    # if it has too little samples, the parent must be turned into a leaf and the
                    # parents other child needs to be removed!
                    # probably needs a new structure of what is returned here, e.g. not only the
                    # index of leaves but per index the information if the leaf_status is this
                    # node or the parent node
                    # min_impurity_decrease
                    # get the parents score
                    if node.parent:
                        # only works for non root nodes
                        parent_score = node.parent.score
                        if parent_score - min_score < self.rf_models[split_idx].min_impurity_decrease:
                            leaf_status_per_node.append(node_idx)
                            continue
                tree_scores.append(node_scores)
                leaf_status_per_tree.append(leaf_status_per_node)
            global_split_scores.append(tree_scores)
            leaf_status_per_split.append(leaf_status_per_tree)

        return global_split_scores, leaf_status_per_split

    def coord_aggregate_leaf_samples(self,
                                     gathered_leaf_samples: List[List[List[List[List[int]]]]]) \
                                    -> List[List[List[int]]]:
        """
        Finds for all leave nodes of the global model which global_class the leaf corresponds to.

        Args:
            gathered_leaf_samples: List[List[List[Dict[int, int]]]] (clients x splits x trees x
                leaf_nodes): Per leaf node, the amount of samples per class as a
                Dict[class] = frequency.

        Returns:
            List[List[List[int]]] (splits x trees x leaf_nodes): per leaf node the class_idx which
            globally has the highest frequency
        """
        gathered_leaf_samples_np = np.array(gathered_leaf_samples)
        # we need to collapse the clients axis
        gathered_leaf_samples_np = np.sum(gathered_leaf_samples_np, axis=0)
        # now we can find the class with the highest frequency
        # we need to get the index of the last dimension with the highest value in that dimension
        # this is the class index
        leaf_classes = np.argmax(gathered_leaf_samples_np, axis=-1)
        return leaf_classes

    def coord_aggregate_oob(self, gathered_oob_errors: List[List[List[Tuple[int, int]]]]) -> \
            List[List[float]]:
        """
        Calculates the oob error rate for the random forest globally and returns the weights of
        all trees

        Args:
            gathered_oob_errors: List[List[List[Tuple[int, int]]]]
                (clients x splits x trees x (error, num_oob_samples)):
                The amount of incorrect oob sample predictions for each estimator in each split.
                Per tree, the tuple contains the number of incorrect predictions and the number of
                oob samples.

        Returns:
            List[List[float]] (splits x trees):
                The weight of each tree in the random forest of each split.
        """
        #TODO: implement the aggregation of the oob errors
        # collapse the client axis
        gathered_oob_errors_np = np.array(gathered_oob_errors)
        gathered_oob_errors_np = np.sum(gathered_oob_errors_np, axis=0)
            # shape is splits x trees x 2
        # in the last dimension (error, num_oob_samples) we need to calculate the weight
        # the weight is simply the number of (1 - the incorrect predictions divided by the number of oob samples)
        error_rate = gathered_oob_errors_np[:, :, 0] / gathered_oob_errors_np[:, :, 1]
        weights = 1 - error_rate
        return weights

    def update_weights(self, weights: List[List[float]]) -> None:
        """
        Updates the RF_models weights with the given weights.

        Args:
            weights: List[List[float]] (splits x trees): The weights of the trees.

        Returns:
            None, updates the weights in the RF_models.
        """
        if not self.rf_models:
            raise ValueError('The forest has not been initialized yet')
        if len(weights) != len(self.rf_models):
            raise ValueError('The number of splits differ between the weights and the models')
        for split_idx, rf_model in enumerate(self.rf_models):
            rf_model.set_weights(weights[split_idx])


    def _read_config(self, config: dict):
        """
        Reads the configuration file and sets the parameters for the random forest.
        If a config dict is given instead, uses that to read the parameters.
        Check the repositories example config.yml/ the README
        for more information on the config file.
        https://github.com/FeatureCloud/fc-fed-random-forest
        """
        if not config:
            # try to read the config from file
            config_name = "config.yml"
            if not os.path.exists(f'{self.inputfolder}/{config_name}'):
                config_name = "config.yaml"
            config = bios.read(f'{self.inputfolder}/{config_name}')
        try:
            config = config['fc-rand-forest']

            config_input = config['input']
            self.train_filename: str = config_input['train']
            self.test_input_filename: str = config_input['test']

            config_output = config['output']
            self.pred_filename: str = config_output['pred']
            self.test_output_filename: str = config_output['test']

            config_format = config['format']
            self.sep: str = config_format.get('sep', ',')
            self.label_col: str = config_format['label_col']

            config_split = config['split']
            self.split_mode: str = config_split['mode']
            self.split_dir: str = config_split['dir']

            # Parameters RandomForest
            self.n_estimators: int = int(config.get('n_estimators', 100))
            self.criterion: str = config.get('criterion', 'gini')
            self.max_depth: int = int(config.get('max_depth', 10))
            self.min_samples_split: int = config.get('min_samples_split', 2)
            self.__max_features_raw: Union[str, float, int] = config.get('max_features', 'sqrt')
            self.bootstrap: bool = config.get('bootstrap', True)
            self.max_samples_raw: Union[None, float, int] = config.get('max_samples', None)
            self.random_state: int = int(config.get('random_state', 0))
            self.weight_classes_bool: bool = config.get('use_weighted_classes', False)
            self.__quantile_idcs_raw: List[int] = config.get('quantile', [])
            self.oob: bool = config.get('oob', False)

            self.prediction_mode: str = config['mode']
            if self.prediction_mode != 'classification':
                raise ValueError('Mode must be "classification"')
            # if self.prediction_mode == 'regression' and self.weight_classes_bool:
            #     raise ValueError('Weights are not supported for regression, " +\
            #                      "there are no classes to weight in regression')

            n_bins: Union[str, int] = config['n_bins']
            try:
                self.n_bins = int(n_bins)
            except ValueError as e:
                raise ValueError('Number of bins must be an integer') from e
            if self.n_bins < 2:
                raise ValueError('Number of bins must be at least 2')

            self.output_mode = config.get('output_mode', 'model')
            if self.output_mode not in ['model', 'pred', 'model+pred']:
                raise ValueError('Output mode must be either "model" or "pred" or "model+pred".')

        except KeyError as e:
            raise KeyError('Config file is missing key') from e
        except TypeError as e:
            raise TypeError('Config file does contain an invalid value') from e
        except Exception as e:
            raise ValueError('Unknown error while reading the config') from e

    def _read_files(self, train: str, test_input: str) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
        """
        Reads the train and test data files and returns the data as numpy arrays.
        Ensures that the feature names in the train and test data match, if not
        raises an ValueError.
        Args:
            train: str: Name of the train file. Should be the full path.
            test_input: str: Name of the test file. Should be the full path.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
                X: np.ndarray: Train data.
                y: np.ndarray: Train labels.
                X_test: np.ndarray: Test data.
                y_test: np.ndarray: Test labels.
                feature_names: List[str]: Feature names.
        Raises:
            ValueError: If the feature names in the train and test data do not match.
        """
        train_df = pd.read_csv(train, sep=self.sep)
        test = pd.read_csv(test_input, sep=self.sep)
        X_train = train_df.drop(self.label_col, axis=1)
        X_test = test.drop(self.label_col, axis=1)
        y_train = train_df.loc[:, self.label_col]
        y_test = test.loc[:, self.label_col]
        # check if we have any missing values and raise an error if yes
        if X_train.isnull().values.any() or y_train.isnull().any():
            raise ValueError("Missing values in train data.")
        if X_test.isnull().values.any() or y_test.isnull().any():
            raise ValueError("Missing values in test data.")
        feature_names = X_train.columns
        if not feature_names.equals(X_test.columns):
            print("Feature names in train and test data match.")
            print(f"Features train,test:\n{X_train.columns}\n{X_test.columns}")
            print("Features just in train data: " +\
                  f"{set(X_train.columns) - set(X_test.columns)}")
            print("Features just in test data: " +\
                  f"{set(X_test.columns) - set(X_train.columns)}")
            raise ValueError("Feature names in train and test data do not match.")
        # MISSING_VALUES_SUPPORT: remove columns without ANY values, also
        # remove them from feature_names
        X = convert_to_np(X_train)
        y = convert_to_np(y_train)
        X_test = convert_to_np(X_test)
        y_test = convert_to_np(y_test)

        return X, y, X_test, y_test, feature_names

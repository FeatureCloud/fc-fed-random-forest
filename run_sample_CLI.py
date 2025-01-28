"""
Small CLI to run the sample data through the random forest model
"""

import argparse
import os

import run_app_simulation as run_funcs

def main():
    """
    Main function to run the sample data through the random forest model
    """
    parser = argparse.ArgumentParser(description='Run the sample data through the random forest model')
    parser.add_argument('--sample_data','-d',
                        help='Either breast_cancer or iris',
                        required=True,
                        choices=['breast_cancer', 'iris'])
    parser.add_argument('--random_labels', '-r',
                        help='Whether to use the data with random labels',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--native',
                        help='If set to False (Default), FeatureCloud is used for the training',
                        action='store_true',
                        default=False,
                        required=False)
    args = parser.parse_args()

    # base variables
    basepath = os.path.join(os.path.dirname(__file__), 'sample_data')

    # datapaths
    if args.sample_data == 'breast_cancer':
        data_path = os.path.join(basepath, 'breast_cancer')
    elif args.sample_data == 'iris':
        data_path = os.path.join(basepath, 'iris')
    else:
        raise ValueError('Invalid sample data given')

    if args.random_labels:
        data_path = os.path.join(data_path, '3_clients_random_label')
    else:
        data_path = os.path.join(data_path, '3_clients')

    clientnames = ['client1', 'client2', 'client3']
    generic_dir = 'generic'

    # Using FeatureCloud
    if not args.native:
        run_funcs.run_simulation_featurecloud(data_path=data_path,
                                              clientnames=clientnames,
                                              generic_dir=generic_dir)
    else:
        clientpaths = [os.path.join(data_path, client) for client in clientnames]
        outputfolders = [os.path.join(data_path, 'output', client) for client in clientnames]
        generic_path = os.path.join(data_path, generic_dir)

        # ensure output folders exist
        if not os.path.exists(os.path.join(data_path, 'output')):
            os.makedirs(os.path.join(data_path, 'output'))

        for outputfolder in outputfolders:
            if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)

        run_funcs.run_simulation_native(clientpaths=clientpaths,
                                        outputfolders=outputfolders,
                                        generic_dir=generic_path)

if __name__ == '__main__':
    main()
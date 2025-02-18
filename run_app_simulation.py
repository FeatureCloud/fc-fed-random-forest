"""
Contains two functions to run a simulation of the federated random forest algorithm.
1. Via FeatureCloud's Testing module
2. Via a custom simulation (natively)
"""
from typing import List

# featurecloud imports
import time
import docker
import FeatureCloud.api.imp.controller.commands as controller
import FeatureCloud.api.imp.test.commands as test

# native imports
import threading
from src.helper.localfedlearningsimulator import LocalFedLearningSimulationWrapper
from logic import main

def run_simulation_featurecloud(data_path: str, clientnames: List[str], generic_dir: str):
    """
    Run the simulation via FeatureCloud's Testing module.

    Args:
        data_path: The path to the data generally. All clients data is in that folder
        clientnames: The path to the clientfolders from the data_path.
        generic_dir: The path to the generic directory. The content of this folder is used
            in all clients.
    """
    # build the image
    docker_client = docker.from_env()
    docker_client.images.build(path='.', tag='rforest:latest')

    # stop and start the controller
    controller.stop(name='')
    print(f'Starting controller with data_dir={data_path}')
    controller.start(name=controller.DEFAULT_CONTROLLER_NAME,
                        port=8000,
                        data_dir=data_path,
                        controller_image='',
                        with_gpu=False,
                        mount='',
                        blockchain_address='')
    # wait some time until the controller is online
    time.sleep(10)

    # start the test
    test.start(controller_host='http://localhost:8000',
                client_dirs=','.join(clientnames),
                generic_dir=generic_dir,
                app_image='rforest',
                channel='local',
                query_interval=3,
                download_results='')

    # inform the user where they can see the test
    print('You can follow the test along at https://featurecloud.ai/development/test')
    print('You need to login to be able to see the test')

def run_simulation_native(clientpaths: List[str], outputfolders: List[str], generic_dir: str):
    """
    Using the helpers from src.helper.localfedlearningsimulator.py, run the simulation natively.

    Args:
        clientpaths: The path to the clientfolders. Recommendation to use absolute paths.
        outputfolders: The path to the outputfolders. Recommendation to use absolute paths.
        generic_dir: The path to the generic directory. The content of this folder is copied to
            each client folder. If the file in the generic folder already exists in the
            client folder, it is not copied. Folders in the generic folder are not copied.
            The first clientfolder is used as the coordinator.
    """
    # create the wrapper
    wrapper = LocalFedLearningSimulationWrapper(clientfolders=clientpaths,
                                                outputfolders=outputfolders,
                                                generic_dir=generic_dir)

    # run the simulation
    # start all clients as threads
    threads = []
    for idx, local_client in enumerate(wrapper.clients):
        threads.append(threading.Thread(
            target=main,
            args=(local_client, clientpaths[idx], outputfolders[idx])))
        threads[-1].start()

    # done, perform cleanup
    for thread in threads:
        thread.join()
    wrapper.cleanup_created_files()

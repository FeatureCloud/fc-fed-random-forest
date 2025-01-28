"""
A class to use to simulate a federated learning environment locally. This class
adheres to the ProtocolFedLearning protocol of this project. This class represents
a single client in a federated learning environment.
#TODO: describe the classes
"""
from threading import Lock
import os
import time
import shutil
from typing import Any, Optional, List
from .protocolfedlearningclass import ProtocolFedLearning

class SharedDictionary:
    """
    A helper class that allows multiple instances to share a single dictionary concurrently.
    """

    def __init__(self):
        self._shared_dict = {}
        self._lock = Lock()

    def get(self, key):
        """
        Gets the value associated with the given key from the shared dictionary.
        """
        with self._lock:
            return self._shared_dict.get(key)

    def get_all_values(self):
        """
        Gets all key-value pairs from the shared dictionary.
        """
        with self._lock:
            return list(self._shared_dict.values())

    def set(self, key, value):
        """
        Sets the value associated with the given key in the shared dictionary.
        """
        with self._lock:
            self._shared_dict[key] = value

    def delete(self, key):
        """
        Removes the key-value pair from the shared dictionary.
        """
        with self._lock:
            if key in self._shared_dict:
                del self._shared_dict[key]

class LocalFedLearningSimulator(ProtocolFedLearning):
    """
    The simulator class, representing a single client in a federated learning environment.
    """
    def __init__(self,
                 is_coordinator: bool,
                 client_id: int,
                 num_clients: int,
                 inputfolder: str,
                 outputfolder: str,
                 shared_dict: SharedDictionary):
        """
        The constructor, the following parameters are required:

        Args:
            is_coordinator: Boolean variable, if True the this client
                represents the coordinator. False otherwise.
            client_id: The id of the client
            num_clients: The total number of clients in the federated learning
                environment. This is needed to correctly gather data,
                specifically to know how many data packets to wait for
            inputfolder: The path to the input folder containing the data for this client
            outputfolder: The path to the output folder to store the results
            shared_dict: A shared dictionary to store the data.
                The other clients (instances of this class) receive the same
                dictionary. The client uses it's client_id as key to access the data.
        """
        self.coordinator = is_coordinator
        self.client_id = client_id
        if client_id == 'global':
            raise ValueError("The client_id 'global' is reserved for global data")
        self.num_clients = num_clients
        self.inputfolder = inputfolder
        self.outputfolder = outputfolder
        self.shared_dict = shared_dict

    @property
    def is_coordinator(self):
        return self.coordinator

    def send_data_to_coordinator(self,
                                 data,
                                 send_to_self=True,
                                 use_smpc=False,
                                 use_dp=False,
                                 memo=None):
        self.shared_dict.set(self.client_id, data)

    def gather_data(self,
                    is_json: bool=False,
                    use_smpc: bool=False,
                    use_dp: bool=False,
                    memo: Optional[Any]=None):
        if not self.coordinator:
            raise ValueError("Only the coordinator can gather data")
        # wait for enough data to arrive in a loop
        while True:
            data_packets = self.shared_dict.get_all_values()
            if len(data_packets) == self.num_clients:
                return data_packets
            time.sleep(5)

    def broadcast_data(self,
                       data: Any,
                       send_to_self: bool = True,
                       use_dp: bool = False,
                       memo: Optional[Any] = None) -> None:
        if not self.coordinator:
            raise ValueError("Only the coordinator can broadcast data")
        self.shared_dict.set('global', data)

    def await_data(self,
                     n: int = 1,
                     unwrap: bool = True,
                     is_json: bool = False,
                     use_dp: bool = False,
                     use_smpc: bool = False,
                     memo: Optional[Any] = None):

        if n != 1:
            raise ValueError("This method only supports n=1 right now")
        while True:
            data = self.shared_dict.get('global')
            if not data:
                time.sleep(5)
        # unwrap is not needed as we never wrap the data in the first place
        return data

class LocalFedLearningSimulationWrapper:
    """
    A wrapper class to simulate a federated learning environment locally.
    This class is used to create multiple instances of LocalFedLearningSimulator
    and run them concurrently.
    """
    def __init__(self,
                 clientfolders: List[str],
                 outputfolders: List[str],
                 generic_dir: str) -> None:
        """
        The following arguments are required:

        Args:
            clientfolders: A list of paths to the client folders containing the data
            generic_dir: The path to the generic directory. The content of this folder is copied to
                each client folder. If the file in the generic folder already exists in the
                client folder, it is not copied. Folders in the generic folder are not copied.
                The first clientfolder is used as the coordinator.
        """
        # checks
        if len(clientfolders) < 2:
            raise ValueError("At least two client folders are required")
        if len(clientfolders) != len(outputfolders):
            raise ValueError("The number of client folders and output folders must be the same")
        # basic variables
        self.num_clients = len(clientfolders)
        self.shared_dict = SharedDictionary()

        # copy files from the generic folder to each client folder
        for clientfolder in clientfolders:
            # copy files from the generic folder to the client folder
            for root, _, files in os.walk(generic_dir):
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(clientfolder, file)
                    if not os.path.exists(dst_file):
                        shutil.copy(src_file, dst_file)

        # create the client instances
        self.clients: List[LocalFedLearningSimulator] = []
        for i, clientfolder in enumerate(clientfolders):
            is_coordinator = i == 0
            client = LocalFedLearningSimulator(is_coordinator=is_coordinator,
                                               client_id=i,
                                               num_clients=self.num_clients,
                                               inputfolder=clientfolder,
                                               outputfolder=outputfolders[i],
                                               shared_dict=self.shared_dict)
            self.clients.append(client)
